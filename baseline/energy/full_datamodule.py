"""
Full-data datamodule for energy-based OSR detection.

Role definitions (strictly enforced, zero leakage):
  ID models       -> train, val, test_id, wild_id
  Covariate models-> val_cov, test_cov, wild_cov   (never train)
  Zero-shot models-> test_zs only                  (never train, val, wild)
  Human           -> train, val, test (all pillars), wild_human

Split construction order:
  1. Wild carved first from ID/covariate/human pools proportionally (wild_size total,
     wild_ratios composition). Zero-shot models contribute nothing to wild.
  2. Remainder of ID/human split by SplitRatios (train/val/test, sum=1.0).
     Remainder of covariate split goes entirely to val_cov + test_cov (no train).
     Remainder of zero-shot goes entirely to test_zs.
  3. Attacks appended to test pillars only (never train, val, wild).
  4. Each test pillar capped at test_pool_cap by uniform subsample.

Human test indices are shared across all three test pillars.
Overlap assertions verify train/val/wild disjointness at construction time.
"""

import os
import random

import numpy as np
import torch
import lightning as L
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

FAMILY_MAP = {
    'gpt2': 'GPT', 'gpt3': 'GPT',
    'chatgpt': 'ChatGPT', 'gpt4': 'ChatGPT',
    'llama-chat': 'Meta-LLaMA',
    'mpt': 'MPT', 'mpt-chat': 'MPT',
    'cohere': 'Cohere', 'cohere-chat': 'Cohere',
    'mistral': 'Mistral', 'mistral-chat': 'Mistral',
    'human': 'human',
}

SPLIT_FAMILIES = {
    "temporal":             {'GPT': 0, 'MPT': 1},
    "temporal-ablate":      {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "scone-temporal":       {'GPT': 0, 'MPT': 1, 'ChatGPT': 2, 'Meta-LLaMA': 3},
    "scone-temporal-ablate":{'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "all":                  {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3, 'Cohere': 4, 'Mistral': 5},
}

# strict role assignment per split strategy
# covariate and zero_shot are EXCLUSIVE of id — no model appears in two roles
SPLIT_ROLES = {
    "scone-temporal": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat'],
        "covariate": ['mpt-chat', 'gpt4'],
        "zero_shot": ['mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "scone-temporal-ablate": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'mpt-chat', 'gpt4'],
        "covariate": [],
        "zero_shot": ['cohere', 'cohere-chat'],
    },
    "temporal": {
        "id":        ['gpt2', 'gpt3', 'mpt'],
        "covariate": ['chatgpt', 'llama-chat'],
        "zero_shot": ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "temporal-ablate": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat'],
        "covariate": [],
        "zero_shot": ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "all": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'gpt4', 'mpt-chat',
                      'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
        "covariate": [],
        "zero_shot": [],
    },
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class SplitRatios:
    """
    Per-model train/val/test allocation ratios applied to remainder after wild carve-out.
    Applied only to ID and human pools. Must sum to 1.0.

    :param train: fraction of remainder to train.
    :param val: fraction of remainder to val.
    :param test: fraction of remainder to test.
    """
    train: float = 0.90
    val:   float = 0.05
    test:  float = 0.05

    def __post_init__(self):
        total = self.train + self.val + self.test
        assert abs(total - 1.0) < 1e-6, f"SplitRatios must sum to 1.0, got {total:.6f}"
        for name, v in vars(self).items():
            assert 0.0 < v < 1.0, f"SplitRatios.{name}={v} must be in (0, 1)"


def _proportional_wild_carve(
    pools: dict[str, np.ndarray],
    target: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Carves `target` total examples from `pools` proportionally to each pool's size.
    If total available < target, uses all available.

    :param pools: model_name -> shuffled clean index array.
    :param target: total examples to carve across all pools.
    :returns: (carved, remainder) both model_name -> index array.
    """
    total_available = sum(len(v) for v in pools.values())
    actual_target   = min(target, total_available)

    carved    = {}
    remainder = {}
    for m, idx in pools.items():
        n_take = int(actual_target * len(idx) / total_available) if total_available > 0 else 0
        carved[m]    = idx[:n_take]
        remainder[m] = idx[n_take:]
    return carved, remainder


def _split_by_ratios(
    idx: np.ndarray,
    ratios: SplitRatios,
) -> dict[str, np.ndarray]:
    """
    Splits index array into train/val/test by SplitRatios. array must be pre-shuffled.

    :param idx: pre-shuffled index array.
    :param ratios: SplitRatios instance.
    :returns: dict with keys train, val, test.
    """
    n       = len(idx)
    n_train = int(n * ratios.train)
    n_val   = int(n * ratios.val)
    assert n_train >= 1 and n_val >= 1, (
        f"pool too small ({n}) to satisfy SplitRatios — reduce wild_size or wild_ratios"
    )
    cuts = np.cumsum([n_train, n_val])
    return {"train": idx[:cuts[0]], "val": idx[cuts[0]:cuts[1]], "test": idx[cuts[1]:]}


def _cap_pool(
    idx: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random subsample to cap if pool exceeds cap. no-op if cap <= 0."""
    if cap <= 0 or len(idx) <= cap:
        return idx
    return idx[rng.choice(len(idx), size=cap, replace=False)]


class FullEnergyRAIDSubset(Dataset):
    """
    :param hf_dataset: HuggingFace dataset slice.
    :param tokenizer: pretrained tokenizer.
    :param family_to_idx: mapping from family name to classifier class index.
    :param max_length: tokenizer max length.
    :param is_unlabeled: sets label=-1, family_idx=-1 (wild pool).
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        family_to_idx: dict[str, int],
        max_length: int = 512,
        is_unlabeled: bool = False,
    ):
        self.dataset       = hf_dataset
        self.tokenizer     = tokenizer
        self.family_to_idx = family_to_idx
        self.max_length    = max_length
        self.is_unlabeled  = is_unlabeled

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example    = self.dataset[idx]
        enc        = self.tokenizer(
            example["generation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens     = enc["input_ids"].squeeze(0)
        mask       = enc["attention_mask"].squeeze(0)
        model_name = example["model"]

        if self.is_unlabeled:
            return {
                "tokens": tokens, "mask": mask,
                "label":      torch.tensor(-1, dtype=torch.long),
                "family_idx": torch.tensor(-1, dtype=torch.long),
                "group": model_name,
            }

        label      = torch.tensor(0 if model_name == "human" else 1, dtype=torch.long)
        family     = FAMILY_MAP.get(model_name, "human")
        family_idx = torch.tensor(self.family_to_idx.get(family, -1), dtype=torch.long)
        return {"tokens": tokens, "mask": mask, "label": label, "family_idx": family_idx, "group": model_name}


class FullEnergyRAIDDataModule(L.LightningDataModule):
    """
    Full-data datamodule for energy-based MGT detection on RAID.

    Strict role enforcement — zero data leakage across ID/covariate/zero-shot boundaries.
    Wild carved first from ID+covariate+human pools. Remainder split by SplitRatios
    for ID and human only. Covariate remainder -> val_cov+test_cov. ZS -> test_zs only.

    Val loaders:  [id+human, covariate_dev].
    Test loaders: [id_retention, covariate, zero_shot].

    :param tokenizer_name: HuggingFace hub identifier.
    :param split_strategy: one of 'temporal', 'temporal-ablate', 'scone-temporal',
                           'scone-temporal-ablate', 'all'.
    :param split_ratios: SplitRatios or dict(train, val, test). must sum to 1.0.
                         applied to ID and human remainder only.
    :param wild_size: target total wild pool size (integer).
    :param wild_ratios: (pi_id, pi_cov, pi_sem) composition of wild pool. must sum to 1.
    :param test_pool_cap: max examples per test pillar after attack augmentation. <=0 = no cap.
    :param batch_size: per-device batch size.
    :param max_length: tokenizer max length.
    :param attacks: include adversarial samples in test pillars only.
    :param seed: random seed.
    """

    def __init__(
        self,
        tokenizer_name: str,
        split_strategy: str = "scone-temporal",
        split_ratios: SplitRatios | dict = None,
        wild_size: int = 20_000,
        wild_ratios: tuple[float, float, float] = (0.1, 0.6, 0.3),
        test_pool_cap: int = 10_000,
        batch_size: int = 32,
        max_length: int = 512,
        attacks: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        if split_ratios is None:
            split_ratios = SplitRatios()
        elif isinstance(split_ratios, dict):
            split_ratios = SplitRatios(**split_ratios)
        assert isinstance(split_ratios, SplitRatios)
        assert split_strategy in SPLIT_FAMILIES,   f"unknown split_strategy: {split_strategy}"
        assert abs(sum(wild_ratios) - 1.0) < 1e-6, f"wild_ratios must sum to 1: {wild_ratios}"
        assert wild_size > 0,                       f"wild_size must be > 0"

        # verify no model appears in more than one role
        roles = SPLIT_ROLES[split_strategy]
        all_role_models = roles["id"] + roles["covariate"] + roles["zero_shot"]
        assert len(all_role_models) == len(set(all_role_models)), (
            f"split_strategy '{split_strategy}' has overlapping role assignments: "
            f"{[m for m in all_role_models if all_role_models.count(m) > 1]}"
        )

        self.tokenizer_name = tokenizer_name
        self.split_strategy = split_strategy
        self.split_ratios   = split_ratios
        self.wild_size      = wild_size
        self.wild_ratios    = wild_ratios
        self.test_pool_cap  = test_pool_cap
        self.batch_size     = batch_size
        self.max_length     = max_length
        self.attacks        = attacks
        self.seed           = seed

        self.id_families  = SPLIT_FAMILIES[split_strategy]
        self.n_classes    = len(self.id_families)
        self.tokenizer    = None
        self._sanity_stats = {}

        set_seed(seed)

    def prepare_data(self):
        load_dataset("liamdugan/raid", split="train")
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        rng = np.random.default_rng(self.seed)
        ds  = load_dataset("liamdugan/raid", split="train")

        models_col  = np.array(ds["model"])
        attacks_col = np.array(ds["attack"])
        clean_mask  = attacks_col == "none"

        roles      = SPLIT_ROLES[self.split_strategy]
        id_models  = set(roles["id"])
        cov_models = set(roles["covariate"])
        zs_models  = set(roles["zero_shot"])

        def cat(arrays):
            arrays = [a for a in arrays if len(a) > 0]
            return np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)

        # collect and shuffle clean indices per model
        clean_idx: dict[str, np.ndarray] = {}
        for m in set(models_col.tolist()):
            idx = np.where((models_col == m) & clean_mask)[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            clean_idx[m] = idx

        pi_id, pi_cov, pi_sem = self.wild_ratios

        # step 1 — carve wild from ID, covariate, human pools independently
        id_clean  = {m: clean_idx[m] for m in id_models  if m in clean_idx}
        cov_clean = {m: clean_idx[m] for m in cov_models if m in clean_idx}
        hum_clean = {"human": clean_idx["human"]} if "human" in clean_idx else {}

        wild_id_carved,  id_remainder  = _proportional_wild_carve(id_clean,  int(self.wild_size * pi_id),  rng)
        wild_cov_carved, cov_remainder = _proportional_wild_carve(cov_clean, int(self.wild_size * pi_cov), rng)
        wild_hum_carved, hum_remainder = _proportional_wild_carve(hum_clean, int(self.wild_size * pi_sem), rng)

        wild_pool = cat(
            list(wild_id_carved.values()) +
            list(wild_cov_carved.values()) +
            list(wild_hum_carved.values())
        )
        rng.shuffle(wild_pool)

        # step 2 — split ID and human remainders into train/val/test
        id_splits  = {m: _split_by_ratios(rem, self.split_ratios) for m, rem in id_remainder.items()}
        hum_splits = {m: _split_by_ratios(rem, self.split_ratios) for m, rem in hum_remainder.items()}

        # covariate remainder -> val_cov + test_cov (split 50/50, no train)
        cov_val_idx, cov_test_idx = [], []
        for m, rem in cov_remainder.items():
            mid = len(rem) // 2
            cov_val_idx.append(rem[:mid])
            cov_test_idx.append(rem[mid:])

        # zero-shot clean -> test_zs only
        zs_test_idx = [clean_idx[m] for m in zs_models if m in clean_idx]

        # step 3 — attacks to test pillars only
        def get_attacks(model_set):
            arrs = []
            for m in model_set:
                atk = np.where((models_col == m) & ~clean_mask)[0]
                if len(atk) > 0:
                    arrs.append(atk)
            return cat(arrs)

        id_test_atk  = get_attacks(id_models)
        cov_test_atk = get_attacks(cov_models)
        zs_test_atk  = get_attacks(zs_models)
        hum_test_atk = get_attacks({"human"})

        # assemble pools
        train_pool = cat([sp["train"] for sp in id_splits.values()] +
                         [sp["train"] for sp in hum_splits.values()])
        val_pool   = cat([sp["val"]   for sp in id_splits.values()] +
                         [sp["val"]   for sp in hum_splits.values()])
        val_cov_pool = cat(cov_val_idx)

        test_hum_clean = cat([sp["test"] for sp in hum_splits.values()])
        test_hum_pool  = cat([test_hum_clean, hum_test_atk])

        test_id_raw  = cat([sp["test"] for sp in id_splits.values()] + [id_test_atk,  test_hum_pool])
        test_cov_raw = cat(cov_test_idx + [cov_test_atk, test_hum_pool])
        test_zs_raw  = cat(zs_test_idx  + [zs_test_atk,  test_hum_pool])

        # step 4 — cap test pillars
        test_id_pool  = _cap_pool(test_id_raw,  self.test_pool_cap, rng)
        test_cov_pool = _cap_pool(test_cov_raw, self.test_pool_cap, rng)
        test_zs_pool  = _cap_pool(test_zs_raw,  self.test_pool_cap, rng)

        # overlap assertions — train/val/wild must be fully disjoint
        train_s = set(train_pool.tolist())
        val_s   = set(val_pool.tolist())
        wild_s  = set(wild_pool.tolist())
        assert not (train_s & val_s),  "overlap: train ∩ val"
        assert not (train_s & wild_s), "overlap: train ∩ wild"
        assert not (val_s   & wild_s), "overlap: val ∩ wild"

        # zero-shot leakage assertion
        zs_all_clean = set(cat(zs_test_idx).tolist())
        assert not (train_s & zs_all_clean), "leakage: zero-shot models in train"
        assert not (wild_s  & zs_all_clean), "leakage: zero-shot models in wild"

        # covariate leakage assertion
        cov_all_clean = set(cat(list(cov_remainder.values()) + list(wild_cov_carved.values())).tolist())
        assert not (train_s & cov_all_clean), "leakage: covariate models in train"

        assert len(train_pool) > 0, "train_pool is empty"
        assert len(val_pool)   > 0, "val_pool is empty"
        assert len(wild_pool)  > 0, "wild_pool is empty"

        kw = dict(tokenizer=self.tokenizer, family_to_idx=self.id_families, max_length=self.max_length)

        if stage in ("fit", None):
            self.train_ds   = FullEnergyRAIDSubset(ds.select(train_pool),   **kw)
            self.wild_ds    = FullEnergyRAIDSubset(ds.select(wild_pool),    **kw, is_unlabeled=True)
            self.val_ds     = FullEnergyRAIDSubset(ds.select(val_pool),     **kw)
            self.val_cov_ds = FullEnergyRAIDSubset(ds.select(val_cov_pool), **kw)

        if stage in ("test", None):
            self.test_id_ds  = FullEnergyRAIDSubset(ds.select(test_id_pool),  **kw)
            self.test_cov_ds = FullEnergyRAIDSubset(ds.select(test_cov_pool), **kw)
            self.test_zs_ds  = FullEnergyRAIDSubset(ds.select(test_zs_pool),  **kw)

        # record per-model stats for sanity check
        per_model_stats = {}
        for m in sorted(clean_idx.keys()):
            w = (len(wild_id_carved.get(m, [])) + len(wild_cov_carved.get(m, [])) +
                 len(wild_hum_carved.get(m, [])))
            if m in id_splits:
                sp = id_splits[m]
                role = "id"
            elif m in hum_splits:
                sp = hum_splits[m]
                role = "human"
            elif m in cov_remainder:
                sp = {"train": np.array([]), "val": np.array([]), "test": cov_remainder[m]}
                role = "covariate"
            elif m in zs_models:
                sp = {"train": np.array([]), "val": np.array([]), "test": clean_idx.get(m, np.array([]))}
                role = "zero_shot"
            else:
                continue
            per_model_stats[m] = {
                "role":  role,
                "clean": len(clean_idx[m]),
                "wild":  w,
                "train": len(sp["train"]),
                "val":   len(sp["val"]),
                "test":  len(sp["test"]),
            }

        self._sanity_stats = {
            "per_model": per_model_stats,
            "pool_sizes": {
                "train":    len(train_pool),
                "val":      len(val_pool),
                "val_cov":  len(val_cov_pool),
                "wild":     len(wild_pool),
                "test_id":  len(test_id_pool),
                "test_cov": len(test_cov_pool),
                "test_zs":  len(test_zs_pool),
            },
            "wild_composition": {
                "id":       sum(len(v) for v in wild_id_carved.values()),
                "covariate":sum(len(v) for v in wild_cov_carved.values()),
                "human":    sum(len(v) for v in wild_hum_carved.values()),
            },
        }
        self._sanity_check()

    def _sanity_check(self):
        """
        Prints per-model split counts with role labels, pool totals, wild composition,
        leakage summary, and data utilization. call after setup() to verify correctness.
        """
        s = self._sanity_stats
        print("\n" + "=" * 80)
        print(f"  SANITY CHECK — {self.split_strategy}  (wild_size={self.wild_size}, cap={self.test_pool_cap})")
        print("=" * 80)

        print(f"\n  {'model':<18} {'role':<12} {'clean':>7} {'wild':>6} {'train':>7} {'val':>6} {'test*':>7}")
        print("  " + "-" * 65)
        grand_clean = 0
        for m, c in s["per_model"].items():
            grand_clean += c["clean"]
            print(f"  {m:<18} {c['role']:<12} {c['clean']:>7} {c['wild']:>6} "
                  f"{c['train']:>7} {c['val']:>6} {c['test']:>7}")
        print("  " + "-" * 65)
        print(f"  {'TOTAL (clean)':<31} {grand_clean:>7}")
        print(f"  * test col is pre-cap clean remainder; attacks and human not reflected per-model.")

        print(f"\n  pool sizes (post-construction, post-cap):")
        for k, v in s["pool_sizes"].items():
            print(f"    {k:<12}: {v:>8}")

        wc = s["wild_composition"]
        wt = sum(wc.values())
        print(f"\n  wild composition (target {self.wild_ratios}, actual total={wt}):")
        for k, v in wc.items():
            pct = v / wt * 100 if wt > 0 else 0.0
            print(f"    {k:<12}: {v:>7}  ({pct:.1f}%)")

        used = s["pool_sizes"]["train"] + s["pool_sizes"]["val"] + s["pool_sizes"]["wild"]
        print(f"\n  utilization (train+val+wild / clean total): {used} / {grand_clean} = {used/grand_clean*100:.1f}%")
        print("  leakage assertions: PASSED (zero-shot and covariate not in train/wild)")
        print("  human test indices shared across all three test pillars.")
        print("=" * 80 + "\n")

    def train_dataloader(self):
        return {
            "id":   DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True),
            "wild": DataLoader(self.wild_ds,  batch_size=self.batch_size, shuffle=True),
        }

    def val_dataloader(self):
        return [
            DataLoader(self.val_ds,     batch_size=self.batch_size),
            DataLoader(self.val_cov_ds, batch_size=self.batch_size),
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_id_ds,  batch_size=self.batch_size),
            DataLoader(self.test_cov_ds, batch_size=self.batch_size),
            DataLoader(self.test_zs_ds,  batch_size=self.batch_size),
        ]
