"""
Full-data datamodule for energy-based OSR detection.

Split philosophy:
  - train/val/test allocated first via split_ratios, applied independently per model.
  - wild pool constructed from *leftovers* after train/val/test, subject to wild_ratios
    composition (pi_id, pi_cov, pi_sem). the constraining component (whichever pool has
    fewest leftover examples relative to its target fraction) sets total wild size.
    excess leftovers from other components are discarded, not overflowed.
  - human text is treated as its own model for split_ratios purposes, then its leftover
    contributes to the pi_sem slice of the wild pool.
  - human test examples are shared (identical indices) across all three test pillars.
  - zero overlap guaranteed: each example index appears in exactly one of
    {train, val, test_machine, wild} per model.
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import lightning as L
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
    "temporal": {'GPT': 0, 'MPT': 1},
    "temporal-ablate": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "scone-temporal": {'GPT': 0, 'MPT': 1, 'ChatGPT': 2, 'Meta-LLaMA': 3},
    "scone-temporal-ablate": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "all": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3, 'Cohere': 4, 'Mistral': 5},
}

# model -> split role
SPLIT_ROLES = {
    "scone-temporal": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat'],
        "covariate": ['mpt-chat', 'gpt4'],
        "zero_shot": ['mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "scone-temporal-ablate": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'mpt-chat', 'gpt4'],
        "covariate": ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'mpt-chat', 'gpt4'],
        "zero_shot": ['cohere', 'cohere-chat'],
    },
    "temporal": {
        "id":        ['gpt2', 'gpt3', 'mpt'],
        "covariate": ['chatgpt', 'llama-chat'],
        "zero_shot": ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "temporal-ablate": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat'],
        "covariate": ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat'],
        "zero_shot": ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
    },
    "all": {
        "id":        ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'gpt4', 'mpt-chat',
                      'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
        "covariate": ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'gpt4', 'mpt-chat',
                      'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
        "zero_shot": ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'gpt4', 'mpt-chat',
                      'mistral', 'mistral-chat', 'cohere', 'cohere-chat'],
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
    Per-model allocation ratios. must sum to 1.0.

    :param train: fraction of each model's examples allocated to train.
    :param val: fraction allocated to val.
    :param test: fraction allocated to test.
    :param wild: fraction allocated to wild pool (leftovers after train/val/test).
                 wild pool is further constrained by wild_ratios composition.
    """
    train: float = 0.85
    val:   float = 0.05
    test:  float = 0.05
    wild:  float = 0.05

    def __post_init__(self):
        total = self.train + self.val + self.test + self.wild
        assert abs(total - 1.0) < 1e-6, f"split_ratios must sum to 1.0, got {total:.6f}"
        for name, v in vars(self).items():
            assert 0.0 < v < 1.0, f"split_ratios.{name}={v} must be in (0, 1)"


def _split_model_indices(
    indices: np.ndarray,
    ratios: SplitRatios,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Splits a model's index array into train/val/test/wild with no overlap.

    :param indices: all clean indices for one model.
    :param ratios: SplitRatios instance.
    :param rng: seeded numpy rng.
    :returns: dict with keys train, val, test, wild.
    """
    idx = indices.copy()
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios.train)
    n_val   = int(n * ratios.val)
    n_test  = int(n * ratios.test)
    # wild gets the remainder — at least 1 each for train/val/test
    assert n_train >= 1 and n_val >= 1 and n_test >= 1, (
        f"model has too few examples ({n}) to satisfy split_ratios"
    )
    cuts = np.cumsum([n_train, n_val, n_test])
    return {
        "train": idx[:cuts[0]],
        "val":   idx[cuts[0]:cuts[1]],
        "test":  idx[cuts[1]:cuts[2]],
        "wild":  idx[cuts[2]:],
    }


def _build_wild_pool(
    wild_leftovers: dict[str, list[np.ndarray]],
    wild_ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Constructs the wild pool from leftover indices, respecting wild_ratios composition.

    The constraining component (fewest available examples relative to its target fraction)
    sets the total wild pool size. Other components are subsampled to match.

    :param wild_leftovers: keys 'id', 'covariate', 'human'. each value is a list of
                           per-model index arrays to concatenate.
    :param wild_ratios: (pi_id, pi_cov, pi_sem). must sum to 1.
    :returns: shuffled wild pool index array.
    """
    pi_id, pi_cov, pi_sem = wild_ratios
    assert abs(pi_id + pi_cov + pi_sem - 1.0) < 1e-6, f"wild_ratios must sum to 1: {wild_ratios}"

    pools = {
        "id":       np.concatenate(wild_leftovers["id"])       if wild_leftovers["id"]       else np.array([], dtype=np.int64),
        "covariate":np.concatenate(wild_leftovers["covariate"])if wild_leftovers["covariate"]else np.array([], dtype=np.int64),
        "human":    np.concatenate(wild_leftovers["human"])    if wild_leftovers["human"]     else np.array([], dtype=np.int64),
    }
    for k, arr in pools.items():
        rng.shuffle(arr)

    fracs = {"id": pi_id, "covariate": pi_cov, "human": pi_sem}

    # find constraining component
    max_totals = {}
    for key, frac in fracs.items():
        if frac > 0:
            max_totals[key] = int(len(pools[key]) / frac)
        else:
            max_totals[key] = int(1e18)

    n_total = min(max_totals.values())
    assert n_total > 0, "wild pool is empty — check that split_ratios.wild > 0 and models have sufficient examples"

    slices = {}
    for key, frac in fracs.items():
        n_take = int(n_total * frac)
        available = len(pools[key])
        assert n_take <= available, (
            f"wild pool construction: need {n_take} {key} examples but only {available} available. "
            f"reduce split_ratios.wild or adjust wild_ratios."
        )
        slices[key] = pools[key][:n_take]

    combined = np.concatenate(list(slices.values()))
    rng.shuffle(combined)
    return combined, {k: len(v) for k, v in slices.items()}


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
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.family_to_idx = family_to_idx
        self.max_length = max_length
        self.is_unlabeled = is_unlabeled

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        enc = self.tokenizer(
            example["generation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = enc["input_ids"].squeeze(0)
        mask   = enc["attention_mask"].squeeze(0)
        model_name = example["model"]

        if self.is_unlabeled:
            return {
                "tokens": tokens, "mask": mask,
                "label": torch.tensor(-1, dtype=torch.long),
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

    Uses the entire dataset with no fixed budget caps. Each model's examples are
    split independently by split_ratios (train/val/test/wild). The wild pool is then
    constructed from wild leftovers subject to wild_ratios composition constraints.

    Val loaders: [id+semantic, covariate_dev].
    Test loaders: [id_retention, wild_memorization, zero_shot].
      Human test examples are shared across all three test pillars.

    :param tokenizer_name: HuggingFace hub identifier.
    :param split_strategy: one of 'temporal', 'temporal-ablate', 'scone-temporal',
                           'scone-temporal-ablate', 'all'.
    :param split_ratios: SplitRatios instance or dict with keys train/val/test/wild.
    :param wild_ratios: (pi_id, pi_cov, pi_sem) composition of wild pool. must sum to 1.
    :param batch_size: per-device batch size.
    :param max_length: tokenizer max length.
    :param attacks: include adversarial samples in test pool.
    :param seed: random seed.
    """

    def __init__(
        self,
        tokenizer_name: str,
        split_strategy: str = "scone-temporal",
        split_ratios: SplitRatios | dict = None,
        wild_ratios: tuple[float, float, float] = (0.1, 0.6, 0.3),
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
        assert split_strategy in SPLIT_FAMILIES, f"unknown split_strategy: {split_strategy}"
        assert abs(sum(wild_ratios) - 1.0) < 1e-6, f"wild_ratios must sum to 1: {wild_ratios}"

        self.tokenizer_name = tokenizer_name
        self.split_strategy = split_strategy
        self.split_ratios   = split_ratios
        self.wild_ratios    = wild_ratios
        self.batch_size     = batch_size
        self.max_length     = max_length
        self.attacks        = attacks
        self.seed           = seed

        self.id_families = SPLIT_FAMILIES[split_strategy]
        self.n_classes   = len(self.id_families)
        self.tokenizer   = None
        self._sanity_stats: dict = {}

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

        roles = SPLIT_ROLES[self.split_strategy]
        id_models  = roles["id"]
        cov_models = roles["covariate"]
        zs_models  = roles["zero_shot"]

        all_machine_models = list({m for m in models_col if m != "human"})

        # per-model splits (clean only for train/val/wild; test gets attacks if enabled)
        per_model: dict[str, dict[str, np.ndarray]] = {}

        for m in all_machine_models + ["human"]:
            clean_idx = np.where((models_col == m) & clean_mask)[0]
            if len(clean_idx) == 0:
                continue
            per_model[m] = _split_model_indices(clean_idx, self.split_ratios, rng)

            if self.attacks and m != "human":
                atk_idx = np.where((models_col == m) & ~clean_mask)[0]
                if len(atk_idx) > 0:
                    # append attack samples to test slice only
                    per_model[m]["test"] = np.concatenate([per_model[m]["test"], atk_idx])

        # accumulate pools by role
        train_id_idx, train_hum_idx = [], []
        val_id_idx,   val_hum_idx   = [], []
        test_id_idx, test_cov_idx, test_zs_idx, test_hum_idx = [], [], [], []
        wild_leftovers: dict[str, list] = {"id": [], "covariate": [], "human": []}

        for m, splits in per_model.items():
            if m == "human":
                train_hum_idx.append(splits["train"])
                val_hum_idx.append(splits["val"])
                test_hum_idx.append(splits["test"])
                wild_leftovers["human"].append(splits["wild"])
            elif m in id_models:
                train_id_idx.append(splits["train"])
                val_id_idx.append(splits["val"])
                test_id_idx.append(splits["test"])
                wild_leftovers["id"].append(splits["wild"])
                if m in cov_models:
                    test_cov_idx.append(splits["test"])
                    wild_leftovers["covariate"].append(splits["wild"])
                if m in zs_models:
                    test_zs_idx.append(splits["test"])
            else:
                # OOD-only models (covariate or zero-shot, not in id_models)
                if m in cov_models:
                    test_cov_idx.append(splits["test"])
                    if m not in id_models:
                        wild_leftovers["covariate"].append(splits["wild"])
                if m in zs_models:
                    test_zs_idx.append(splits["test"])

        wild_pool, wild_composition = _build_wild_pool(wild_leftovers, self.wild_ratios, rng)

        def cat(lists): return np.concatenate(lists) if lists else np.array([], dtype=np.int64)

        train_pool = cat(train_id_idx + train_hum_idx)
        val_pool   = cat(val_id_idx + val_hum_idx)
        test_hum   = cat(test_hum_idx)

        test_id_pool  = cat(test_id_idx  + [test_hum])
        test_cov_pool = cat(test_cov_idx + [test_hum])
        test_zs_pool  = cat(test_zs_idx  + [test_hum])

        assert len(train_pool) > 0, "train_pool is empty"
        assert len(val_pool)   > 0, "val_pool is empty"
        assert len(test_hum)   > 0, "test human pool is empty"
        assert len(wild_pool)  > 0, "wild_pool is empty"

        # overlap check — train/val/wild must be disjoint (test shares human)
        train_s = set(train_pool.tolist())
        val_s   = set(val_pool.tolist())
        wild_s  = set(wild_pool.tolist())
        assert not (train_s & val_s),   "overlap: train ∩ val"
        assert not (train_s & wild_s),  "overlap: train ∩ wild"
        assert not (val_s   & wild_s),  "overlap: val ∩ wild"

        kw = dict(tokenizer=self.tokenizer, family_to_idx=self.id_families, max_length=self.max_length)

        if stage in ("fit", None):
            self.train_ds    = FullEnergyRAIDSubset(ds.select(train_pool), **kw)
            self.wild_ds     = FullEnergyRAIDSubset(ds.select(wild_pool),  **kw, is_unlabeled=True)
            self.val_ds      = FullEnergyRAIDSubset(ds.select(val_pool),   **kw)
            self.val_cov_ds  = FullEnergyRAIDSubset(ds.select(cat(val_id_idx)), **kw)

        if stage in ("test", None):
            self.test_id_ds  = FullEnergyRAIDSubset(ds.select(test_id_pool),  **kw)
            self.test_cov_ds = FullEnergyRAIDSubset(ds.select(test_cov_pool), **kw)
            self.test_zs_ds  = FullEnergyRAIDSubset(ds.select(test_zs_pool),  **kw)

        self._sanity_stats = {
            "per_model": {m: {k: len(v) for k, v in splits.items()} for m, splits in per_model.items()},
            "pool_sizes": {
                "train": len(train_pool),
                "val":   len(val_pool),
                "wild":  len(wild_pool),
                "test_id":  len(test_id_pool),
                "test_cov": len(test_cov_pool),
                "test_zs":  len(test_zs_pool),
            },
            "wild_composition": wild_composition,
        }
        self._sanity_check()

    def _sanity_check(self):
        """
        Prints per-model split counts, pool totals, wild composition, and
        a rough data-utilization summary. call after setup() to verify correctness.
        """
        s = self._sanity_stats
        print("\n" + "=" * 70)
        print(f"  SANITY CHECK — {self.split_strategy}")
        print("=" * 70)

        print(f"\n{'model':<20} {'total':>7} {'train':>7} {'val':>6} {'test':>6} {'wild':>6}")
        print("-" * 57)
        grand_total = 0
        for m, counts in sorted(s["per_model"].items()):
            total = sum(counts.values())
            grand_total += total
            print(f"  {m:<18} {total:>7} {counts['train']:>7} {counts['val']:>6} {counts['test']:>6} {counts['wild']:>6}")
        print("-" * 57)
        print(f"  {'TOTAL (clean)':<18} {grand_total:>7}")

        print(f"\n  pool sizes (post-construction):")
        for k, v in s["pool_sizes"].items():
            print(f"    {k:<12}: {v:>7}")

        wc = s["wild_composition"]
        wild_total = sum(wc.values())
        print(f"\n  wild composition (target {self.wild_ratios}):")
        for k, v in wc.items():
            pct = v / wild_total * 100 if wild_total > 0 else 0
            print(f"    {k:<12}: {v:>6}  ({pct:.1f}%)")

        used = s["pool_sizes"]["train"] + s["pool_sizes"]["val"] + s["pool_sizes"]["wild"]
        print(f"\n  data utilization (train+val+wild / clean total): {used}/{grand_total} = {used/grand_total*100:.1f}%")
        print("  note: test slices reuse machine+human subsets — not double-counted above.")
        print("=" * 70 + "\n")

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