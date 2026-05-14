"""
Datamodule for energy-based OSR detection.
Mirrors RAID_DM.py split strategies: temporal, temporal-ablate, all.
Provides dev_cov loader for proxy covariate validation.
"""

import os
import random
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

# family indices per split strategy — only ID families get classifier labels
SPLIT_FAMILIES = {
    "temporal": {'GPT': 0, 'MPT': 1},
    "temporal-ablate": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "scone-temporal": {'GPT': 0, 'MPT': 1, 'ChatGPT': 2, 'Meta-LLaMA': 3},
    "scone-temporal-ablate": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3},
    "all": {'GPT': 0, 'ChatGPT': 1, 'Meta-LLaMA': 2, 'MPT': 3, 'Cohere': 4, 'Mistral': 5},
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class EnergyRAIDSubset(Dataset):
    """
    :param hf_dataset: HuggingFace dataset slice.
    :param tokenizer: pretrained tokenizer.
    :param family_to_idx: mapping from family name to classifier class index.
    :param max_length: tokenizer max length.
    :param is_unlabeled: sets label=-1, family_idx=-1 (wild pool).
    """

    def __init__(self, hf_dataset, tokenizer, family_to_idx: dict[str, int], max_length: int = 512,
                 is_unlabeled: bool = False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.family_to_idx = family_to_idx
        self.max_length = max_length
        self.is_unlabeled = is_unlabeled

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        tokens_raw = self.tokenizer(
            example["generation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = tokens_raw["input_ids"].squeeze(0)
        mask = tokens_raw["attention_mask"].squeeze(0)
        model_name = example["model"]

        if self.is_unlabeled:
            return {
                "tokens": tokens, "mask": mask,
                "label": torch.tensor(-1, dtype=torch.long),
                "family_idx": torch.tensor(-1, dtype=torch.long),
                "group": model_name,
            }

        label = torch.tensor(0 if model_name == "human" else 1, dtype=torch.long)
        family = FAMILY_MAP.get(model_name, "human")
        family_idx = torch.tensor(self.family_to_idx.get(family, -1), dtype=torch.long)
        return {"tokens": tokens, "mask": mask, "label": label, "family_idx": family_idx, "group": model_name}


class EnergyRAIDDataModule(L.LightningDataModule):
    """
    Datamodule for energy-based MGT detection on RAID.
    Supports temporal, temporal-ablate, and all split strategies.

    Val returns two loaders: [id+semantic, covariate_dev].
    Test returns three loaders: [id_retention, wild_memorization, zero_shot].
    Wild loader provided but unused by base energy model (reserved for SCONE extension).

    :param tokenizer_name: HuggingFace hub identifier.
    :param split_strategy: one of 'temporal', 'temporal-ablate', 'all'.
    :param train_id_budget: max labeled ID training samples.
    :param wild_budget: max unlabeled wild samples.
    :param wild_ratios: (pi_id, pi_cov, pi_sem) — fractions of wild pool. must sum to 1.
    :param eval_budget: max validation samples.
    :param dev_budget: max covariate dev samples for proxy selection.
    :param test_budget: max test samples per pillar.
    :param batch_size: per-device batch size.
    :param max_length: tokenizer max length.
    :param attacks: include adversarial samples in test pool.
    :param seed: random seed.
    """

    def __init__(
            self,
            tokenizer_name: str,
            split_strategy: str = "temporal",
            train_id_budget: int = 10_000,
            wild_budget: int = 10_000,
            wild_ratios: tuple = (0.1, 0.6, 0.3),
            eval_budget: int = 5_000,
            dev_budget: int = 5_000,
            test_budget: int = 5_000,
            batch_size: int = 32,
            max_length: int = 512,
            attacks: bool = True,
            seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = None
        assert split_strategy in SPLIT_FAMILIES, f"unknown split_strategy: {split_strategy}"
        assert abs(sum(wild_ratios) - 1.0) < 1e-6, f"wild_ratios must sum to 1: {wild_ratios}"
        self.id_families = SPLIT_FAMILIES[split_strategy]
        self.n_classes = len(self.id_families)
        set_seed(seed)

    def prepare_data(self):
        load_dataset("liamdugan/raid", split="train")
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    def setup(self, stage: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        rng = np.random.default_rng(self.hparams.seed)
        ds = load_dataset("liamdugan/raid", split="train")

        models = np.array(ds["model"])
        attacks_col = np.array(ds["attack"])
        clean_mask = attacks_col == "none"

        if self.hparams.split_strategy == "temporal":
            id_models = ['gpt2', 'gpt3', 'mpt']
            ood_covariate_models = ['chatgpt', 'llama-chat']
            zero_shot_models = ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat']

        elif self.hparams.split_strategy == "temporal-ablate":
            id_models = ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat']
            ood_covariate_models = id_models
            zero_shot_models = ['gpt4', 'mpt-chat', 'mistral', 'mistral-chat', 'cohere', 'cohere-chat']

        elif self.hparams.split_strategy == "scone-temporal":
            id_models = ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat']
            ood_covariate_models = ['mpt-chat', 'gpt4']
            zero_shot_models = ['mistral', 'mistral-chat', 'cohere', 'cohere-chat']

        elif self.hparams.split_strategy == "scone-temporal-ablate":
            id_models = ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'mpt-chat', 'gpt4']
            ood_covariate_models = id_models
            zero_shot_models = ['cohere', 'cohere-chat']

        elif self.hparams.split_strategy == "all":
            id_models = ['gpt2', 'gpt3', 'mpt', 'chatgpt', 'llama-chat', 'gpt4', 'mpt-chat', 'mistral', 'mistral-chat',
                         'cohere', 'cohere-chat']
            ood_covariate_models = id_models
            zero_shot_models = id_models

        id_clean = np.where(np.isin(models, id_models) & clean_mask)[0]
        ood_sem_clean = np.where((models == "human") & clean_mask)[0]
        ood_cov_clean = np.where(np.isin(models, ood_covariate_models) & clean_mask)[0]
        zs_clean = np.where(np.isin(models, zero_shot_models) & clean_mask)[0]

        for arr in [id_clean, ood_sem_clean, ood_cov_clean, zs_clean]:
            rng.shuffle(arr)

        print(
            f"pool sizes (clean): id={len(id_clean)}, cov={len(ood_cov_clean)}, zs={len(zs_clean)}, human={len(ood_sem_clean)}")

        used_id, used_sem, used_cov, used_zs = 0, 0, 0, 0
        max_hum = len(ood_sem_clean) // 3

        # train
        train_id = id_clean[used_id: used_id + self.hparams.train_id_budget];
        used_id += len(train_id)
        train_hum = ood_sem_clean[used_sem: used_sem + min(self.hparams.train_id_budget, max_hum)];
        used_sem += len(train_hum)
        train_pool = np.concatenate([train_id, train_hum])

        # wild — composition controlled by wild_ratios (pi_id, pi_cov, pi_sem)
        pi_id, pi_cov, pi_sem = self.hparams.wild_ratios
        n_wild_id = int(self.hparams.wild_budget * pi_id)
        n_wild_cov = int(self.hparams.wild_budget * pi_cov)
        n_wild_sem = self.hparams.wild_budget - n_wild_id - n_wild_cov
        wild_id_samples = id_clean[used_id: used_id + n_wild_id];
        used_id += len(wild_id_samples)
        wild_cov = ood_cov_clean[used_cov: used_cov + n_wild_cov];
        used_cov += len(wild_cov)
        wild_sem = ood_sem_clean[used_sem: used_sem + n_wild_sem];
        used_sem += len(wild_sem)
        wild_pool = np.concatenate([wild_id_samples, wild_cov, wild_sem])

        # val id+sem
        val_id = id_clean[used_id: used_id + self.hparams.eval_budget // 2];
        used_id += len(val_id)
        val_sem = ood_sem_clean[used_sem: used_sem + self.hparams.eval_budget // 2];
        used_sem += len(val_sem)
        val_pool = np.concatenate([val_id, val_sem])

        # dev cov (proxy selection)
        dev_cov = ood_cov_clean[used_cov: used_cov + self.hparams.dev_budget // 2];
        used_cov += len(dev_cov)

        # test
        test_size = self.hparams.test_budget // 2
        if self.hparams.attacks:
            id_atk = np.where(np.isin(models, id_models) & ~clean_mask)[0]
            cov_atk = np.where(np.isin(models, ood_covariate_models) & ~clean_mask)[0]
            zs_atk = np.where(np.isin(models, zero_shot_models) & ~clean_mask)[0]
            sem_atk = np.where((models == "human") & ~clean_mask)[0]
            test_id_pool = np.concatenate([id_clean[used_id:], id_atk])
            test_cov_pool = np.concatenate([ood_cov_clean[used_cov:], cov_atk])
            test_zs_pool = np.concatenate([zs_clean[used_zs:], zs_atk])
            test_sem_pool = np.concatenate([ood_sem_clean[used_sem:], sem_atk])
        else:
            test_id_pool = id_clean[used_id:]
            test_cov_pool = ood_cov_clean[used_cov:]
            test_zs_pool = zs_clean[used_zs:]
            test_sem_pool = ood_sem_clean[used_sem:]

        for arr in [test_id_pool, test_cov_pool, test_zs_pool, test_sem_pool]:
            rng.shuffle(arr)

        test_id = test_id_pool[:test_size]
        test_cov = test_cov_pool[:test_size]
        test_zs = test_zs_pool[:test_size]
        test_sem = test_sem_pool[:self.hparams.test_budget]

        assert len(train_id) > 0, f"train_id exhausted"
        assert len(val_id) > 0, f"val_id exhausted"
        assert len(val_sem) > 0, f"val_sem exhausted"
        assert len(dev_cov) > 0, f"dev_cov exhausted"

        kw = dict(tokenizer=self.tokenizer, family_to_idx=self.id_families, max_length=self.hparams.max_length)

        if stage in ('fit', None):
            self.train_ds = EnergyRAIDSubset(ds.select(train_pool), **kw)
            self.wild_ds = EnergyRAIDSubset(ds.select(wild_pool), **kw, is_unlabeled=True)
            self.val_ds = EnergyRAIDSubset(ds.select(val_pool), **kw)
            self.val_cov_ds = EnergyRAIDSubset(ds.select(dev_cov), **kw)

        if stage in ('test', None):
            self.test_id_ds = EnergyRAIDSubset(ds.select(np.concatenate([test_id, test_sem])), **kw)
            self.test_cov_ds = EnergyRAIDSubset(ds.select(np.concatenate([test_cov, test_sem])), **kw)
            self.test_zs_ds = EnergyRAIDSubset(ds.select(np.concatenate([test_zs, test_sem])), **kw)

    def train_dataloader(self):
        return {
            "id": DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True),
            "wild": DataLoader(self.wild_ds, batch_size=self.hparams.batch_size, shuffle=True),
        }

    def val_dataloader(self):
        return [
            DataLoader(self.val_ds, batch_size=self.hparams.batch_size),
            DataLoader(self.val_cov_ds, batch_size=self.hparams.batch_size),
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_id_ds, batch_size=self.hparams.batch_size),
            DataLoader(self.test_cov_ds, batch_size=self.hparams.batch_size),
            DataLoader(self.test_zs_ds, batch_size=self.hparams.batch_size),
        ]