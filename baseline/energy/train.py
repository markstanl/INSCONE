"""
Training loop for energy-based OSR MGT detector.
Mirrors scone/train.py structure: Fabric, cosine LR, per-epoch test, diagnostic logging.
"""

import torch
import lightning as L
from tqdm import tqdm
from collections import defaultdict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.fabric.strategies import DDPStrategy

from baseline.energy.model import EnergyDetector
from baseline.energy.datamodule import EnergyRAIDDataModule
from baseline.energy.test import test
from baseline.energy.dev import _log_energy_diagnostics, _log_per_model_energy

def train(
    tokenizer_name: str = "princeton-nlp/unsup-simcse-roberta-base",
    split_strategy: str = "temporal",
    batch_size_per_gpu: int = 32,
    epochs: int = 20,
    learning_rate: float = 2e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    alpha_contrastive: float = 1.0,
    alpha_classify: float = 1.0,
    alpha_energy: float = 0.01,
    m_in: float = -27.0,
    m_out: float = -5.0,
    freeze_embedding_layer: bool = True,
    devices: int = 1,
    train_id_budget: int = 10_000,
    wild_budget: int = 10_000,
    eval_budget: int = 5_000,
    test_budget: int = 5_000,
    attacks: bool = True,
) -> None:
    """
    :param tokenizer_name: HuggingFace hub identifier.
    :param batch_size_per_gpu: per-device batch size.
    :param epochs: total training epochs.
    :param learning_rate: peak lr.
    :param warmup_steps: linear warmup steps.
    :param weight_decay: adamw weight decay.
    :param temperature: contrastive loss temperature.
    :param alpha_contrastive: contrastive loss weight.
    :param alpha_classify: classifier CE loss weight.
    :param alpha_energy: energy margin loss weight.
    :param m_in: energy margin for ID samples (negative).
    :param m_out: energy margin for OOD samples (less negative than m_in).
    :param freeze_embedding_layer: freeze encoder embedding layer.
    :param devices: number of GPUs.
    :param train_id_budget: max labeled ID training samples.
    :param wild_budget: max unlabeled wild samples (reserved for SCONE extension).
    :param eval_budget: max validation samples.
    :param test_budget: max test samples per pillar.
    :param attacks: include adversarial samples in test pool.
    """
    torch.set_float32_matmul_precision("medium")

    fabric = L.Fabric(
        accelerator="auto",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
    )
    fabric.launch()

    dm = EnergyRAIDDataModule(
        tokenizer_name=tokenizer_name,
        train_id_budget=train_id_budget,
        wild_budget=wild_budget,
        eval_budget=eval_budget,
        test_budget=test_budget,
        batch_size=batch_size_per_gpu,
        attacks=attacks,
        split_strategy=split_strategy
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    dm.setup(stage="test")

    model = EnergyDetector(
        model_name=tokenizer_name,
        n_classes=dm.n_classes,
        temperature=temperature,
        alpha_contrastive=alpha_contrastive,
        alpha_classify=alpha_classify,
        alpha_energy=alpha_energy,
        m_in=m_in,
        m_out=m_out,
    )

    if freeze_embedding_layer:
        for name, param in model.encoder.named_parameters():
            if "emb" in name:
                param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=weight_decay,
    )

    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method("_encode")
    model.mark_forward_method("compute_loss")
    model.mark_forward_method("compute_energy")

    raw_loaders = dm.train_dataloader()
    id_loader = fabric.setup_dataloaders(raw_loaders["id"])
    # wild loader reserved for future SCONE extension — not passed to model
    _ = fabric.setup_dataloaders(raw_loaders["wild"])
    val_id_sem_loader, val_cov_loader = fabric.setup_dataloaders(*dm.val_dataloader())

    num_batches = len(id_loader)
    total_steps = epochs * num_batches - warmup_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=learning_rate / 10)

    best_auroc = 0.0
    best_epoch = -1
    diag_interval = max(1, num_batches // 3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_con = epoch_cls = epoch_eng = 0.0
        n_steps = 0

        for i, batch in tqdm(enumerate(id_loader), desc=f"Epoch {epoch}"):
            current_step = epoch * num_batches + i

            if current_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate * current_step / max(warmup_steps, 1)

            optimizer.zero_grad()

            loss, l_con, l_cls, l_eng = model.compute_loss(
                batch["tokens"], batch["mask"], batch["label"], batch["family_idx"]
            )

            if torch.isnan(loss):
                fabric.print(f"nan at step {current_step}, skipping")
                continue

            fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if current_step >= warmup_steps:
                scheduler.step()

            if i % diag_interval == 0:
                _log_energy_diagnostics(fabric, model, batch, i)

            epoch_loss += loss.item()
            epoch_con += l_con.item()
            epoch_cls += l_cls.item()
            epoch_eng += l_eng.item()
            n_steps += 1

        d = max(n_steps, 1)
        fabric.print(
            f"epoch {epoch} | loss: {epoch_loss/d:.4f} "
            f"(con: {epoch_con/d:.4f}, cls: {epoch_cls/d:.4f}, eng: {epoch_eng/d:.4f})"
        )

        # val energy gap
        model.eval()
        with torch.no_grad():
            e_id, e_ood, e_cov = [], [], []
            for batch in val_id_sem_loader:
                phi = model._encode(batch["tokens"], batch["mask"])
                e = model.compute_energy(phi)
                lbl = batch["label"]
                e_id.append(e[lbl == 1].detach())
                e_ood.append(e[lbl == 0].detach())
            for batch in val_cov_loader:
                phi = model._encode(batch["tokens"], batch["mask"])
                e_cov.append(model.compute_energy(phi).detach())

            e_id_all = torch.cat(e_id)
            e_ood_all = torch.cat(e_ood)
            e_cov_all = torch.cat(e_cov)
            fabric.print(
                f"  energy | id: {e_id_all.mean():.3f} +/- {e_id_all.std():.3f} "
                f"| ood_sem: {e_ood_all.mean():.3f} +/- {e_ood_all.std():.3f} "
                f"| ood_cov: {e_cov_all.mean():.3f} +/- {e_cov_all.std():.3f}"
            )

        if epoch % 5 == 0:
            fabric.save(f"checkpoints/energy_epoch_{epoch}.pt", {"model": model, "epoch": epoch})
            _log_per_model_energy(fabric, model, dm, fabric.device)

        metrics = test(model, dm, fabric.device)
        current_auroc = metrics.compute()["overall_auroc"].item()
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_epoch = epoch
            fabric.save("checkpoints/energy_best.pt", {
                "model": model, "epoch": epoch, "auroc": best_auroc
            })
            fabric.print(f"  new best: {best_auroc:.4f} (epoch {epoch})")

        fabric.print(metrics.generate_table())

    fabric.print(f"\ndone. best auroc: {best_auroc:.4f} at epoch {best_epoch}")
    fabric.save("checkpoints/energy_final.pt", {"model": model})


if __name__ == "__main__":
    train(
        tokenizer_name="princeton-nlp/unsup-simcse-roberta-base",
        split_strategy="all",
        batch_size_per_gpu=32,
        devices=1,
        epochs=10,
        learning_rate=2e-5,
        warmup_steps=2000,
        alpha_contrastive=1.0,
        alpha_classify=1.0,
        alpha_energy=0.01,
        m_in=-27.0,
        m_out=-5.0,
        train_id_budget=10_000,
        wild_budget=5_000,
        eval_budget=5_000,
        test_budget=5_000,
        attacks=True,
        freeze_embedding_layer=True,
    )