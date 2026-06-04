"""
Training loop for SCONE-extended energy-based MGT detector using full dataset.
Swaps EnergyRAIDDataModule → FullEnergyRAIDDataModule; all other logic identical.
"""

import argparse
import torch
import lightning as L
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.fabric.strategies import DDPStrategy

from inscone.inscone_energy.model import INSCONEEnergyDetector, SCONEEnergyDetector
from baseline.energy.full_datamodule import FullEnergyRAIDDataModule, SplitRatios
from baseline.energy.test import test
from baseline.energy.dev import _log_energy_diagnostics, _log_per_model_energy


def train(
    tokenizer_name: str = "princeton-nlp/unsup-simcse-roberta-base",
    split_strategy: str = "scone-temporal",
    batch_size_per_gpu: int = 32,
    epochs: int = 20,
    learning_rate: float = 2e-5,
    warmup_steps: int = 2000,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    alpha_contrastive: float = 1.0,
    alpha_classify: float = 1.0,
    alpha_energy: float = 0.001,
    lambda_scone: float = 0.01,
    m_in: float = -5.0,
    m_out: float = -1.0,
    m_in_wild: float | None = None,
    m_out_wild: float | None = None,
    scone_warmup_epochs: int = 2,
    freeze_embedding_layer: bool = True,
    devices: int = 1,
    split_ratios: SplitRatios | dict = None,
    wild_size: int = 20_000,
    wild_ratios: tuple[float, float, float] = (0.1, 0.6, 0.3),
    test_pool_cap: int = 10_000,
    attacks: bool = True,
    buffer: float = 0.0,
    seed: int = 42,
    silent: bool = False,
    checkpoint_name: str = "scone_energy_full",
    model_type: str = "inscone",
) -> None:
    """
    :param split_strategy: one of 'scone-temporal', 'scone-temporal-ablate', 'all'.
    :param split_ratios: SplitRatios or dict with keys train/val/test. must sum to 1.0.
    :param wild_size: target total wild pool size (integer).
    :param wild_ratios: (π_id, π_c, π_s) composition of wild pool.
    :param test_pool_cap: max examples per test pillar after attack augmentation.
    :param m_in: η — energy margin for labeled ID samples.
    :param m_out: energy margin for labeled OOD (human) samples.
    :param m_in_wild: energy target for proximal wild samples. defaults to m_in.
    :param m_out_wild: energy target for distal wild samples. defaults to m_out.
    :param lambda_scone: weight for wild margin loss.
    :param scone_warmup_epochs: epochs before wild loss activates.
    :param model_type: 'inscone' or 'scone'.
    :param silent: suppress per-step diagnostics; only print test table.
    :param checkpoint_name: base name for saved checkpoints.
    """
    if split_ratios is None:
        split_ratios = SplitRatios()
    elif isinstance(split_ratios, dict):
        split_ratios = SplitRatios(**split_ratios)

    torch.set_float32_matmul_precision("medium")

    fabric = L.Fabric(
        accelerator="auto",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
    )
    fabric.launch()
    fabric.print(
        f"config | split={split_strategy} lambda_scone={lambda_scone} "
        f"m_in={m_in} m_out={m_out} seed={seed}\n"
        f"  split_ratios | train={split_ratios.train} val={split_ratios.val} test={split_ratios.test}\n"
        f"  wild_size={wild_size}  wild_ratios={wild_ratios}  test_pool_cap={test_pool_cap}"
    )

    dm = FullEnergyRAIDDataModule(
        tokenizer_name=tokenizer_name,
        split_strategy=split_strategy,
        split_ratios=split_ratios,
        wild_size=wild_size,
        wild_ratios=wild_ratios,
        test_pool_cap=test_pool_cap,
        batch_size=batch_size_per_gpu,
        attacks=attacks,
        seed=seed,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    dm.setup(stage="test")

    ModelCls = INSCONEEnergyDetector if model_type == "inscone" else SCONEEnergyDetector

    model = ModelCls(
        model_name=tokenizer_name,
        n_classes=dm.n_classes,
        temperature=temperature,
        alpha_contrastive=alpha_contrastive,
        alpha_classify=alpha_classify,
        alpha_energy=alpha_energy,
        lambda_scone=lambda_scone,
        m_in=m_in,
        m_out=m_out,
        m_in_wild=m_in_wild,
        m_out_wild=m_out_wild,
        scone_warmup_epochs=scone_warmup_epochs,
        wild_ratios=wild_ratios,
        buffer=buffer,
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

    raw_loaders       = dm.train_dataloader()
    id_loader         = fabric.setup_dataloaders(raw_loaders["id"])
    wild_loader       = fabric.setup_dataloaders(raw_loaders["wild"], use_distributed_sampler=False)
    val_id_sem_loader = fabric.setup_dataloaders(dm.val_dataloader()[0])

    num_batches = len(id_loader)
    total_steps = epochs * num_batches - warmup_steps
    scheduler   = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=learning_rate / 10)

    diag_interval = max(1, num_batches // 3)
    wild_iter     = iter(wild_loader)

    for epoch in range(epochs):
        model.train()
        model.current_epoch = epoch
        epoch_loss = epoch_con = epoch_cls = epoch_eng = epoch_scone = 0.0
        n_steps = 0

        for i, id_batch in tqdm(enumerate(id_loader), desc=f"Epoch {epoch}"):
            current_step = epoch * num_batches + i

            if current_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate * current_step / max(warmup_steps, 1)

            try:
                wild_batch = next(wild_iter)
            except StopIteration:
                wild_iter  = iter(wild_loader)
                wild_batch = next(wild_iter)

            optimizer.zero_grad()
            loss, l_con, l_cls, l_eng, l_scone = model.compute_loss(
                id_batch["tokens"], id_batch["mask"],
                id_batch["label"],  id_batch["family_idx"],
                wild_tokens=wild_batch["tokens"], wild_mask=wild_batch["mask"],
            )

            if torch.isnan(loss):
                if not silent:
                    fabric.print(f"nan at step {current_step}, skipping")
                continue

            fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if current_step >= warmup_steps:
                scheduler.step()

            if not silent and i % diag_interval == 0:
                _log_energy_diagnostics(fabric, model, id_batch, i)

            epoch_loss  += loss.item()
            epoch_con   += l_con.item()
            epoch_cls   += l_cls.item()
            epoch_eng   += l_eng.item()
            epoch_scone += l_scone.item()
            n_steps += 1

        d = max(n_steps, 1)
        if not silent:
            fabric.print(
                f"epoch {epoch} | loss: {epoch_loss/d:.4f} "
                f"(con: {epoch_con/d:.4f}, cls: {epoch_cls/d:.4f}, "
                f"eng: {epoch_eng/d:.4f}, scone: {epoch_scone/d:.4f})"
            )

        model.eval()
        with torch.no_grad():
            e_id, e_ood_sem = [], []
            for batch in val_id_sem_loader:
                phi = model._encode(batch["tokens"], batch["mask"])
                e   = model.compute_energy(phi)
                lbl = batch["label"]
                e_id.append(e[lbl == 1].detach())
                e_ood_sem.append(e[lbl == 0].detach())

            e_id_all  = torch.cat(e_id)
            e_ood_all = torch.cat(e_ood_sem)

            if not silent:
                fabric.print(
                    f"  energy | id: {e_id_all.mean():.3f} ± {e_id_all.std():.3f} "
                    f"| ood_sem: {e_ood_all.mean():.3f} ± {e_ood_all.std():.3f}"
                )

            emp_thresh = torch.quantile(e_id_all.float(), 0.95) if len(e_id_all) > 0 else torch.tensor(0.0)
            model.threshold_fpr95.fill_(emp_thresh.item())

        if not silent and epoch % 5 == 0:
            fabric.save(f"checkpoints/{checkpoint_name}_epoch_{epoch}.pt", {"model": model, "epoch": epoch})
            _log_per_model_energy(fabric, model, dm, fabric.device)

        metrics = test(model, dm, fabric.device)
        fabric.print(metrics.generate_table())

    fabric.print(f"\ndone. final threshold: {model.threshold_fpr95.item():.4f}")
    fabric.save(f"checkpoints/{checkpoint_name}_final.pt", {"model": model})
    return model


def _parse_args():
    p = argparse.ArgumentParser(description="SCONE energy MGT detector — full dataset training")
    p.add_argument("--tokenizer",       default="princeton-nlp/unsup-simcse-roberta-base")
    p.add_argument("--split",           default="scone-temporal",
                   choices=["scone-temporal", "scone-temporal-ablate", "all"])
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup_steps",    type=int,   default=2000)
    p.add_argument("--alpha_energy",    type=float, default=0.001)
    p.add_argument("--lambda_scone",    type=float, default=0.01)
    p.add_argument("--m_in",            type=float, default=-7.0)
    p.add_argument("--m_out",           type=float, default=-2.0)
    p.add_argument("--m_in_wild",       type=float, default=None)
    p.add_argument("--m_out_wild",      type=float, default=None)
    p.add_argument("--scone_warmup",    type=int,   default=4)
    p.add_argument("--wild_ratios",     type=float, nargs=3, default=[0.1, 0.6, 0.3],
                   metavar=("PI_ID", "PI_C", "PI_S"))
    p.add_argument("--wild_size",       type=int,   default=20_000)
    p.add_argument("--test_pool_cap",   type=int,   default=10_000)
    p.add_argument("--buffer",          type=float, default=0.1)
    p.add_argument("--train_ratio",     type=float, default=0.90)
    p.add_argument("--val_ratio",       type=float, default=0.05)
    p.add_argument("--test_ratio",      type=float, default=0.05)
    p.add_argument("--devices",         type=int,   default=1)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--silent",          action="store_true")
    p.add_argument("--no_attacks",      action="store_true")
    p.add_argument("--checkpoint_name", default="scone_energy_full")
    p.add_argument("--model",           default="inscone", choices=["inscone", "scone"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        tokenizer_name=args.tokenizer,
        split_strategy=args.split,
        batch_size_per_gpu=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        alpha_energy=args.alpha_energy,
        lambda_scone=args.lambda_scone,
        m_in=args.m_in,
        m_out=args.m_out,
        m_in_wild=args.m_in_wild,
        m_out_wild=args.m_out_wild,
        scone_warmup_epochs=args.scone_warmup,
        wild_ratios=tuple(args.wild_ratios),
        wild_size=args.wild_size,
        test_pool_cap=args.test_pool_cap,
        buffer=args.buffer,
        split_ratios=SplitRatios(
            train=args.train_ratio,
            val=args.val_ratio,
            test=args.test_ratio,
        ),
        devices=args.devices,
        seed=args.seed,
        silent=args.silent,
        attacks=not args.no_attacks,
        checkpoint_name=args.checkpoint_name,
        model_type=args.model,
    )
