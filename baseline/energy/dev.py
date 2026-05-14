import torch
import torch.nn.functional as F
from collections import defaultdict

def _log_energy_diagnostics(fabric, model, batch, i):
    """per-step energy breakdown by model group."""
    with torch.no_grad():
        phi = model._encode(batch["tokens"], batch["mask"])
        energy = model.compute_energy(phi)
        group_e = defaultdict(list)
        for g, e in zip(batch["group"], energy.tolist()):
            group_e[g].append(e)
        parts = " | ".join(f"{g}: {sum(v)/len(v):.3f}" for g, v in sorted(group_e.items()))
        fabric.print(f"  [step {i}] energy [{parts}]")


def _log_per_model_energy(fabric, model, dm, device):
    """epoch-level per-model energy across all test pillars."""
    pillar_names = ["id_retention", "wild_memorization", "zero_shot"]
    model.eval()
    with torch.no_grad():
        for pillar, loader in zip(pillar_names, dm.test_dataloader()):
            group_e = defaultdict(list)
            for batch in loader:
                phi = model._encode(batch["tokens"].to(device), batch["mask"].to(device))
                energy = model.compute_energy(phi)
                for g, e in zip(batch["group"], energy.tolist()):
                    group_e[g].append(e)
            fabric.print(f"  [{pillar}] per-model energy:")
            for g, vals in sorted(group_e.items()):
                t = torch.tensor(vals)
                fabric.print(f"    {g:20s} | mean: {t.mean():.3f} +/- {t.std():.3f} (n={len(vals)})")

def _collect_energies(model, loader, device):
    """collect energy scores and binary labels from a labeled loader."""
    energies, labels = [], []
    for batch in loader:
        phi = model._encode(batch["tokens"], batch["mask"])
        energies.append(model.compute_energy(phi).detach())
        labels.append(batch["label"].detach())
    return torch.cat(energies), torch.cat(labels)