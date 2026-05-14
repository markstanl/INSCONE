import torch
import torch.nn as nn
from tqdm import tqdm
from baseline.energy.metric import EnergyMetric


def test(model: nn.Module, dm, device: str | torch.device) -> EnergyMetric:
    """
    :param model: trained EnergyDetector.
    :param dm: initialized EnergyRAIDDataModule with test splits.
    :param device: target device.
    :returns: EnergyMetric instance (call .compute() or .generate_table()).
    """
    model.eval()
    metrics = EnergyMetric().to(device)

    with torch.no_grad():
        for loader_idx, loader in tqdm(enumerate(dm.test_dataloader()), total=3, desc="Testing"):
            for batch in loader:
                tokens = batch["tokens"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)  # machine=1, human=0

                phi = model(tokens, mask)
                energy = model.compute_energy(phi)

                # lower energy = more machine-like → negate for scoring
                metrics.update(-energy, labels, dataloader_idx=loader_idx)

    return metrics