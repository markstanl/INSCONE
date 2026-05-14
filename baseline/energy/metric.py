import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc, binary_average_precision, binary_roc


class EnergyMetric(Metric):
    """
    auroc, aupr, fpr95 across topological test pillars for energy-based detection.

    scores are expected as -energy (higher = more machine-like).
    labels: machine=1, human=0.

    :param pillar_names: ordered list matching test_dataloader() indices.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        pillar_names: list[str] = ["id_retention", "wild_memorization", "zero_shot"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pillar_names = pillar_names
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("pillars", default=[], dist_reduce_fx="cat")

    def update(self, scores: torch.Tensor, labels: torch.Tensor, dataloader_idx: int) -> None:
        """
        :param scores: -energy values [B]. higher = more machine-like.
        :param labels: binary targets [B]. machine=1, human=0.
        :param dataloader_idx: pillar index.
        """
        self.scores.append(scores.float())
        self.labels.append(labels.long())
        self.pillars.append(torch.full_like(labels, dataloader_idx, dtype=torch.long))

    def _fpr95(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        fpr, tpr, _ = binary_roc(scores, labels)
        valid = torch.where(tpr >= 0.95)[0]
        return fpr[valid[0]] if len(valid) > 0 else torch.tensor(1.0, device=scores.device)

    def compute(self) -> dict[str, torch.Tensor]:
        all_s = self.scores if isinstance(self.scores, torch.Tensor) else torch.cat(self.scores)
        all_l = self.labels if isinstance(self.labels, torch.Tensor) else torch.cat(self.labels)
        all_p = self.pillars if isinstance(self.pillars, torch.Tensor) else torch.cat(self.pillars)

        results = {
            "overall_auroc": binary_auroc(all_s, all_l),
            "overall_aupr": binary_average_precision(all_s, all_l),
            "overall_fpr95": self._fpr95(all_s, all_l),
        }
        for idx, name in enumerate(self.pillar_names):
            mask = all_p == idx
            if not mask.any():
                continue
            s, l = all_s[mask], all_l[mask]
            results[f"{name}_auroc"] = binary_auroc(s, l)
            results[f"{name}_aupr"] = binary_average_precision(s, l)
            results[f"{name}_fpr95"] = self._fpr95(s, l)
        return results

    def generate_table(self) -> str:
        res = self.compute()
        header = f"{'Pillar':<20} | {'AUROC':<6} | {'AUPR':<6} | {'FPR95':<6}"
        sep = "-" * len(header)
        lines = [sep, header, sep,
                 f"{'OVERALL':<20} | {res['overall_auroc']:.4f} | {res['overall_aupr']:.4f} | {res['overall_fpr95']:.4f}",
                 sep]
        for name in self.pillar_names:
            if f"{name}_auroc" in res:
                lines.append(f"{name:<20} | {res[f'{name}_auroc']:.4f} | {res[f'{name}_aupr']:.4f} | {res[f'{name}_fpr95']:.4f}")
        lines.append(sep)
        return "\n".join(lines)