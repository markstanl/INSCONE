"""
Two energy-based MGT detectors extending EnergyDetector with wild data margin losses.

SCONEEnergyDetector  — original SCONE formulation (Bai et al., 2023):
    pushes all wild samples toward E > 0 uniformly.
    covariate OOD generalization relies entirely on the passive Lipschitz drag
    from the ID margin constraint.

INSCONEEnergyDetector — informed SCONE (ours):
    splits wild by known π_s into proximal (estimated covariate) and distal
    (estimated semantic OOD), applying opposing losses to each partition.
    requires known wild composition — exact when wild is explicitly curated.

bai et al. (2023) https://arxiv.org/abs/2306.09158
"""

import torch
import torch.nn.functional as F
from baseline.energy.model import EnergyDetector


class SCONEEnergyDetector(EnergyDetector):
    """
    Original SCONE wild data margin loss.

    Pushes all wild samples toward E > 0 via relu(-E_wild)².
    Covariate OOD benefit is passive — relies on ID margin dragging nearby
    covariate OOD below E=0 via the Lipschitz argument (Prop. 3.1, Bai et al. 2023).

    :param model_name: HuggingFace hub identifier.
    :param n_classes: number of ID LLM family classes.
    :param temperature: contrastive loss temperature.
    :param alpha_contrastive: contrastive loss weight.
    :param alpha_classify: classifier CE loss weight.
    :param alpha_energy: weight for labeled energy margin loss.
    :param lambda_scone: weight for wild margin loss.
    :param m_in: η — energy margin for labeled ID samples.
    :param m_out: energy margin for labeled OOD (human) samples.
    :param scone_warmup_epochs: epochs before wild loss activates.
    """
    def __init__(
        self,
        model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        n_classes: int = 4,
        temperature: float = 0.07,
        alpha_contrastive: float = 1.0,
        alpha_classify: float = 1.0,
        alpha_energy: float = 0.001,
        lambda_scone: float = 0.01,
        m_in: float = -5.0,
        m_out: float = -1.0,
        scone_warmup_epochs: int = 2,
        wild_ratios: tuple[float] = (0.1, 0.6, 0.3),  # kept for API compat, unused in loss
        **kwargs, # allow extra args for flexible init from train.py
    ):
        super().__init__(
            model_name=model_name,
            n_classes=n_classes,
            temperature=temperature,
            alpha_contrastive=alpha_contrastive,
            alpha_classify=alpha_classify,
            alpha_energy=alpha_energy,
            m_in=m_in,
            m_out=m_out,
        )
        self.lambda_scone = lambda_scone
        self.scone_warmup_epochs = scone_warmup_epochs
        self.current_epoch = 0
        self.register_buffer("threshold_fpr95", torch.tensor(0.0))

    def _compute_scone_loss(self, e_wild: torch.Tensor) -> torch.Tensor:
        return F.relu(-e_wild).pow(2).mean()

    def compute_loss(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        family_idx: torch.Tensor,
        wild_tokens: torch.Tensor | None = None,
        wild_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param tokens: [B, L].
        :param mask: [B, L].
        :param labels: binary [B]. machine=1, human=0.
        :param family_idx: classifier class index [B]. -1 for human.
        :param wild_tokens: [B_w, L] or None.
        :param wild_mask: [B_w, L] or None.
        :returns: (total, loss_contrastive, loss_classify, loss_energy, loss_scone).
        """
        bsz = tokens.size(0)
        phi = self._encode(tokens, mask)
        k, k_label = phi.clone().detach(), labels.clone().detach()

        logits_con = self._compute_logits(phi, labels, k, k_label)
        loss_contrastive = F.cross_entropy(logits_con, torch.zeros(bsz, dtype=torch.long, device=phi.device))

        machine_mask = (labels == 1) & (family_idx >= 0)
        loss_classify = (
            F.cross_entropy(self.head(phi[machine_mask]), family_idx[machine_mask])
            if machine_mask.any() else torch.tensor(0.0, device=phi.device)
        )

        energy = self.compute_energy(phi)
        loss_energy = torch.where(
            labels == 1,
            F.relu(energy - self.m_in) ** 2,
            F.relu(self.m_out - energy) ** 2,
        ).mean()

        total = (
            self.alpha_contrastive * loss_contrastive
            + self.alpha_classify * loss_classify
            + self.alpha_energy * loss_energy
        )

        loss_scone = torch.tensor(0.0, device=tokens.device)
        if wild_tokens is not None and self.lambda_scone > 0 and self.current_epoch >= self.scone_warmup_epochs:
            phi_wild = self._encode(wild_tokens, wild_mask)
            e_wild = self.compute_energy(phi_wild)
            loss_scone = self._compute_scone_loss(e_wild)
            total = total + self.lambda_scone * loss_scone

        return total, loss_contrastive, loss_classify, loss_energy, loss_scone


class INSCONEEnergyDetector(SCONEEnergyDetector):
    """
    Informed SCONE — proximal/distal split using known wild composition π_s.

    Replaces the uniform wild push with a composition-aware split:
      - proximal partition (bottom 1-π_s quantile): estimated covariate OOD
        → pulled toward low energy via relu(E - m_in)²
      - distal partition (top π_s quantile): estimated semantic OOD
        → pushed above E=0 via relu(-E)²
      - buffer creates a dead zone around the boundary to handle uncertain membership

    since π_s is known exactly from explicit wild curation, the quantile split
    is exact rather than estimated — a strict advantage over Bai et al. (2023)
    who must sweep η blindly (Appendix B).

    :param buffer: fraction of wild samples around the composition boundary
                   that receive no gradient. 0.0 = hard split at π_s boundary.
    :param wild_ratios: (π_id, π_c, π_s). π_s = wild_ratios[2].
    """
    def __init__(
        self,
        model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        n_classes: int = 4,
        temperature: float = 0.07,
        alpha_contrastive: float = 1.0,
        alpha_classify: float = 1.0,
        alpha_energy: float = 0.001,
        lambda_scone: float = 0.01,
        m_in: float = -5.0,
        m_out: float = -1.0,
        buffer: float = 0.0,
        scone_warmup_epochs: int = 2,
        wild_ratios: tuple[float] = (0.1, 0.6, 0.3),
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            n_classes=n_classes,
            temperature=temperature,
            alpha_contrastive=alpha_contrastive,
            alpha_classify=alpha_classify,
            alpha_energy=alpha_energy,
            lambda_scone=lambda_scone,
            m_in=m_in,
            m_out=m_out,
            scone_warmup_epochs=scone_warmup_epochs,
            wild_ratios=wild_ratios,
        )
        self.tau_proximal = max(0.0, 1.0 - wild_ratios[2] - buffer / 2)
        self.tau_distal   = min(1.0, 1.0 - wild_ratios[2] + buffer / 2)
        assert self.tau_proximal <= self.tau_distal

    def _compute_scone_loss(self, e_wild: torch.Tensor) -> torch.Tensor:
        """
        Proximal pull + distal push with uncertainty buffer.

        :param e_wild: [B_w] wild energies.
        """
        e_prox = torch.quantile(e_wild.detach(), self.tau_proximal)
        e_dist = torch.quantile(e_wild.detach(), self.tau_distal)

        proximal_mask = e_wild <= e_prox
        distal_mask   = e_wild >= e_dist

        loss = torch.tensor(0.0, device=e_wild.device)
        if proximal_mask.any():
            loss = loss + F.relu(e_wild[proximal_mask] - self.m_in).pow(2).mean()
        if distal_mask.any():
            loss = loss + F.relu(-e_wild[distal_mask]).pow(2).mean()
        return loss