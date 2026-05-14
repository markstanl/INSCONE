"""
Energy-based OSR detector for MGT detection.

Trains a multi-class classifier over ID LLM families. Free energy from
classifier logits serves as the OOD score — ID machine text produces
high-confidence (low energy) predictions; human text produces diffuse
(high energy) predictions.

Liu et al. (2020) "Energy-based Out-of-distribution Detection"
https://arxiv.org/abs/2010.03759
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.deepsvdd.model import TextEmbeddingModel


class ClassificationHead(nn.Module):
    """
    3-layer MLP classification head.

    :param in_dim: input embedding dimension.
    :param n_classes: number of ID LLM family classes.
    """

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, in_dim // 16),
            nn.Tanh(),
            nn.Linear(in_dim // 16, n_classes),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnergyDetector(nn.Module):
    """
    SimCSE encoder + multi-class head + energy-based OOD scoring.

    The classifier is trained on ID machine families via cross-entropy.
    Free energy E(x) = -log Σᵢ exp(fᵢ(x)) serves as the anomaly score:
    low energy = confident ID prediction = machine text.
    high energy = diffuse prediction = human or unknown LLM.

    Energy margin loss explicitly shapes the energy surface with separate
    in/out margins.

    :param model_name: HuggingFace hub identifier for base encoder.
    :param n_classes: number of ID LLM family classes.
    :param temperature: contrastive loss temperature.
    :param alpha_contrastive: weight for SimCLR-style contrastive loss.
    :param alpha_classify: weight for classifier cross-entropy.
    :param alpha_energy: weight for energy margin loss.
    :param m_in: energy margin for ID samples (should be negative).
    :param m_out: energy margin for OOD samples (should be > m_in).
    """

    def __init__(
            self,
            model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
            n_classes: int = 2,
            temperature: float = 0.07,
            alpha_contrastive: float = 1.0,
            alpha_classify: float = 1.0,
            alpha_energy: float = 0.01,
            m_in: float = -27.0,
            m_out: float = -5.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_contrastive = alpha_contrastive
        self.alpha_classify = alpha_classify
        self.alpha_energy = alpha_energy
        self.register_buffer("m_in", torch.tensor(m_in))
        self.register_buffer("m_out", torch.tensor(m_out))

        self.model = TextEmbeddingModel(model_name)
        self.encoder = self.model.model
        self.out_dim = self.encoder.config.hidden_size
        self.head = ClassificationHead(self.out_dim, n_classes)
        self.esp = torch.tensor(1e-6)

        self.register_buffer("threshold_fpr95", torch.tensor(0.0))

    def _encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model({"input_ids": tokens, "attention_mask": mask})

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._encode(tokens, mask)

    def compute_energy(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Free energy from classifier logits.

        :param phi: embeddings [B, D].
        :returns: energy scores [B]. lower = more ID-like.
        """
        logits = self.head(phi)
        return -torch.logsumexp(logits, dim=1)

    def _compute_logits(self, q: torch.Tensor, q_label: torch.Tensor, k: torch.Tensor,
                        k_label: torch.Tensor) -> torch.Tensor:
        """SimCLR-style contrastive logits using binary machine/human labels."""
        logits = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).T / self.temperature
        same = (q_label.view(-1, 1) == k_label.view(1, -1))
        pos = torch.sum(logits * same, dim=1) / torch.max(same.float().sum(dim=1), self.esp)
        neg = logits * ~same
        return torch.cat([pos.unsqueeze(1), neg], dim=1)

    def compute_loss(self, tokens, mask, labels, family_idx):
        bsz = tokens.size(0)
        phi = self._encode(tokens, mask)
        k, k_label = phi.clone().detach(), labels.clone().detach()

        # binary contrastive — matches original exactly
        logits_con = self._compute_logits(phi, labels, k, k_label)
        loss_contrastive = F.cross_entropy(logits_con, torch.zeros(bsz, dtype=torch.long, device=phi.device))

        # family-level classifier on machine text only
        machine_mask = (labels == 1) & (family_idx >= 0)
        if machine_mask.any():
            loss_classify = F.cross_entropy(self.head(phi[machine_mask]), family_idx[machine_mask])
        else:
            loss_classify = torch.tensor(0.0, device=phi.device)

        energy = self.compute_energy(phi)
        loss_energy = torch.where(
            labels == 1,
            F.relu(energy - self.m_in) ** 2,
            F.relu(self.m_out - energy) ** 2,
        ).mean()

        total = self.alpha_contrastive * loss_contrastive + self.alpha_classify * loss_classify + self.alpha_energy * loss_energy
        return total, loss_contrastive, loss_classify, loss_energy