"""
ADAPTED FROM https://github.com/cong-zeng/ood-llm-detect/blob/main/model.py

TextEmbeddingModel is kept identical to the official repo.
HTAODetector wraps it for the fine-tuning baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name, output_hidden_states=False):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, model_output, attention_mask, use_pooling='average', hidden_states=False):
        if hidden_states:
            model_output.masked_fill(~attention_mask[None, ..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:, :, 0]
            emb = emb.permute(1, 0, 2)
        else:
            model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif use_pooling == "cls":
                emb = model_output[:, 0]
        return emb

    def forward(self, encoded_batch, use_pooling='average', hidden_states=False):
        if "t5" in self.model_name.lower():
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch,
                                      decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)

        if 'bge' in self.model_name.lower() or 'mxbai' in self.model_name.lower():
            use_pooling = 'cls'
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = model_output["hidden_states"]
                model_output = torch.stack(model_output, dim=0)
            else:
                model_output = model_output["last_hidden_state"]

        emb = self.pooling(model_output, encoded_batch['attention_mask'], use_pooling, hidden_states)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class HTAODetector(nn.Module):
    """
    combines all logic (besides the embedding model) into a single class for ease

    
    adapted from the official htao repo to work with our raid datamodule
    batch format: {tokens, mask, model, group}.

    label convention:
        our datamodule: LLM=1 (machine/ID), human=0 (OOD)

        idk if that is conventional but what i decided

    Args:
        model_name (str): hf model
        temperature (float): contrastive loss temperature.
        alpha_contrastive (float): weight for the contrastive loss (paper's beta).
        alpha_svdd (float): weight for the deepsvdd loss (paper's alpha).
        objective (str): 'one-class' or 'soft-boundary'.
        nu (float): soft-boundary quantile parameter.
    """
    def __init__(
        self,
        model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        temperature: float = 0.07,
        alpha_contrastive: float = 1.0,
        alpha_svdd: float = 1.0,
        objective: str = "one-class",
        nu: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_contrastive = alpha_contrastive
        self.alpha_svdd = alpha_svdd
        self.objective = objective
        self.nu = nu

        # model
        self.model = TextEmbeddingModel(model_name)
        self.out_dim = self.model.model.config.hidden_size
        self.encoder = self.model.model # helpful

        # params
        self.esp = torch.tensor(1e-6)
        self.c = nn.Parameter(torch.zeros(self.out_dim), requires_grad=False)
        self.R = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def _encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        bridge our batch format to TextEmbeddingModel's dict interface.

        Args:
            tokens (torch.Tensor): input token ids [B, L].
            mask (torch.Tensor): attention mask [B, L].

        Returns:
            torch.Tensor: l2-normalized embeddings [B, hidden_size].
        """
        encoded_batch = {"input_ids": tokens, "attention_mask": mask}
        return self.model(encoded_batch)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        encoder forward, for use in eval/test scoring.

        Args:
            tokens (torch.Tensor): input token ids [B, L].
            mask (torch.Tensor): attention mask [B, L].

        Returns:
            torch.Tensor: embeddings [B, hidden_size].
        """
        return self._encode(tokens, mask)

    def compute_energy(self, phi: torch.Tensor) -> torch.Tensor:
        """
        l2 distance from center.
        call it energy because im scone-coded

        Args:
            phi (torch.Tensor): embeddings [B, hidden_size].

        Returns:
            torch.Tensor: squared distances [B].
        """
        return torch.sum((phi.float() - self.c.float()) ** 2, dim=1)

    @torch.no_grad()
    def initialize_center_c(self, train_loader, device, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the machine data.
        taken from the simclr thing, adapted for our data format        
        """
        n_samples = 0
        c = torch.zeros(self.out_dim, device=device)

        self.eval()
        print('Initializing center c, to device:{}', device)

        for batch in tqdm(train_loader, desc="initializing center"):
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["model"].to(device)

            machine_mask = (labels == 1)
            if not machine_mask.any():
                continue

            outputs = self._encode(tokens[machine_mask], mask[machine_mask])
            c += outputs.float().sum(dim=0)
            n_samples += outputs.shape[0]

        c /= n_samples
        # Normalize to the hypersphere surface.
        c = c / torch.norm(c)
        self.c.data = c

    def _compute_logits(self, q, q_label, k, k_label):
        """identical to simclr"""
        def cosine_similarity_matrix(q, k):
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            cosine_similarity = q_norm @ k_norm.T
            return cosine_similarity

        logits = cosine_similarity_matrix(q, k) / self.temperature

        q_labels = q_label.view(-1, 1)# N,1
        k_labels = k_label.view(1, -1)# 1,N+K

        same_label = (q_labels == k_labels)# N,N+K

        #model:model set
        pos_logits_model = torch.sum(logits * same_label, dim=1) / torch.max(torch.sum(same_label, dim=1), self.esp)
        neg_logits_model = logits * torch.logical_not(same_label)
        logits_model = torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1)

        return logits_model

    def _compute_svdd_loss(self, outputs, machine_txt_idx, human_txt_idx):
        if not machine_txt_idx.any() or not human_txt_idx.any():
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        machine_outputs = outputs[machine_txt_idx]
        human_outputs = outputs[human_txt_idx]

        machine_outputs = machine_outputs.float()
        human_outputs = human_outputs.float()
        c_float = self.c.float()

        # Check if the input includes Nan or inf
        if torch.isnan(machine_outputs).any() or torch.isnan(human_outputs).any() or torch.isnan(c_float).any():
            print("Warning: NaN detected in inputs")
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        if torch.isinf(machine_outputs).any() or torch.isinf(human_outputs).any() or torch.isinf(c_float).any():
            print("Warning: Inf detected in inputs")
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        # compute the distance
        diff_machine = machine_outputs - c_float
        dist_machine = torch.sum(diff_machine ** 2, dim=1)
        dist_machine = torch.clamp(dist_machine, min=1e-12, max=1e6)

        diff_human = human_outputs - c_float
        dist_human = torch.sum(diff_human ** 2, dim=1)
        dist_human = torch.clamp(dist_human, min=1e-12, max=1e6)

        # Compute the avg distance
        avg_dist_machine = dist_machine.mean()
        avg_dist_human = dist_human.mean()

        if torch.isnan(avg_dist_machine) or torch.isnan(avg_dist_human):
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        diff = avg_dist_machine - avg_dist_human

        diff = torch.clamp(diff, min=-100, max=100)
        loss = F.softplus(diff)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        return loss

    def compute_loss(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        combined training loss matching the official htao objective.

        Args:
            tokens (torch.Tensor): input token ids [B, L].
            mask (torch.Tensor): attention mask [B, L].
            labels (torch.Tensor): binary labels [B]. LLM=1, human=0.

        Returns:
            tuple: (total_loss, loss_contrastive, loss_svdd).
        """
        bsz = tokens.size(0)
        q = self._encode(tokens, mask)
        k = q.clone().detach()
        k_label = labels.clone().detach()

        # Compute contrastive logits.
        logits_label = self._compute_logits(q, labels, k, k_label)

        # Calculate DeepSVDD loss.
        machine_txt_idx = (labels == 1).view(-1)
        human_txt_idx = (labels == 0).view(-1)
        loss_svdd = self._compute_svdd_loss(q, machine_txt_idx, human_txt_idx)

        # Compute contrastive loss.
        gt = torch.zeros(bsz, dtype=torch.long, device=logits_label.device)
        loss_contrastive = F.cross_entropy(logits_label, gt)

        # Combine both losses with their respective weights.
        total = self.alpha_svdd * loss_svdd + self.alpha_contrastive * loss_contrastive
        return total, loss_contrastive, loss_svdd
