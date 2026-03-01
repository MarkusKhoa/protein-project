"""Loss functions for binding site prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = torch.tensor([1.0, self.pos_weight], dtype=logits.dtype, device=logits.device)
        return F.cross_entropy(logits, labels, weight=weight)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs[torch.arange(probs.size(0), device=probs.device), labels]
        focal_term = (1 - true_probs) ** self.gamma
        alpha_weight = torch.where(labels == 1, self.alpha, 1.0 - self.alpha).to(logits.device)
        loss = alpha_weight * focal_term * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ImprovedPositionAwareLoss(nn.Module):
    """Position-aware loss with device-agnostic weight tensor."""

    def __init__(self, pos_weight: float = 7.0, position_weight: float = 0.1, device: torch.device | None = None):
        super().__init__()
        self.pos_weight = pos_weight
        self.position_weight = position_weight
        self._weight = None
        self._device = device

    def _get_weight(self, logits: torch.Tensor) -> torch.Tensor:
        if self._weight is None or self._weight.device != logits.device:
            self._weight = torch.tensor([1.0, self.pos_weight], dtype=logits.dtype, device=logits.device)
        return self._weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch=None,
    ) -> torch.Tensor:
        weight = self._get_weight(logits)
        loss = F.cross_entropy(logits, labels, weight=weight)

        if batch is not None and hasattr(batch, "batch"):
            probs = torch.softmax(logits, dim=1)[:, 1]
            position_loss = torch.tensor(0.0, device=logits.device)

            unique_graphs = torch.unique(batch.batch)
            for graph_idx in unique_graphs:
                graph_mask = batch.batch == graph_idx
                graph_nodes = torch.where(graph_mask)[0]

                if len(graph_nodes) > 1:
                    for i in range(len(graph_nodes) - 1):
                        curr_idx = graph_nodes[i]
                        next_idx = graph_nodes[i + 1]
                        if labels[curr_idx] == 1 or labels[next_idx] == 1:
                            position_loss = position_loss + torch.abs(
                                probs[curr_idx] - probs[next_idx]
                            )

            loss = loss + self.position_weight * position_loss

        return loss


def add_binding_features(batch, device: str | torch.device = "cpu"):
    """Add binding-prone and binding-propensity features to batch."""
    aa_vocab = [
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
        "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    ]
    binding_prone_aa = {"C", "H", "W", "R", "Y"}
    binding_site_percentage = {
        "C": 25.7, "H": 22.0, "W": 15.9, "Y": 12.8, "R": 12.1, "F": 10.3,
        "D": 8.7, "N": 8.4, "K": 8.4, "M": 8.0, "G": 7.6, "T": 7.1,
        "S": 7.0, "E": 6.6, "I": 6.3, "Q": 6.2, "V": 5.8, "P": 5.7,
        "L": 5.7, "A": 4.6,
    }
    min_val = min(binding_site_percentage.values())
    max_val = max(binding_site_percentage.values())
    norm_binding_prob = {aa: (v - min_val) / (max_val - min_val) for aa, v in binding_site_percentage.items()}

    if hasattr(batch, "amino_acid_types"):
        aa_types = batch.amino_acid_types
        if not isinstance(aa_types, list):
            aa_types = [aa_vocab[i] for i in aa_types.cpu().tolist()]
    elif batch.x.shape[1] >= 20:
        aa_indices = batch.x[:, :20].argmax(dim=1)
        aa_types = [aa_vocab[i] for i in aa_indices.cpu().tolist()]
    else:
        raise ValueError("Cannot infer amino acid types from batch.")

    is_binding_prone = torch.tensor(
        [1 if aa in binding_prone_aa else 0 for aa in aa_types],
        dtype=torch.float32,
        device=batch.x.device,
    )
    batch.is_binding_prone = is_binding_prone

    binding_propensity = torch.tensor(
        [norm_binding_prob[aa] for aa in aa_types],
        dtype=torch.float32,
        device=batch.x.device,
    ).unsqueeze(1)
    batch.x = torch.cat([batch.x, binding_propensity], dim=1)
    return batch
