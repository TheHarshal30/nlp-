import torch
import torch.nn.functional as F


def pairwise_distance(left: torch.Tensor, right: torch.Tensor, distance: str) -> torch.Tensor:
    if distance == "l2":
        return torch.norm(left - right, dim=-1)
    if distance == "cosine":
        return 1.0 - F.cosine_similarity(left, right, dim=-1)
    raise ValueError(f"Unsupported distance: {distance}")


def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float, distance: str) -> torch.Tensor:
    pos = pairwise_distance(anchor, positive, distance)
    neg = pairwise_distance(anchor, negative, distance)
    return torch.clamp(pos - neg + margin, min=0.0).mean()
