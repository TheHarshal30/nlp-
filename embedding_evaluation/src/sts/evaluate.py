import numpy as np

from ..utils.metrics import pearson_correlation
from .data import load_sts_pairs


def evaluate_sts(embedder, pairs_path: str, batch_size: int, smoke_limit: int | None = None) -> dict:
    pairs = load_sts_pairs(pairs_path)
    if smoke_limit:
        pairs = pairs[:smoke_limit]
    if not pairs:
        return {"pearson": 0.0, "pairs": 0}
    left = [row["sentence1"] for row in pairs]
    right = [row["sentence2"] for row in pairs]
    scores = np.array([row["score"] for row in pairs], dtype=np.float32)
    left_emb = embedder.encode(left, batch_size=batch_size)
    right_emb = embedder.encode(right, batch_size=batch_size)
    left_norm = left_emb / np.clip(np.linalg.norm(left_emb, axis=1, keepdims=True), 1e-12, None)
    right_norm = right_emb / np.clip(np.linalg.norm(right_emb, axis=1, keepdims=True), 1e-12, None)
    predicted = np.sum(left_norm * right_norm, axis=1)
    return {
        "pearson": pearson_correlation(predicted, scores),
        "pairs": len(pairs),
    }
