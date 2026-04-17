import math

import numpy as np


def cosine_similarity_matrix(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    c = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
    return q @ c.T


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_mean = left.mean()
    right_mean = right.mean()
    left_centered = left - left_mean
    right_centered = right - right_mean
    denom = math.sqrt(float((left_centered ** 2).sum() * (right_centered ** 2).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((left_centered * right_centered).sum() / denom)


def macro_f1_score(labels: list[str], predictions: list[str]) -> float:
    classes = sorted(set(labels) | set(predictions))
    if not classes:
        return 0.0
    f1s = []
    for label in classes:
        tp = sum(1 for truth, pred in zip(labels, predictions) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(labels, predictions) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(labels, predictions) if truth == label and pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))
