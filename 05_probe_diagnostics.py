"""
05_probe_diagnostics.py
───────────────────────
Run leakage-resistant diagnostics on frozen Word2Vec phrase vectors.

Experiments:
  - cosine baseline with threshold calibration
  - linear probe on concatenated vectors
  - MLP probe on concatenated vectors
  - linear probe on symmetric features
  - MLP probe on symmetric features
  - MLP probe on symmetric features with hard negatives

Outputs:
  - results/*.json
  - results/*.svg
  - results/final_report.md
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def stable_bucket(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100


def canonical_pair(a_idx: int, b_idx: int) -> tuple[int, int]:
    return (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)


def split_name_for_phrase(text: str, val_pct: int, test_pct: int) -> str:
    bucket = stable_bucket(text)
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def phrase_to_vector(text: str, wv: KeyedVectors) -> np.ndarray | None:
    tokens = [token for token in text.lower().split() if token in wv]
    if not tokens:
        return None
    vec = np.mean([wv[token] for token in tokens], axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return None
    return vec / norm


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_points(scores: np.ndarray, labels: np.ndarray, num_thresholds: int = 400) -> list[tuple[float, float]]:
    thresholds = np.linspace(float(scores.max()), float(scores.min()), num_thresholds)
    points = []
    pos = max(int(labels.sum()), 1)
    neg = max(int((1 - labels).sum()), 1)
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tpr = tp / pos
        fpr = fp / neg
        points.append((fpr, tpr))
    points.append((0.0, 0.0))
    points.append((1.0, 1.0))
    points = sorted(set(points))
    return points


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    accuracy = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        roc_auc = float("nan")
    else:
        ranks = rankdata(scores)
        rank_sum_pos = float(ranks[labels == 1].sum())
        roc_auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
    }


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, lo: float, hi: float, steps: int = 801) -> tuple[float, dict[str, float]]:
    best_threshold = lo
    best_metrics = None
    best_f1 = -1.0
    for threshold in np.linspace(lo, hi, steps):
        metrics = binary_metrics(scores, labels, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_svg_line_plot(path: str, series: list[tuple[float, float]], title: str, x_label: str, y_label: str) -> None:
    width, height = 700, 420
    pad = 50
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    if not series:
        series = [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0

    def px(x: float) -> float:
        return pad + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        return height - pad - (y - y_min) / (y_max - y_min) * plot_h

    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in series)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{points}"/>
<text x="{width/2:.0f}" y="{height-10}" text-anchor="middle" font-size="14" font-family="sans-serif">{x_label}</text>
<text x="18" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 18 {height/2:.0f})">{y_label}</text>
</svg>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def save_svg_multi_line_plot(path: str, histories: dict[str, list[tuple[float, float]]], title: str, x_label: str, y_label: str) -> None:
    width, height = 760, 460
    pad = 55
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    all_points = [pt for series in histories.values() for pt in series]
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0

    def px(x: float) -> float:
        return pad + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        return height - pad - (y - y_min) / (y_max - y_min) * plot_h

    legend = []
    polylines = []
    for i, (name, series) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in series)
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend_y = 30 + i * 18
        legend.append(f'<rect x="{width-180}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
        legend.append(f'<text x="{width-162}" y="{legend_y}" font-size="12" font-family="sans-serif">{name}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>
{"".join(polylines)}
{"".join(legend)}
<text x="{width/2:.0f}" y="{height-10}" text-anchor="middle" font-size="14" font-family="sans-serif">{x_label}</text>
<text x="18" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 18 {height/2:.0f})">{y_label}</text>
</svg>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


@dataclass
class PreparedData:
    phrase_vectors: np.ndarray
    phrase_texts: list[str]
    split_phrase_ids: dict[str, np.ndarray]
    split_positive_pairs: dict[str, np.ndarray]
    synonym_sets: dict[int, set[int]]
    validation: dict[str, bool | int]


class PairDataset(Dataset):
    def __init__(self, left: np.ndarray, right: np.ndarray, labels: np.ndarray):
        self.left = left
        self.right = right
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.left[idx]),
            torch.from_numpy(self.right[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class ProbeNet(nn.Module):
    def __init__(self, input_dim: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(f"Unknown model kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_features(left: np.ndarray, right: np.ndarray, mode: str) -> np.ndarray:
    if mode == "concat":
        return np.concatenate([left, right], axis=1).astype(np.float32)
    if mode == "symmetric":
        return np.concatenate([np.abs(left - right), left * right], axis=1).astype(np.float32)
    raise ValueError(f"Unknown feature mode: {mode}")


def prepare_data(
    pairs_path: str,
    wv: KeyedVectors,
    val_pct: int,
    test_pct: int,
    max_pairs: int | None,
) -> PreparedData:
    phrase_to_idx: dict[str, int] = {}
    phrase_vectors: list[np.ndarray] = []
    invalid_phrases: set[str] = set()
    split_positive_pairs: dict[str, list[tuple[int, int]]] = {"train": [], "val": [], "test": []}
    split_phrase_ids: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    synonym_sets: dict[int, set[int]] = defaultdict(set)
    split_pairs_seen: dict[str, set[tuple[int, int]]] = {"train": set(), "val": set(), "test": set()}

    def get_phrase_idx(text: str) -> int | None:
        if text in phrase_to_idx:
            return phrase_to_idx[text]
        if text in invalid_phrases:
            return None
        vec = phrase_to_vector(text, wv)
        if vec is None:
            invalid_phrases.add(text)
            return None
        idx = len(phrase_vectors)
        phrase_to_idx[text] = idx
        phrase_vectors.append(vec.astype(np.float16))
        return idx

    processed = 0
    with open(pairs_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Preparing phrase-level splits", unit="pair"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            split_a = split_name_for_phrase(a, val_pct=val_pct, test_pct=test_pct)
            split_b = split_name_for_phrase(b, val_pct=val_pct, test_pct=test_pct)
            if split_a != split_b:
                continue

            a_idx = get_phrase_idx(a)
            b_idx = get_phrase_idx(b)
            if a_idx is None or b_idx is None or a_idx == b_idx:
                continue

            split_name = split_a
            pair = canonical_pair(a_idx, b_idx)
            if pair in split_pairs_seen[split_name]:
                continue

            split_pairs_seen[split_name].add(pair)
            split_positive_pairs[split_name].append(pair)
            split_phrase_ids[split_name].update(pair)
            synonym_sets[pair[0]].add(pair[1])
            synonym_sets[pair[1]].add(pair[0])
            processed += 1
            if max_pairs is not None and processed >= max_pairs:
                break

    phrase_texts = [None] * len(phrase_to_idx)
    for text, idx in phrase_to_idx.items():
        phrase_texts[idx] = text

    train_phrases = split_phrase_ids["train"]
    val_phrases = split_phrase_ids["val"]
    test_phrases = split_phrase_ids["test"]
    assert train_phrases.isdisjoint(val_phrases)
    assert train_phrases.isdisjoint(test_phrases)
    assert val_phrases.isdisjoint(test_phrases)
    assert split_pairs_seen["train"].isdisjoint(split_pairs_seen["val"])
    assert split_pairs_seen["train"].isdisjoint(split_pairs_seen["test"])
    assert split_pairs_seen["val"].isdisjoint(split_pairs_seen["test"])

    validation = {
        "phrase_overlap_train_val": False,
        "phrase_overlap_train_test": False,
        "phrase_overlap_val_test": False,
        "duplicate_pairs_across_splits": False,
        "num_phrases": len(phrase_texts),
        "num_train_pairs": len(split_positive_pairs["train"]),
        "num_val_pairs": len(split_positive_pairs["val"]),
        "num_test_pairs": len(split_positive_pairs["test"]),
    }

    return PreparedData(
        phrase_vectors=np.vstack(phrase_vectors).astype(np.float16),
        phrase_texts=phrase_texts,
        split_phrase_ids={k: np.asarray(sorted(v), dtype=np.int32) for k, v in split_phrase_ids.items()},
        split_positive_pairs={k: np.asarray(v, dtype=np.int32) for k, v in split_positive_pairs.items()},
        synonym_sets=synonym_sets,
        validation=validation,
    )


class NegativeSampler:
    def __init__(
        self,
        phrase_vectors: np.ndarray,
        split_phrase_ids: np.ndarray,
        synonym_sets: dict[int, set[int]],
        strategy: str,
        seed: int,
        hard_pool_size: int,
        hard_topk: int,
    ):
        self.vectors = phrase_vectors.astype(np.float32)
        self.split_phrase_ids = split_phrase_ids
        self.split_phrase_set = set(int(x) for x in split_phrase_ids.tolist())
        self.synonym_sets = synonym_sets
        self.strategy = strategy
        self.seed = seed
        self.hard_pool_size = hard_pool_size
        self.hard_topk = hard_topk
        self.cache: dict[int, np.ndarray] = {}

    def _valid_candidates(self, anchor_idx: int) -> np.ndarray:
        invalid = set(self.synonym_sets.get(anchor_idx, set()))
        invalid.add(anchor_idx)
        candidates = [idx for idx in self.split_phrase_ids.tolist() if int(idx) not in invalid]
        if not candidates:
            raise ValueError(f"No valid negatives left for phrase id {anchor_idx}")
        return np.asarray(candidates, dtype=np.int32)

    def _hard_neighbors(self, anchor_idx: int) -> np.ndarray:
        if anchor_idx in self.cache:
            return self.cache[anchor_idx]

        rng = np.random.default_rng(self.seed + anchor_idx)
        candidates = self._valid_candidates(anchor_idx)
        if len(candidates) > self.hard_pool_size:
            sampled = rng.choice(candidates, size=self.hard_pool_size, replace=False)
        else:
            sampled = candidates

        anchor = self.vectors[anchor_idx]
        scores = self.vectors[sampled] @ anchor
        order = np.argsort(scores)[::-1]
        top = sampled[order[: max(1, min(self.hard_topk, len(order)))]]
        self.cache[anchor_idx] = top.astype(np.int32)
        return self.cache[anchor_idx]

    def sample(self, anchor_idx: int, pair_offset: int) -> int:
        if self.strategy == "random":
            candidates = self._valid_candidates(anchor_idx)
            rng = np.random.default_rng(self.seed + pair_offset)
            return int(candidates[int(rng.integers(0, len(candidates)))])
        if self.strategy == "hard":
            top = self._hard_neighbors(anchor_idx)
            rng = np.random.default_rng(self.seed + pair_offset)
            return int(top[int(rng.integers(0, len(top)))])
        raise ValueError(f"Unknown negative strategy: {self.strategy}")


def build_examples(
    positive_pairs: np.ndarray,
    phrase_vectors: np.ndarray,
    sampler: NegativeSampler,
    feature_mode: str | None,
    score_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_list = []
    right_list = []
    labels = []
    for idx, (a_idx, b_idx) in enumerate(tqdm(positive_pairs, desc="Building examples", unit="pair")):
        left_list.append(phrase_vectors[a_idx].astype(np.float32))
        right_list.append(phrase_vectors[b_idx].astype(np.float32))
        labels.append(1.0)

        neg_idx = sampler.sample(int(a_idx), idx)
        left_list.append(phrase_vectors[a_idx].astype(np.float32))
        right_list.append(phrase_vectors[neg_idx].astype(np.float32))
        labels.append(0.0)

    left = np.stack(left_list).astype(np.float32)
    right = np.stack(right_list).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.float32)
    if score_only:
        return left, right, labels_np
    return build_features(left, right, feature_mode), labels_np, right


def cosine_scores(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1).astype(np.float32)


def run_cosine_experiment(
    data: PreparedData,
    results_dir: str,
    seed: int,
    hard_negative: bool = False,
    max_eval_pairs: int | None = None,
) -> dict:
    name = "cosine_hard_negatives" if hard_negative else "cosine_baseline"
    strategy = "hard" if hard_negative else "random"
    subset = lambda arr: arr[:max_eval_pairs] if max_eval_pairs is not None else arr
    val_pairs = subset(data.split_positive_pairs["val"])
    test_pairs = subset(data.split_positive_pairs["test"])

    val_sampler = NegativeSampler(
        data.phrase_vectors,
        data.split_phrase_ids["val"],
        data.synonym_sets,
        strategy=strategy,
        seed=seed + 11,
        hard_pool_size=4096,
        hard_topk=10,
    )
    test_sampler = NegativeSampler(
        data.phrase_vectors,
        data.split_phrase_ids["test"],
        data.synonym_sets,
        strategy=strategy,
        seed=seed + 19,
        hard_pool_size=4096,
        hard_topk=10,
    )

    val_left, val_right, val_labels = build_examples(val_pairs, data.phrase_vectors, val_sampler, feature_mode=None, score_only=True)
    test_left, test_right, test_labels = build_examples(test_pairs, data.phrase_vectors, test_sampler, feature_mode=None, score_only=True)
    val_scores = cosine_scores(val_left, val_right)
    test_scores = cosine_scores(test_left, test_right)
    threshold, val_metrics = calibrate_threshold(val_scores, val_labels, lo=-1.0, hi=1.0)
    test_metrics = binary_metrics(test_scores, test_labels, threshold=threshold)

    payload = {
        "experiment": name,
        "negative_strategy": strategy,
        "validation": data.validation,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "num_val_examples": int(len(val_labels)),
        "num_test_examples": int(len(test_labels)),
    }
    save_json(os.path.join(results_dir, f"{name}.json"), payload)
    save_svg_line_plot(
        os.path.join(results_dir, f"{name}_roc.svg"),
        roc_points(test_scores, test_labels.astype(np.int64)),
        title=f"ROC Curve: {name}",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
    )
    return payload


def evaluate_model_scores(loader: DataLoader, model: ProbeNet) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_chunks = []
    label_chunks = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(torch.float32))
            probs = torch.sigmoid(logits)
            score_chunks.append(probs.cpu().numpy())
            label_chunks.append(labels.cpu().numpy())
    return np.concatenate(score_chunks), np.concatenate(label_chunks)


def run_learned_experiment(
    data: PreparedData,
    results_dir: str,
    seed: int,
    model_kind: str,
    feature_mode: str,
    negative_strategy: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_threads: int,
    max_train_pairs: int | None,
    max_eval_pairs: int | None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)

    train_pairs = data.split_positive_pairs["train"][:max_train_pairs] if max_train_pairs is not None else data.split_positive_pairs["train"]
    val_pairs = data.split_positive_pairs["val"][:max_eval_pairs] if max_eval_pairs is not None else data.split_positive_pairs["val"]
    test_pairs = data.split_positive_pairs["test"][:max_eval_pairs] if max_eval_pairs is not None else data.split_positive_pairs["test"]

    train_sampler = NegativeSampler(
        data.phrase_vectors,
        data.split_phrase_ids["train"],
        data.synonym_sets,
        strategy=negative_strategy,
        seed=seed + 101,
        hard_pool_size=4096,
        hard_topk=10,
    )
    val_sampler = NegativeSampler(
        data.phrase_vectors,
        data.split_phrase_ids["val"],
        data.synonym_sets,
        strategy=negative_strategy,
        seed=seed + 102,
        hard_pool_size=4096,
        hard_topk=10,
    )
    test_sampler = NegativeSampler(
        data.phrase_vectors,
        data.split_phrase_ids["test"],
        data.synonym_sets,
        strategy=negative_strategy,
        seed=seed + 103,
        hard_pool_size=4096,
        hard_topk=10,
    )

    train_features, train_labels, _ = build_examples(train_pairs, data.phrase_vectors, train_sampler, feature_mode=feature_mode)
    val_features, val_labels, val_right_raw = build_examples(val_pairs, data.phrase_vectors, val_sampler, feature_mode=feature_mode)
    test_features, test_labels, test_right_raw = build_examples(test_pairs, data.phrase_vectors, test_sampler, feature_mode=feature_mode)

    train_ds = PairDataset(train_features, train_features, train_labels)
    val_ds = PairDataset(val_features, val_features, val_labels)
    test_ds = PairDataset(test_features, test_features, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[2] for item in batch]),
    ))
    eval_loader = lambda ds: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[2] for item in batch]),
    ))

    input_dim = train_features.shape[1]
    model = ProbeNet(input_dim=input_dim, kind=model_kind)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_state = None
    best_threshold = 0.5
    best_val_auc = -math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for features, labels in tqdm(train_loader, desc=f"{model_kind}-{feature_mode}-{negative_strategy}", unit="batch"):
            optimiser.zero_grad()
            logits = model(features.to(torch.float32))
            labels = labels.to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss_total += float(loss.item())

        val_scores, val_targets = evaluate_model_scores(eval_loader(val_ds), model)
        threshold, val_metrics = calibrate_threshold(val_scores, val_targets, lo=0.0, hi=1.0)
        train_loss = train_loss_total / max(len(train_loader), 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "threshold": threshold,
        })

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = threshold

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")

    model.load_state_dict(best_state)
    test_scores, test_targets = evaluate_model_scores(eval_loader(test_ds), model)
    test_metrics = binary_metrics(test_scores, test_targets, threshold=best_threshold)

    reversed_test_features = build_features(test_right_raw, test_features[:, : data.phrase_vectors.shape[1]], mode=feature_mode) if feature_mode == "concat" else test_features
    if feature_mode == "concat":
        reversed_scores, _ = evaluate_model_scores(
            DataLoader(PairDataset(reversed_test_features, reversed_test_features, test_labels), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: (
                torch.stack([item[0] for item in batch]),
                torch.stack([item[2] for item in batch]),
            )),
            model,
        )
        order_diff = float(np.mean(np.abs(test_scores - reversed_scores)))
    else:
        order_diff = 0.0

    experiment_name = {
        ("linear", "concat", "random"): "linear_probe",
        ("mlp", "concat", "random"): "mlp_concat",
        ("linear", "symmetric", "random"): "linear_symmetric",
        ("mlp", "symmetric", "random"): "mlp_symmetric",
        ("mlp", "symmetric", "hard"): "hard_negatives",
    }[(model_kind, feature_mode, negative_strategy)]

    payload = {
        "experiment": experiment_name,
        "model_kind": model_kind,
        "feature_mode": feature_mode,
        "negative_strategy": negative_strategy,
        "validation": data.validation,
        "history": history,
        "best_val_roc_auc": best_val_auc,
        "threshold": best_threshold,
        "order_invariance_mean_abs_diff": order_diff,
        "test_metrics": test_metrics,
        "num_train_examples": int(len(train_labels)),
        "num_val_examples": int(len(val_labels)),
        "num_test_examples": int(len(test_labels)),
    }
    save_json(os.path.join(results_dir, f"{experiment_name}.json"), payload)
    save_svg_line_plot(
        os.path.join(results_dir, f"{experiment_name}_roc.svg"),
        roc_points(test_scores, test_targets.astype(np.int64)),
        title=f"ROC Curve: {experiment_name}",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
    )
    save_svg_multi_line_plot(
        os.path.join(results_dir, f"{experiment_name}_training.svg"),
        {
            "train_loss": [(row["epoch"], row["train_loss"]) for row in history],
            "val_auc": [(row["epoch"], row["val_roc_auc"]) for row in history],
        },
        title=f"Training Curves: {experiment_name}",
        x_label="Epoch",
        y_label="Value",
    )
    return payload


def load_old_baseline(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_final_report(results_dir: str, rows: list[tuple[str, dict | None]], interpretation: list[str], invalid_flags: list[str]) -> None:
    lines = ["# Final Report", "", "| Experiment | Accuracy | Precision | Recall | ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for name, payload in rows:
        metrics = None
        if payload is None:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a |")
            continue
        if "final" in payload:
            metrics = payload["final"]
        elif "test_metrics" in payload:
            metrics = payload["test_metrics"]
        else:
            metrics = payload
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['roc_auc']:.4f} |"
        )

    lines.extend(["", "## Invalidity Checks", ""])
    if invalid_flags:
        lines.extend([f"- {flag}" for flag in invalid_flags])
    else:
        lines.append("- No invalidity conditions triggered.")

    lines.extend(["", "## Conclusions", ""])
    lines.extend([f"- {line}" for line in interpretation])

    with open(os.path.join(results_dir, "final_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run strict diagnostics on frozen Word2Vec embeddings")
    parser.add_argument("--w2v_bin", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--old_baseline_metrics", default="models/probes/mlp_metrics.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_pct", type=int, default=10)
    parser.add_argument("--test_pct", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    parser.add_argument("--max_eval_pairs", type=int, default=200000)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size > 512:
        raise ValueError("Batch size must be <= 512.")

    os.makedirs(args.results_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    log.info("Loading Word2Vec from %s", args.w2v_bin)
    wv = KeyedVectors.load_word2vec_format(args.w2v_bin, binary=True)
    data = prepare_data(args.pairs, wv, val_pct=args.val_pct, test_pct=args.test_pct, max_pairs=args.max_pairs)
    save_json(os.path.join(args.results_dir, "split_validation.json"), data.validation)

    old_baseline = load_old_baseline(args.old_baseline_metrics)
    cosine = run_cosine_experiment(data, args.results_dir, seed=args.seed, hard_negative=False, max_eval_pairs=args.max_eval_pairs)
    linear = run_learned_experiment(data, args.results_dir, seed=args.seed, model_kind="linear", feature_mode="concat", negative_strategy="random", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_threads=args.num_threads, max_train_pairs=args.max_train_pairs, max_eval_pairs=args.max_eval_pairs)
    mlp_concat = run_learned_experiment(data, args.results_dir, seed=args.seed + 1, model_kind="mlp", feature_mode="concat", negative_strategy="random", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_threads=args.num_threads, max_train_pairs=args.max_train_pairs, max_eval_pairs=args.max_eval_pairs)
    linear_sym = run_learned_experiment(data, args.results_dir, seed=args.seed + 2, model_kind="linear", feature_mode="symmetric", negative_strategy="random", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_threads=args.num_threads, max_train_pairs=args.max_train_pairs, max_eval_pairs=args.max_eval_pairs)
    mlp_sym = run_learned_experiment(data, args.results_dir, seed=args.seed + 3, model_kind="mlp", feature_mode="symmetric", negative_strategy="random", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_threads=args.num_threads, max_train_pairs=args.max_train_pairs, max_eval_pairs=args.max_eval_pairs)
    hard = run_cosine_experiment(data, args.results_dir, seed=args.seed + 4, hard_negative=True, max_eval_pairs=args.max_eval_pairs)
    hard_mlp = run_learned_experiment(data, args.results_dir, seed=args.seed + 5, model_kind="mlp", feature_mode="symmetric", negative_strategy="hard", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_threads=args.num_threads, max_train_pairs=args.max_train_pairs, max_eval_pairs=args.max_eval_pairs)

    invalid_flags = []
    if data.validation["phrase_overlap_train_test"] or data.validation["duplicate_pairs_across_splits"]:
        invalid_flags.append("Split validation failed.")

    if old_baseline is not None and cosine["test_metrics"]["accuracy"] + 0.15 < old_baseline["final"]["accuracy"]:
        invalid_flags.append("Old baseline accuracy greatly exceeds cosine baseline; inspect probe capacity or leakage.")

    interpretation = []
    if cosine["test_metrics"]["roc_auc"] >= 0.9:
        interpretation.append("Cosine baseline is strong, so the frozen embeddings already carry substantial synonym signal.")
    else:
        interpretation.append("Cosine baseline is weak, so raw embedding geometry alone is not sufficient.")

    if mlp_concat["test_metrics"]["roc_auc"] > cosine["test_metrics"]["roc_auc"] + 0.03:
        interpretation.append("The MLP probe adds noticeable capacity beyond raw cosine, so some of the performance comes from the probe.")
    else:
        interpretation.append("The MLP probe adds little over cosine, which suggests the embeddings are doing most of the work.")

    if hard_mlp["test_metrics"]["accuracy"] < mlp_sym["test_metrics"]["accuracy"] - 0.05:
        interpretation.append("Hard negatives reduce performance materially, which means the original random-negative task was too easy.")
    else:
        interpretation.append("Hard negatives do not collapse performance, so the signal survives more difficult negatives.")

    if old_baseline is not None and mlp_concat["test_metrics"]["accuracy"] < old_baseline["final"]["accuracy"] - 0.03:
        interpretation.append("Performance dropped under phrase-level splits, which is evidence that the old setup likely benefited from leakage or overlap.")
    else:
        interpretation.append("Phrase-level splits did not materially hurt performance, so leakage is less likely to explain the results.")

    write_final_report(
        args.results_dir,
        rows=[
            ("Baseline (old)", old_baseline),
            ("Cosine", cosine),
            ("Linear", linear),
            ("MLP (concat)", mlp_concat),
            ("MLP (symmetric)", mlp_sym),
            ("Hard negatives", hard_mlp),
        ],
        interpretation=interpretation,
        invalid_flags=invalid_flags,
    )


if __name__ == "__main__":
    main()
