"""
04_probe_word2vec.py
────────────────────
Train lightweight CPU-only probes on top of frozen Word2Vec phrase vectors.

Supported probes
----------------
1. MLP classifier over concatenated phrase vectors
2. Siamese MLP with:
   - contrastive loss (preferred)
   - cosine+BCE baseline

This script does not update the Word2Vec embeddings. It evaluates how much
semantic structure is already present in the frozen vector space.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
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


def split_tokens(text: str) -> list[str]:
    return text.lower().split()


def phrase_to_vector(text: str, wv: KeyedVectors) -> np.ndarray | None:
    tokens = [token for token in split_tokens(text) if token in wv]
    if not tokens:
        return None
    vec = np.mean([wv[token] for token in tokens], axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return None
    return vec / norm


def pair_split(a: str, b: str, val_pct: int, test_pct: int) -> str:
    key = f"{a}\t{b}".encode("utf-8")
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


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


def binary_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    labels = labels.astype(np.int64)

    accuracy = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        roc_auc = float("nan")
    else:
        ranks = rankdata(probs)
        rank_sum_pos = float(ranks[labels == 1].sum())
        roc_auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": float(roc_auc),
    }


@dataclass
class PairIndexSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def preprocess_pairs(
    pairs_path: str,
    wv: KeyedVectors,
    val_pct: int,
    test_pct: int,
    max_pairs: int | None,
) -> tuple[np.ndarray, list[str], PairIndexSplits]:
    phrase_to_idx: dict[str, int] = {}
    phrase_vectors: list[np.ndarray] = []
    invalid_phrases: set[str] = set()
    train_pairs: list[tuple[int, int]] = []
    val_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []

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

    seen_pairs = 0
    with open(pairs_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Filtering UMLS pairs", unit="pair"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            a_idx = get_phrase_idx(a)
            b_idx = get_phrase_idx(b)
            if a_idx is None or b_idx is None or a_idx == b_idx:
                continue

            split = pair_split(a, b, val_pct=val_pct, test_pct=test_pct)
            pair = (a_idx, b_idx)
            if split == "train":
                train_pairs.append(pair)
            elif split == "val":
                val_pairs.append(pair)
            else:
                test_pairs.append(pair)

            seen_pairs += 1
            if max_pairs is not None and seen_pairs >= max_pairs:
                break

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError(
            "Pair preprocessing produced an empty train/val/test split. "
            "Lower the split ratios or increase available pairs."
        )

    matrix = np.vstack(phrase_vectors).astype(np.float16)
    phrases = [None] * len(phrase_to_idx)
    for phrase, idx in phrase_to_idx.items():
        phrases[idx] = phrase

    log.info(
        "Preprocessed pairs: train=%s val=%s test=%s phrases=%s",
        f"{len(train_pairs):,}",
        f"{len(val_pairs):,}",
        f"{len(test_pairs):,}",
        f"{len(phrases):,}",
    )

    return (
        matrix,
        phrases,
        PairIndexSplits(
            train=np.asarray(train_pairs, dtype=np.int32),
            val=np.asarray(val_pairs, dtype=np.int32),
            test=np.asarray(test_pairs, dtype=np.int32),
        ),
    )


class ProbePairDataset(Dataset):
    def __init__(
        self,
        pair_indices: np.ndarray,
        phrase_vectors: np.ndarray,
        seed: int,
        deterministic: bool = False,
    ):
        self.pairs = pair_indices
        self.vectors = phrase_vectors
        self.num_phrases = phrase_vectors.shape[0]
        self.seed = seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.pairs) * 2

    def _negative_index(self, anchor_idx: int, positive_idx: int, base_idx: int) -> int:
        if self.deterministic:
            rng = np.random.default_rng(self.seed + base_idx)
            candidate = int(rng.integers(0, self.num_phrases))
        else:
            candidate = random.randrange(self.num_phrases)
        while candidate == anchor_idx or candidate == positive_idx:
            candidate = (candidate + 1) % self.num_phrases
        return candidate

    def __getitem__(self, idx: int):
        base_idx = idx // 2
        anchor_idx, positive_idx = self.pairs[base_idx]
        label = 1.0 if idx % 2 == 0 else 0.0

        left = self.vectors[anchor_idx].astype(np.float32)
        if label == 1.0:
            right = self.vectors[positive_idx].astype(np.float32)
        else:
            neg_idx = self._negative_index(anchor_idx, positive_idx, base_idx)
            right = self.vectors[neg_idx].astype(np.float32)

        return (
            torch.from_numpy(left),
            torch.from_numpy(right),
            torch.tensor(label, dtype=torch.float32),
        )


class MLPProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1)).squeeze(-1)


class SiameseProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(left), self.encode(right)


def contrastive_loss(z_left: torch.Tensor, z_right: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    dist = torch.norm(z_left - z_right, dim=-1)
    pos = labels * dist.pow(2)
    neg = (1.0 - labels) * torch.clamp(margin - dist, min=0.0).pow(2)
    return (pos + neg).mean()


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0

    for left, right, labels in tqdm(loader, desc="train", unit="batch"):
        left = left.to(torch.float32)
        right = right.to(torch.float32)
        labels = labels.to(torch.float32)

        optimiser.zero_grad()

        if args.model == "mlp":
            logits = model(left, right)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            z_left, z_right = model(left, right)
            if args.loss == "contrastive":
                loss = contrastive_loss(z_left, z_right, labels, margin=args.margin)
            else:
                cosine = F.cosine_similarity(z_left, z_right)
                probs = (cosine + 1.0) * 0.5
                loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), labels)

        loss.backward()
        optimiser.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def evaluate(loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for left, right, labels in tqdm(loader, desc="eval", unit="batch"):
            left = left.to(torch.float32)
            right = right.to(torch.float32)
            labels = labels.to(torch.float32)

            if args.model == "mlp":
                logits = model(left, right)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                probs = torch.sigmoid(logits)
            else:
                z_left, z_right = model(left, right)
                cosine = F.cosine_similarity(z_left, z_right)
                if args.loss == "contrastive":
                    dist = torch.norm(z_left - z_right, dim=-1)
                    loss = contrastive_loss(z_left, z_right, labels, margin=args.margin)
                    probs = torch.sigmoid(args.margin - dist)
                else:
                    probs = (cosine + 1.0) * 0.5
                    loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), labels)

            total_loss += float(loss.item())
            probs_all.append(probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    probs_np = np.concatenate(probs_all)
    labels_np = np.concatenate(labels_all)
    return total_loss / max(len(loader), 1), binary_metrics(probs_np, labels_np)


def build_model(args: argparse.Namespace, dim: int) -> nn.Module:
    if args.model == "mlp":
        return MLPProbe(dim)
    return SiameseProbe(dim)


def save_checkpoint(
    output_path: str,
    model: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, float],
    vector_dim: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": vars(args),
        "metrics": metrics,
        "vector_dim": vector_dim,
    }
    torch.save(payload, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CPU-only probe models on Word2Vec phrase vectors")
    parser.add_argument("--w2v_bin", required=True, help="Path to baseline Word2Vec .bin file")
    parser.add_argument("--pairs", required=True, help="Path to UMLS synonym pairs file")
    parser.add_argument("--model", choices=["mlp", "siamese"], required=True)
    parser.add_argument("--loss", choices=["bce", "contrastive"], default="contrastive")
    parser.add_argument("--output", required=True, help="Path to save the trained probe checkpoint")
    parser.add_argument("--metrics_out", default=None, help="Optional JSON metrics output path")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (must be <= 512)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--val_pct", type=int, default=10)
    parser.add_argument("--test_pct", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap for faster experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=8, help="CPU thread count for PyTorch")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size > 512:
        raise ValueError("Probe training is constrained to batch_size <= 512.")
    if args.model == "mlp" and args.loss != "bce":
        raise ValueError("MLP probe only supports --loss bce.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    device = torch.device("cpu")
    log.info("Device: %s", device)
    log.info("Loading Word2Vec from %s", args.w2v_bin)
    wv = KeyedVectors.load_word2vec_format(args.w2v_bin, binary=True)

    phrase_vectors, phrases, splits = preprocess_pairs(
        args.pairs,
        wv,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_pairs=args.max_pairs,
    )
    log.info(
        "Phrase matrix shape=%s dtype=%s estimated_ram=%.2f MB",
        phrase_vectors.shape,
        phrase_vectors.dtype,
        phrase_vectors.nbytes / 1024 / 1024,
    )
    del phrases

    train_ds = ProbePairDataset(splits.train, phrase_vectors, seed=args.seed, deterministic=False)
    val_ds = ProbePairDataset(splits.val, phrase_vectors, seed=args.seed, deterministic=True)
    test_ds = ProbePairDataset(splits.test, phrase_vectors, seed=args.seed + 7, deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args, dim=wv.vector_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -math.inf
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %s/%s", epoch, args.epochs)
        train_loss = train_epoch(train_loader, model, optimiser, args)
        val_loss, val_metrics = evaluate(val_loader, model, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        log.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f val_acc=%.4f val_auc=%.4f val_p=%.4f val_r=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["roc_auc"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate(test_loader, model, args)
    final_metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "best_val_roc_auc": best_val_auc,
        "num_train_pairs": int(len(splits.train)),
        "num_val_pairs": int(len(splits.val)),
        "num_test_pairs": int(len(splits.test)),
        "num_phrases": int(phrase_vectors.shape[0]),
    }

    save_checkpoint(args.output, model, args, final_metrics, vector_dim=wv.vector_size)
    log.info(
        "test_loss=%.4f test_acc=%.4f test_auc=%.4f test_p=%.4f test_r=%.4f",
        final_metrics["test_loss"],
        final_metrics["accuracy"],
        final_metrics["roc_auc"],
        final_metrics["precision"],
        final_metrics["recall"],
    )
    log.info("Saved checkpoint -> %s", args.output)

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as handle:
            json.dump({"history": history, "final": final_metrics}, handle, indent=2)
        log.info("Saved metrics -> %s", args.metrics_out)


if __name__ == "__main__":
    main()
