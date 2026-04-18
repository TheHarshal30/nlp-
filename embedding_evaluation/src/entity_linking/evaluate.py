from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir
from ..utils.metrics import cosine_similarity_matrix
from .data import load_kb, load_queries


def _cache_paths(cache_dir: str | Path, model_name: str, dataset_name: str) -> tuple[Path, Path]:
    cache_dir = ensure_dir(cache_dir)
    stem = f"{model_name}_{dataset_name}"
    return cache_dir / f"{stem}_embeddings.npy", cache_dir / f"{stem}_meta.json"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def evaluate_entity_linking(
    embedder,
    kb_path: str,
    queries_path: str,
    batch_size: int,
    cache_dir: str | Path | None = None,
    smoke_limit: int | None = None,
    normalize: bool = True,
    rerank_top_k: int = 50,
    rerank_alpha: float = 1.0,
) -> dict:
    kb = load_kb(kb_path)
    queries = load_queries(queries_path)
    if smoke_limit:
        queries = queries[:smoke_limit]
    if not kb or not queries:
        return {"acc@1": 0.0, "acc@5": 0.0, "mrr": 0.0, "queries": 0}

    kb_names = [row["name"] for row in kb]
    kb_ids = [row["entity_id"] for row in kb]
    dataset_name = Path(kb_path).stem
    cache_model_name = getattr(embedder, "name", "model")
    cache_mode = getattr(embedder, "inference_mode", "base")
    cache_key = f"{cache_model_name}_{cache_mode}"

    kb_embeddings = None
    if cache_dir:
        emb_path, _ = _cache_paths(cache_dir, cache_key, dataset_name)
        if emb_path.exists():
            kb_embeddings = np.load(emb_path)
    if kb_embeddings is None:
        kb_embeddings = embedder.encode(kb_names, batch_size=batch_size)
        if cache_dir:
            emb_path, _ = _cache_paths(cache_dir, cache_key, dataset_name)
            np.save(emb_path, kb_embeddings)

    kb_norm = _normalize_rows(kb_embeddings) if normalize else kb_embeddings
    mention_embeddings = embedder.encode([row["mention"] for row in queries], batch_size=batch_size)
    mention_norm = _normalize_rows(mention_embeddings) if normalize else mention_embeddings
    scores = cosine_similarity_matrix(mention_norm, kb_norm) if normalize else cosine_similarity_matrix(mention_embeddings, kb_embeddings)
    ranks = []
    top1 = 0
    top5 = 0
    for idx, query in enumerate(queries):
        ranking = np.argsort(-scores[idx])
        if rerank_top_k and rerank_top_k > 0:
            top_indices = ranking[: min(rerank_top_k, len(ranking))]
            cosine_scores = scores[idx][top_indices]
            dot_scores = mention_embeddings[idx] @ kb_embeddings[top_indices].T
            reranked = np.argsort(-(rerank_alpha * cosine_scores + (1.0 - rerank_alpha) * dot_scores))
            ranking = np.concatenate([top_indices[reranked], ranking[len(top_indices):]])
        ranked_ids = [kb_ids[pos] for pos in ranking]
        gold = query["entity_id"]
        rank = ranked_ids.index(gold) + 1 if gold in ranked_ids else len(ranked_ids) + 1
        ranks.append(rank)
        top1 += rank <= 1
        top5 += rank <= 5
    return {
        "acc@1": float(top1 / len(queries)),
        "acc@5": float(top5 / len(queries)),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)),
        "queries": len(queries),
        "normalize": bool(normalize),
        "rerank_top_k": int(rerank_top_k),
        "rerank_alpha": float(rerank_alpha),
    }
