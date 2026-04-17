from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir
from ..utils.metrics import cosine_similarity_matrix
from .data import load_kb, load_queries


def _cache_paths(cache_dir: str | Path, model_name: str, dataset_name: str) -> tuple[Path, Path]:
    cache_dir = ensure_dir(cache_dir)
    stem = f"{model_name}_{dataset_name}"
    return cache_dir / f"{stem}_embeddings.npy", cache_dir / f"{stem}_meta.json"


def evaluate_entity_linking(embedder, kb_path: str, queries_path: str, batch_size: int, cache_dir: str | Path | None = None, smoke_limit: int | None = None) -> dict:
    kb = load_kb(kb_path)
    queries = load_queries(queries_path)
    if smoke_limit:
        queries = queries[:smoke_limit]
    if not kb or not queries:
        return {"acc@1": 0.0, "acc@5": 0.0, "mrr": 0.0, "queries": 0}

    kb_names = [row["name"] for row in kb]
    kb_ids = [row["entity_id"] for row in kb]
    dataset_name = Path(kb_path).stem

    kb_embeddings = None
    if cache_dir:
        emb_path, _ = _cache_paths(cache_dir, getattr(embedder, "name", "model"), dataset_name)
        if emb_path.exists():
            kb_embeddings = np.load(emb_path)
    if kb_embeddings is None:
        kb_embeddings = embedder.encode(kb_names, batch_size=batch_size)
        if cache_dir:
            emb_path, _ = _cache_paths(cache_dir, getattr(embedder, "name", "model"), dataset_name)
            np.save(emb_path, kb_embeddings)

    mention_embeddings = embedder.encode([row["mention"] for row in queries], batch_size=batch_size)
    scores = cosine_similarity_matrix(mention_embeddings, kb_embeddings)
    ranks = []
    top1 = 0
    top5 = 0
    for idx, query in enumerate(queries):
        ranking = np.argsort(-scores[idx])
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
    }
