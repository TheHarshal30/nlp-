from pathlib import Path

from ..entity_linking.evaluate import evaluate_entity_linking
from ..models.loader import load_embedder
from ..nli.evaluate import evaluate_nli
from ..sts.evaluate import evaluate_sts
from ..utils.io import ensure_dir, write_json


def run_all(config: dict) -> dict:
    inference_cfg = config.get("inference", {})
    embedder = load_embedder(config["model_path"], inference=inference_cfg)
    model_name = config.get("model_name") or getattr(embedder, "name", Path(config["model_path"]).name)
    batch_size = config.get("batch_size", 32)
    smoke_limit = config.get("smoke_limit")
    results: dict[str, dict] = {}

    el_cfg = config.get("entity_linking", {})
    if el_cfg.get("enabled", False):
        results["entity_linking"] = evaluate_entity_linking(
            embedder=embedder,
            kb_path=el_cfg["kb_path"],
            queries_path=el_cfg["queries_path"],
            batch_size=batch_size,
            cache_dir=el_cfg.get("cache_dir"),
            smoke_limit=smoke_limit,
            normalize=el_cfg.get("normalize", True),
            rerank_top_k=el_cfg.get("rerank_top_k", 50),
            rerank_alpha=el_cfg.get("rerank_alpha", 1.0),
        )

    sts_cfg = config.get("sts", {})
    if sts_cfg.get("enabled", False):
        results["sts"] = evaluate_sts(
            embedder=embedder,
            pairs_path=sts_cfg["pairs_path"],
            batch_size=batch_size,
            smoke_limit=smoke_limit,
        )

    nli_cfg = config.get("nli", {})
    if nli_cfg.get("enabled", False):
        results["nli"] = evaluate_nli(
            embedder=embedder,
            train_path=nli_cfg["train_path"],
            dev_path=nli_cfg["dev_path"],
            test_path=nli_cfg["test_path"],
            batch_size=batch_size,
            epochs=nli_cfg.get("epochs", 20),
            lr=nli_cfg.get("lr", 1e-3),
            hidden_dim=nli_cfg.get("hidden_dim", 128),
            seed=nli_cfg.get("seed", 42),
            smoke_limit=smoke_limit,
        )

    mode = inference_cfg.get("inference_mode") or ("projected" if inference_cfg.get("use_projection") else "base")
    results["model"] = {"name": model_name, "path": config["model_path"], "mode": mode}
    output_path = ensure_dir("embedding_evaluation/results") / f"{model_name}_{mode}.json"
    write_json(output_path, results)
    return {"output_path": str(output_path), "results": results}
