import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.pretrained import encode_texts_in_batches, load_base_encoder, normalize_rows
from ..data.triples import SplitBundle, TripletDataset, collect_relation_terms, split_by_anchor, write_split_jsonl
from ..data.umls import (
    deduplicate_examples,
    extract_relation_examples,
    filter_examples_by_encoder_coverage,
    load_cui_text_map,
    write_examples,
)
from ..losses.triplet import triplet_margin_loss
from ..mining.hard_negative import HardNegativeMiner
from ..models.relation_triplet import RelationAwareTripletModel
from ..utils.io import ensure_dir, read_json, write_json
from .retrieval import evaluate_disease_drug_retrieval


log = logging.getLogger(__name__)


def prepare_relation_data(config: dict, encoder) -> tuple[list, SplitBundle, Path]:
    data_root = ensure_dir(config["data_root"])
    processed_dir = ensure_dir(Path(data_root) / "processed")
    triples_dir = ensure_dir(Path(data_root) / "triples")
    processed_path = processed_dir / "relation_examples.jsonl"

    cui_to_text = load_cui_text_map(config["mrconso_file"])
    examples = extract_relation_examples(
        config["relations_file"],
        cui_to_text,
        allowed_relations=set(config.get("allowed_relations", [])) or None,
    )
    examples = deduplicate_examples(filter_examples_by_encoder_coverage(examples, encoder))
    write_examples(processed_path, examples)

    splits = split_by_anchor(
        examples,
        train_ratio=config["train_split"],
        val_ratio=config["val_split"],
        seed=config["seed"],
    )
    write_split_jsonl(triples_dir, "train", splits.train)
    write_split_jsonl(triples_dir, "val", splits.val)
    write_split_jsonl(triples_dir, "test", splits.test)
    return examples, splits, triples_dir


def _collate_batch(batch, miner):
    anchors = [example.anchor_text for example in batch]
    relations = [example.relation for example in batch]
    positives = [example.positive_text for example in batch]
    negatives = [miner.sample(example.anchor_text, example.relation, example.positive_text) for example in batch]
    return anchors, relations, positives, negatives


def _metadata(config: dict, model: RelationAwareTripletModel, split_bundle: SplitBundle, retrieval_metrics: dict[str, float]) -> dict:
    return {
        "base_model": config["base_model"],
        "base_model_path": config["base_model_path"],
        "freeze_base": config["freeze_base"],
        "projection_dim": model.projection_head.net[-1].out_features,
        "relation_dim": model.relation_embeddings.embedding_dim,
        "relations": model.relations,
        "distance": config["distance"],
        "margin": config["margin"],
        "train_examples": len(split_bundle.train),
        "val_examples": len(split_bundle.val),
        "test_examples": len(split_bundle.test),
        "retrieval_relation": config["evaluation"]["relation"],
        "retrieval_metrics": retrieval_metrics,
    }


def train_triplet_model(config: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_encoder = load_base_encoder(config["base_model"], config["base_model_path"], freeze=config["freeze_base"])
    examples, split_bundle, _ = prepare_relation_data(config, base_encoder)

    unique_terms = sorted({example.anchor_text for example in examples} | {example.positive_text for example in examples})
    term_to_index = {term: idx for idx, term in enumerate(unique_terms)}
    base_embeddings = encode_texts_in_batches(base_encoder, unique_terms, batch_size=config["batch_size"], device=device).numpy()
    base_embeddings = normalize_rows(base_embeddings)

    miner = HardNegativeMiner(
        examples=split_bundle.train,
        term_embeddings=base_embeddings,
        term_to_index=term_to_index,
        strategy=config["negative_sampling"]["strategy"],
        pool_size=config["negative_sampling"]["pool_size"],
        seed=config["negative_sampling"]["random_seed"],
    )

    relations = sorted({example.relation for example in examples})
    model = RelationAwareTripletModel(
        base_encoder=base_encoder,
        relations=relations,
        projection_dim=config["projection_dim"],
        relation_dim=config["relation_dim"],
    ).to(device)

    if config["freeze_base"]:
        params = list(model.projection_head.parameters()) + list(model.relation_embeddings.parameters()) + list(model.relation_adapter.parameters())
    else:
        params = list(model.parameters())

    optimiser = torch.optim.AdamW(
        params,
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    loader = DataLoader(
        TripletDataset(split_bundle.train),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=lambda batch: _collate_batch(batch, miner),
    )

    log_rows = []
    best_metrics = None
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_anchor_norm = 0.0
        for anchors, relations_batch, positives, negatives in loader:
            optimiser.zero_grad()
            relation_ids = model.relation_ids(relations_batch, device=device)
            anchor_projection = model.encode_texts(anchors)
            anchor_query = model.compose_anchor(anchor_projection, relation_ids)
            positive_projection = model.encode_texts(positives)
            negative_projection = model.encode_texts(negatives)
            loss = triplet_margin_loss(
                anchor_query,
                positive_projection,
                negative_projection,
                margin=config["margin"],
                distance=config["distance"],
            )
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            total_batches += 1
            total_anchor_norm += float(anchor_query.detach().norm(dim=-1).mean().item())

        val_metrics = evaluate_disease_drug_retrieval(
            model,
            split_bundle.val,
            relation_name=config["evaluation"]["relation"],
            candidate_batch_size=config["evaluation"]["candidate_batch_size"],
            device=device,
        )
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_batches, 1),
            "anchor_norm": total_anchor_norm / max(total_batches, 1),
            "val_hits@1": val_metrics["hits@1"],
            "val_hits@5": val_metrics["hits@5"],
            "val_hits@10": val_metrics["hits@10"],
            "val_mrr": val_metrics["mrr"],
        }
        log.info(
            "epoch=%s loss=%.4f anchor_norm=%.4f val_hits@1=%.4f val_mrr=%.4f",
            row["epoch"],
            row["loss"],
            row["anchor_norm"],
            row["val_hits@1"],
            row["val_mrr"],
        )
        log_rows.append(row)
        best_metrics = val_metrics

    test_metrics = evaluate_disease_drug_retrieval(
        model,
        split_bundle.test,
        relation_name=config["evaluation"]["relation"],
        candidate_batch_size=config["evaluation"]["candidate_batch_size"],
        device=device,
    )

    output_root = ensure_dir(config["output_root"])
    weights_dir = ensure_dir(Path(output_root) / "weights")
    torch.save(model.projection_head.state_dict(), Path(weights_dir) / "projection_head.pt")
    torch.save(model.relation_embeddings.state_dict(), Path(weights_dir) / "relation_embeddings.pt")
    torch.save(model.relation_adapter.state_dict(), Path(weights_dir) / "relation_adapter.pt")
    if not config["freeze_base"]:
        torch.save(model.base_encoder.state_dict(), Path(weights_dir) / "base_encoder.pt")
    write_json(Path(output_root) / "training_log.json", {"epochs": log_rows})
    write_json(Path(output_root) / "retrieval_metrics.json", {"validation": best_metrics, "test": test_metrics})
    write_json(Path(output_root) / "metadata.json", _metadata(config, model, split_bundle, test_metrics))

    return {
        "output_root": str(output_root),
        "training_log": log_rows,
        "test_metrics": test_metrics,
        "train_examples": len(split_bundle.train),
        "val_examples": len(split_bundle.val),
        "test_examples": len(split_bundle.test),
    }
