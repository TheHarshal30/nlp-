from collections import defaultdict

import numpy as np
import torch

from ..data.umls import RelationExample


def _rank_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0, "mrr": 0.0, "queries": 0}
    return {
        "hits@1": float(np.mean([rank <= 1 for rank in ranks])),
        "hits@5": float(np.mean([rank <= 5 for rank in ranks])),
        "hits@10": float(np.mean([rank <= 10 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "queries": len(ranks),
    }


def evaluate_disease_drug_retrieval(model, examples: list[RelationExample], relation_name: str, candidate_batch_size: int, device: torch.device) -> dict[str, float]:
    relation_examples = [example for example in examples if example.relation == relation_name]
    if not relation_examples:
        return {"hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0, "mrr": 0.0, "queries": 0}

    anchor_to_positives = defaultdict(set)
    candidate_texts = set()
    for example in relation_examples:
        anchor_to_positives[example.anchor_text].add(example.positive_text)
        candidate_texts.add(example.positive_text)

    candidate_texts = sorted(candidate_texts)
    model.eval()
    ranks = []
    with torch.inference_mode():
        relation_ids = model.relation_ids([relation_name], device=device)
        candidate_vectors = []
        for start in range(0, len(candidate_texts), candidate_batch_size):
            chunk = candidate_texts[start:start + candidate_batch_size]
            candidate_vectors.append(model.encode_texts(chunk).detach().cpu())
        candidate_matrix = torch.cat(candidate_vectors, dim=0)
        normalized_candidates = torch.nn.functional.normalize(candidate_matrix, dim=-1)

        for anchor_text, positives in anchor_to_positives.items():
            anchor_projection = model.encode_texts([anchor_text])
            anchor_query = model.compose_anchor(anchor_projection, relation_ids).detach().cpu()
            anchor_query = torch.nn.functional.normalize(anchor_query, dim=-1)
            scores = torch.mm(anchor_query, normalized_candidates.T).squeeze(0).numpy()
            ranking = np.argsort(-scores)
            positive_indices = {candidate_texts.index(text) for text in positives if text in candidate_texts}
            if not positive_indices:
                continue
            best_rank = min(int(np.where(ranking == idx)[0][0]) + 1 for idx in positive_indices)
            ranks.append(best_rank)

    return _rank_metrics(ranks)
