from __future__ import annotations

import csv
from pathlib import Path

from ..utils.io import ensure_dir, write_json


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install with: pip install datasets"
        ) from exc
    return load_dataset


def _load_public_dataset(dataset_name: str):
    load_dataset = _require_datasets()
    return load_dataset(dataset_name, trust_remote_code=True)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)


def _first(row: dict, names: list[str]):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _extract_text_from_entity(entity: dict, row: dict) -> str | None:
    text = _first(entity, ["text", "mention", "name"])
    if isinstance(text, list):
        return " ".join(str(part) for part in text).strip()
    if isinstance(text, str):
        return text.strip()

    passages = row.get("passages") or row.get("document") or []
    offsets = entity.get("offsets") or entity.get("locations") or []
    if passages and offsets:
        try:
            first_passage = passages[0]
            passage_text = _first(first_passage, ["text", "passage_text"])
            if isinstance(passage_text, list):
                passage_text = " ".join(passage_text)
            start, end = offsets[0]
            return str(passage_text)[int(start):int(end)].strip()
        except Exception:
            return None
    return None


def _extract_entity_id(entity: dict) -> str | None:
    for key in ["entity_id", "db_id", "id"]:
        value = entity.get(key)
        if isinstance(value, str) and value:
            return value
    normalized = entity.get("normalized") or entity.get("db_ids") or entity.get("identifiers")
    if isinstance(normalized, list) and normalized:
        item = normalized[0]
        if isinstance(item, dict):
            db_name = item.get("db_name") or item.get("db")
            db_id = item.get("db_id") or item.get("id")
            if db_name and db_id:
                return f"{db_name}:{db_id}"
            if db_id:
                return str(db_id)
        if isinstance(item, str):
            return item
    if isinstance(normalized, str) and normalized:
        return normalized
    return None


def _extract_entity_type(entity: dict) -> str:
    entity_type = _first(entity, ["type", "label", "entity_type", "semantic_type"])
    if isinstance(entity_type, list):
        return str(entity_type[0]).upper()
    return str(entity_type or "").upper()


def prepare_sts_biosses(output_root: str | Path) -> dict:
    dataset = _load_public_dataset("bigbio/biosses")
    rows = []
    for split_name, split in dataset.items():
        split_rows = []
        for row in split:
            s1 = _first(row, ["sentence1", "text_1", "sent1"])
            s2 = _first(row, ["sentence2", "text_2", "sent2"])
            score = _first(row, ["score", "label"])
            if s1 is None or s2 is None or score is None:
                continue
            split_rows.append([str(s1), str(s2), str(float(score))])
        rows.extend(split_rows)
        _write_tsv(Path(output_root) / "sts" / f"biosses_{split_name}.tsv", split_rows)
    return {"dataset": "bigbio/biosses", "rows": len(rows)}


def prepare_nli4ct(output_root: str | Path) -> dict:
    dataset = _load_public_dataset("tasksource/nli4ct")
    written = {}
    label_names = None
    for split_name, split in dataset.items():
        if hasattr(split, "features") and "label" in split.features:
            feature = split.features["label"]
            if hasattr(feature, "names"):
                label_names = feature.names
        rows = []
        for row in split:
            premise = _first(row, ["premise", "sentence1", "context"])
            hypothesis = _first(row, ["hypothesis", "sentence2", "statement"])
            label = _first(row, ["label", "gold_label"])
            if premise is None or hypothesis is None or label is None:
                continue
            if isinstance(label, int) and label_names and 0 <= label < len(label_names):
                label = label_names[label]
            rows.append([str(premise), str(hypothesis), str(label)])
        _write_tsv(Path(output_root) / "nli" / f"{split_name}.tsv", rows)
        written[split_name] = len(rows)
    return {"dataset": "tasksource/nli4ct", "splits": written}


def _prepare_entity_linking_dataset(dataset_name: str, output_prefix: str, output_root: str | Path) -> dict:
    dataset = _load_public_dataset(dataset_name)

    kb: dict[str, str] = {}
    queries_by_type: dict[str, list[list[str]]] = {}
    total_queries = 0

    for split_name, split in dataset.items():
        split_queries: list[list[str]] = []
        split_queries_by_type: dict[str, list[list[str]]] = {}

        for row in split:
            entities = row.get("entities")
            if not entities:
                continue
            for entity in entities:
                entity_id = _extract_entity_id(entity)
                mention = _extract_text_from_entity(entity, row)
                if not entity_id or not mention:
                    continue
                mention = mention.strip()
                if not mention:
                    continue
                kb.setdefault(entity_id, mention)
                split_queries.append([mention, entity_id])
                entity_type = _extract_entity_type(entity)
                if entity_type:
                    split_queries_by_type.setdefault(entity_type, []).append([mention, entity_id])
                total_queries += 1

        queries_by_type[split_name] = split_queries
        _write_tsv(Path(output_root) / "entity_linking" / f"{output_prefix}_{split_name}_queries.tsv", split_queries)

        for entity_type, rows in split_queries_by_type.items():
            suffix = entity_type.lower()
            _write_tsv(
                Path(output_root) / "entity_linking" / f"{output_prefix}_{suffix}_{split_name}_queries.tsv",
                rows,
            )

    kb_rows = [[entity_id, name] for entity_id, name in sorted(kb.items())]
    _write_tsv(Path(output_root) / "entity_linking" / f"{output_prefix}_kb.tsv", kb_rows)
    return {"dataset": dataset_name, "kb_entities": len(kb_rows), "queries": total_queries}


def prepare_ncbi_disease(output_root: str | Path) -> dict:
    return _prepare_entity_linking_dataset("ncbi/ncbi_disease", "ncbi_disease", output_root)


def prepare_bc5cdr(output_root: str | Path) -> dict:
    return _prepare_entity_linking_dataset("bigbio/bc5cdr", "bc5cdr", output_root)


def prepare_all_public_datasets(output_root: str | Path) -> dict:
    summary = {
        "biosses": prepare_sts_biosses(output_root),
        "nli4ct": prepare_nli4ct(output_root),
        "ncbi_disease": prepare_ncbi_disease(output_root),
        "bc5cdr": prepare_bc5cdr(output_root),
    }
    write_json(Path(output_root) / "download_summary.json", summary)
    return summary
