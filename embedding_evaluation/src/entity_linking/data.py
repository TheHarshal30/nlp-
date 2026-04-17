import json
from pathlib import Path


def _read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            rows.append(parts)
    return rows


def load_kb(path: str | Path) -> list[dict]:
    rows = _read_rows(path)
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return [{"entity_id": row["entity_id"], "name": row.get("name") or row.get("canonical_name")} for row in rows]
    return [{"entity_id": row[0], "name": row[1]} for row in rows]


def load_queries(path: str | Path) -> list[dict]:
    rows = _read_rows(path)
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return [{"mention": row["mention"], "entity_id": row["entity_id"]} for row in rows]
    return [{"mention": row[0], "entity_id": row[1]} for row in rows]
