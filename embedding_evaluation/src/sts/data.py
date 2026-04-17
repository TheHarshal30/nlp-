import json
from pathlib import Path


def load_sts_pairs(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows.append({"sentence1": row["sentence1"], "sentence2": row["sentence2"], "score": float(row["score"])})
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, s2, score = line.split("\t")
            rows.append({"sentence1": s1, "sentence2": s2, "score": float(score)})
    return rows
