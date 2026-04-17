import json
from pathlib import Path


def load_nli_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows.append({"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            premise, hypothesis, label = line.split("\t")
            rows.append({"premise": premise, "hypothesis": hypothesis, "label": label})
    return rows
