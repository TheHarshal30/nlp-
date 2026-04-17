from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import write_jsonl


ALLOWED_RELATIONS = {
    "may_treat",
    "has_finding_site",
    "has_manifestation",
}


@dataclass
class RelationExample:
    anchor_cui: str
    anchor_text: str
    relation: str
    positive_cui: str
    positive_text: str


def _pick_term(current: tuple[int, str] | None, candidate: str, is_pref: bool, ts: str, suppress: str) -> tuple[int, str]:
    score = 0
    if suppress == "N":
        score += 4
    if is_pref:
        score += 2
    if ts == "P":
        score += 1
    if current is None or score > current[0] or (score == current[0] and len(candidate) < len(current[1])):
        return (score, candidate)
    return current


def load_cui_text_map(mrconso_file: str | Path) -> dict[str, str]:
    chosen: dict[str, tuple[int, str]] = {}
    with Path(mrconso_file).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            cui = parts[0]
            lat = parts[1]
            ts = parts[2]
            is_pref = parts[6] == "Y"
            text = parts[14].strip().lower()
            suppress = parts[16]
            if lat != "ENG" or not text:
                continue
            chosen[cui] = _pick_term(chosen.get(cui), text, is_pref, ts, suppress)
    return {cui: payload[1] for cui, payload in chosen.items()}


def normalize_relation(rel: str, rela: str, allowed_relations: set[str] | None = None) -> str | None:
    allowed_relations = allowed_relations or ALLOWED_RELATIONS
    candidate = (rela or rel or "").strip().lower()
    if candidate in allowed_relations:
        return candidate
    return None


def extract_relation_examples(mrrel_file: str | Path, cui_to_text: dict[str, str], allowed_relations: set[str] | None = None) -> list[RelationExample]:
    examples: list[RelationExample] = []
    with Path(mrrel_file).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            anchor_cui = parts[0]
            rel = parts[3]
            positive_cui = parts[4]
            rela = parts[7]
            relation = normalize_relation(rel, rela, allowed_relations=allowed_relations)
            if relation is None:
                continue
            anchor_text = cui_to_text.get(anchor_cui)
            positive_text = cui_to_text.get(positive_cui)
            if not anchor_text or not positive_text or anchor_text == positive_text:
                continue
            examples.append(
                RelationExample(
                    anchor_cui=anchor_cui,
                    anchor_text=anchor_text,
                    relation=relation,
                    positive_cui=positive_cui,
                    positive_text=positive_text,
                )
            )
    return examples


def filter_examples_by_encoder_coverage(examples: list[RelationExample], encoder) -> list[RelationExample]:
    filtered = []
    for example in examples:
        if encoder.has_coverage(example.anchor_text) and encoder.has_coverage(example.positive_text):
            filtered.append(example)
    return filtered


def deduplicate_examples(examples: list[RelationExample]) -> list[RelationExample]:
    seen = set()
    deduped = []
    for example in examples:
        key = (example.anchor_text, example.relation, example.positive_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def write_examples(path: str | Path, examples: list[RelationExample]) -> None:
    write_jsonl(
        path,
        [
            {
                "anchor_cui": ex.anchor_cui,
                "anchor_text": ex.anchor_text,
                "relation": ex.relation,
                "positive_cui": ex.positive_cui,
                "positive_text": ex.positive_text,
            }
            for ex in examples
        ],
    )


def relation_index(examples: list[RelationExample]) -> dict[str, dict[str, set[str]]]:
    index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for ex in examples:
        index[ex.relation][ex.anchor_text].add(ex.positive_text)
    return index
