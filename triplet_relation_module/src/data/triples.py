import random
from collections import defaultdict
from dataclasses import dataclass

from torch.utils.data import Dataset

from .umls import RelationExample
from ..utils.io import write_jsonl


@dataclass
class SplitBundle:
    train: list[RelationExample]
    val: list[RelationExample]
    test: list[RelationExample]


def split_by_anchor(examples: list[RelationExample], train_ratio: float, val_ratio: float, seed: int) -> SplitBundle:
    anchors = sorted({example.anchor_text for example in examples})
    rng = random.Random(seed)
    rng.shuffle(anchors)
    total = len(anchors)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)
    train_anchors = set(anchors[:train_cut])
    val_anchors = set(anchors[train_cut:val_cut])
    test_anchors = set(anchors[val_cut:])
    train = [example for example in examples if example.anchor_text in train_anchors]
    val = [example for example in examples if example.anchor_text in val_anchors]
    test = [example for example in examples if example.anchor_text in test_anchors]
    return SplitBundle(train=train, val=val, test=test)


def write_split_jsonl(root, split_name: str, examples: list[RelationExample]) -> None:
    write_jsonl(
        root / f"{split_name}.jsonl",
        [
            {
                "anchor_text": ex.anchor_text,
                "relation": ex.relation,
                "positive_text": ex.positive_text,
                "anchor_cui": ex.anchor_cui,
                "positive_cui": ex.positive_cui,
            }
            for ex in examples
        ],
    )


class TripletDataset(Dataset):
    def __init__(self, examples: list[RelationExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RelationExample:
        return self.examples[idx]


def collect_relation_terms(examples: list[RelationExample]) -> dict[str, set[str]]:
    relation_terms = defaultdict(set)
    for ex in examples:
        relation_terms[ex.relation].add(ex.anchor_text)
        relation_terms[ex.relation].add(ex.positive_text)
    return relation_terms
