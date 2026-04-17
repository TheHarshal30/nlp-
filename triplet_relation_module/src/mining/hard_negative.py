import random
from collections import defaultdict

import numpy as np

from ..data.umls import RelationExample


class HardNegativeMiner:
    def __init__(
        self,
        examples: list[RelationExample],
        term_embeddings: np.ndarray,
        term_to_index: dict[str, int],
        strategy: str = "hard",
        pool_size: int = 32,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.pool_size = pool_size
        self.rng = random.Random(seed)
        self.term_embeddings = term_embeddings
        self.term_to_index = term_to_index
        self.related = defaultdict(lambda: defaultdict(set))
        self.relation_terms = defaultdict(set)
        for ex in examples:
            self.related[ex.relation][ex.anchor_text].add(ex.positive_text)
            self.relation_terms[ex.relation].add(ex.anchor_text)
            self.relation_terms[ex.relation].add(ex.positive_text)
        self.relation_terms = {key: sorted(values) for key, values in self.relation_terms.items()}

    def sample(self, anchor_text: str, relation: str, positive_text: str) -> str:
        candidates = [
            text
            for text in self.relation_terms[relation]
            if text != anchor_text and text != positive_text and text not in self.related[relation][anchor_text]
        ]
        if not candidates:
            global_candidates = [
                text
                for text in self.term_to_index
                if text != anchor_text and text != positive_text and text not in self.related[relation][anchor_text]
            ]
            if not global_candidates:
                return positive_text
            return self.rng.choice(global_candidates)

        if self.strategy != "hard":
            return self.rng.choice(candidates)

        anchor_index = self.term_to_index[anchor_text]
        anchor_vector = self.term_embeddings[anchor_index]
        candidate_indices = np.array([self.term_to_index[text] for text in candidates], dtype=np.int64)
        similarities = self.term_embeddings[candidate_indices] @ anchor_vector
        top_k = min(self.pool_size, len(candidates))
        hardest = np.argpartition(-similarities, top_k - 1)[:top_k]
        chosen = self.rng.choice(hardest.tolist())
        return candidates[chosen]
