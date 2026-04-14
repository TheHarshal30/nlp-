import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecUMLSEmbedder(BaseEmbedder):
    """
    UMLS-grounded Word2Vec embedder.

    Built in two stages:
      1. Trained as a standard Skip-gram Word2Vec on PubMed abstracts.
      2. Fine-tuned with NT-Xent contrastive loss using UMLS synonym pairs.

    Inference is identical to the baseline: mean-pool over in-vocabulary tokens.
    """

    def __init__(self):
        self.wv = None
        self.umls_vocab = None
        self._name = "word2vec_umls"

    def load(self, model_path: str) -> None:
        """
        Load aligned word vectors and UMLS vocab from the weights folder.
        """
        weights_dir = os.path.join(model_path, "weights")
        bin_path = os.path.join(weights_dir, "word2vec_umls.bin")
        vocab_path = os.path.join(weights_dir, "umls_vocab.json")

        for path in (bin_path, vocab_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Run the training pipeline (01 -> 02 -> 03) first."
                )

        print(f"[word2vec_umls] loading aligned vectors from {bin_path} ...")
        self.wv = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        print(f"[word2vec_umls] ready — vocab: {len(self.wv):,}  dim: {self.wv.vector_size}")

        print(f"[word2vec_umls] loading UMLS vocab from {vocab_path} ...")
        with open(vocab_path, "r", encoding="utf-8") as handle:
            self.umls_vocab = json.load(handle)
        print(f"[word2vec_umls] UMLS vocab loaded — {len(self.umls_vocab):,} CUIs")

    def _embed_one(self, text: str) -> np.ndarray:
        """
        Mean-pool aligned word vectors for a single text string.
        Returns a zero vector for fully OOV input.
        """
        tokens = text.lower().split()
        vectors = [self.wv[t] for t in tokens if t in self.wv]

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert N text strings to a float32 numpy array of shape (N, vector_size).
        """
        if self.wv is None:
            raise RuntimeError("call load() before encode()")

        embeddings = [self._embed_one(t) for t in texts]
        return np.vstack(embeddings).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
