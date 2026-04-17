import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


class Vocabulary:
    def __init__(self, payload: dict):
        tokens = payload["special_tokens"] + payload["tokens"]
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = tokens
        self.pad_id = self.token_to_id["[PAD]"]
        self.unk_id = self.token_to_id["[UNK]"]
        self.cls_id = self.token_to_id["[CLS]"]

    @classmethod
    def from_path(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def encode(self, tokens: list[str], max_length: int, add_cls: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_cls:
            ids = [self.cls_id] + ids
        return ids[:max_length]

    def known_token_count(self, tokens: list[str]) -> int:
        return sum(token in self.token_to_id and token not in {"[PAD]", "[UNK]", "[CLS]"} for token in tokens)


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, ffn_dim: int, dropout: float, max_length: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        return self.encoder(hidden, src_key_padding_mask=(attention_mask == 0))


class BaseTextEncoder(nn.Module):
    model_type: str

    def has_coverage(self, text: str) -> bool:
        raise NotImplementedError

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class Word2VecTextEncoder(BaseTextEncoder):
    model_type = "word2vec"

    def __init__(self, model_dir: str | Path, freeze: bool):
        super().__init__()
        vectors_path = Path(model_dir) / "weights" / "vectors.bin"
        kv = KeyedVectors.load_word2vec_format(str(vectors_path), binary=True)
        self.tokens = list(kv.key_to_index.keys())
        self.word_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        weights = torch.tensor(kv.vectors, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)

    @property
    def output_dim(self) -> int:
        return self.embedding.embedding_dim

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def has_coverage(self, text: str) -> bool:
        return any(token in self.word_to_id for token in self.tokenize(text))

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        outputs = []
        device = self.embedding.weight.device
        for text in texts:
            ids = [self.word_to_id[token] for token in self.tokenize(text) if token in self.word_to_id]
            if not ids:
                outputs.append(torch.zeros(self.output_dim, device=device))
            else:
                tensor = torch.tensor(ids, dtype=torch.long, device=device)
                outputs.append(self.embedding(tensor).mean(dim=0))
        return torch.stack(outputs)


class TransformerTextEncoder(BaseTextEncoder):
    model_type = "transformer"

    def __init__(self, model_dir: str | Path, freeze: bool):
        super().__init__()
        model_dir = Path(model_dir)
        with (model_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.vocab = Vocabulary.from_path(model_dir / "weights" / "vocab.json")
        cfg = self.metadata["model_config"]
        self.pooling = self.metadata.get("pooling", "cls")
        self.max_length = cfg["max_length"]
        self.backbone = TransformerBackbone(
            vocab_size=len(self.vocab.id_to_token),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            dropout=cfg["dropout"],
            max_length=cfg["max_length"],
        )
        state = torch.load(model_dir / "weights" / "transformer.pt", map_location="cpu")
        self.backbone.load_state_dict(state, strict=False)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.backbone.token_embeddings.embedding_dim

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def has_coverage(self, text: str) -> bool:
        tokens = self.tokenize(text)
        return any(self.vocab.token_to_id.get(token, self.vocab.unk_id) != self.vocab.unk_id for token in tokens)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        sequences = [self.vocab.encode(self.tokenize(text), max_length=self.max_length, add_cls=True) for text in texts]
        max_len = max(len(seq) for seq in sequences)
        ids = []
        masks = []
        for seq in sequences:
            pad = max_len - len(seq)
            ids.append(seq + [self.vocab.pad_id] * pad)
            masks.append([1] * len(seq) + [0] * pad)
        device = next(self.backbone.parameters()).device
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        hidden = self.backbone(input_ids, attention_mask)
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return summed / denom


def load_base_encoder(base_model: str, model_dir: str | Path, freeze: bool) -> BaseTextEncoder:
    if base_model == "word2vec":
        return Word2VecTextEncoder(model_dir, freeze=freeze)
    if base_model == "transformer":
        return TransformerTextEncoder(model_dir, freeze=freeze)
    raise ValueError(f"Unsupported base_model: {base_model}")


def encode_texts_in_batches(encoder: BaseTextEncoder, texts: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    encoder = encoder.to(device)
    encoder.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            outputs.append(encoder.encode_texts(chunk).detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, encoder.output_dim), dtype=torch.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)
