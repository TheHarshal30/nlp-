import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .backbones import TransformerEncoderModel, Vocabulary


log = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)
    logits = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(batch, 2 * batch, device=z.device), torch.arange(0, batch, device=z.device)])
    return F.cross_entropy(logits, targets)


class UMLSPairDataset(Dataset):
    def __init__(self, pairs_path: str):
        self.pairs = []
        with Path(pairs_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    self.pairs.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


class BackboneAdapter(nn.Module):
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError


class Word2VecBackboneAdapter(BackboneAdapter):
    def __init__(self, vectors_path: str, freeze: bool):
        super().__init__()
        kv = KeyedVectors.load_word2vec_format(vectors_path, binary=True)
        self.tokens = list(kv.key_to_index.keys())
        self.word_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(kv.vectors, dtype=torch.float32), freeze=freeze)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        outputs = []
        device = self.embedding.weight.device
        for text in texts:
            ids = [self.word_to_id[token] for token in text.lower().split() if token in self.word_to_id]
            if not ids:
                outputs.append(torch.zeros(self.embedding_dim, device=device))
            else:
                tensor = torch.tensor(ids, dtype=torch.long, device=device)
                outputs.append(self.embedding(tensor).mean(dim=0))
        return torch.stack(outputs)


class TransformerBackboneAdapter(BackboneAdapter):
    def __init__(self, model_dir: str, freeze: bool):
        super().__init__()
        metadata = json_load(Path(model_dir) / "metadata.json")
        vocab = Vocabulary.from_json(str(Path(model_dir) / "weights" / "vocab.json"))
        cfg = metadata["model_config"]
        self.pooling = metadata.get("pooling", "cls")
        self.vocab = vocab
        self.model = TransformerEncoderModel(
            vocab_size=len(vocab.id_to_token),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            dropout=cfg["dropout"],
            max_length=cfg["max_length"],
        )
        state = torch.load(Path(model_dir) / "weights" / "transformer.pt", map_location="cpu")
        self.model.load_state_dict(state)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.model.token_embeddings.embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        sequences = [self.vocab.encode_tokens(text.lower().split(), max_length=self.model.position_embeddings.num_embeddings - 1, add_cls=True) for text in texts]
        max_len = max(len(seq) for seq in sequences)
        pad_id = self.vocab.pad_id
        input_ids = []
        masks = []
        for seq in sequences:
            pad = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad)
            masks.append([1] * len(seq) + [0] * pad)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        masks = torch.tensor(masks, dtype=torch.long, device=device)
        return self.model.encode(input_ids, masks, pooling=self.pooling)


def json_load(path: Path) -> dict:
    import json
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class AlignmentArtifacts:
    adapter: BackboneAdapter
    head: ProjectionHead


def load_alignment_components(base_model_type: str, base_model_dir: str, projection_dim: int, freeze_base: bool) -> AlignmentArtifacts:
    if base_model_type == "word2vec":
        adapter = Word2VecBackboneAdapter(str(Path(base_model_dir) / "weights" / "vectors.bin"), freeze=freeze_base)
    elif base_model_type == "transformer":
        adapter = TransformerBackboneAdapter(base_model_dir, freeze=freeze_base)
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")
    head = ProjectionHead(adapter.embedding_dim, projection_dim)
    return AlignmentArtifacts(adapter=adapter, head=head)


def train_alignment(config: dict, context, export_fn) -> None:
    align_cfg = config["alignment"]
    trainer_cfg = config["trainer"]
    artifacts = load_alignment_components(
        base_model_type=align_cfg["base_model_type"],
        base_model_dir=align_cfg["base_model_dir"],
        projection_dim=align_cfg["projection_dim"],
        freeze_base=align_cfg["freeze_base"],
    )
    dataset = UMLSPairDataset(config["data"]["pairs_txt"])
    loader = DataLoader(dataset, batch_size=trainer_cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts.adapter = artifacts.adapter.to(device)
    artifacts.head = artifacts.head.to(device)

    params = list(artifacts.head.parameters()) + [p for p in artifacts.adapter.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=trainer_cfg["lr"])

    start_epoch = 0
    if trainer_cfg.get("resume"):
        checkpoint = context.load_checkpoint()
        if checkpoint is not None:
            artifacts.head.load_state_dict(checkpoint["head"])
            artifacts.adapter.load_state_dict(checkpoint["adapter"], strict=False)
            optimiser.load_state_dict(checkpoint["optim"])
            start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, trainer_cfg["epochs"]):
        total_loss = 0.0
        for anchors, positives in tqdm(loader, desc=f"alignment epoch {epoch+1}", unit="batch"):
            optimiser.zero_grad()
            z_a = artifacts.head(artifacts.adapter.encode_texts(list(anchors)))
            z_b = artifacts.head(artifacts.adapter.encode_texts(list(positives)))
            loss = nt_xent(z_a, z_b, temperature=align_cfg["temperature"])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())

        context.save_checkpoint(
            {
                "epoch": epoch,
                "head": artifacts.head.state_dict(),
                "adapter": artifacts.adapter.state_dict(),
                "optim": optimiser.state_dict(),
            }
        )
        log.info("epoch=%s loss=%.4f", epoch + 1, total_loss / max(len(loader), 1))

    export_fn(config=config, context=context, artifacts=artifacts)

