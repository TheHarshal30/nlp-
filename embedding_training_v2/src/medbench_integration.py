import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORD2VEC_TEMPLATE = """import json
import os
import sys
import numpy as np
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))
from base_embedder import BaseEmbedder


class {class_name}(BaseEmbedder):
    def __init__(self):
        self.wv = None
        self.metadata = None
        self._name = '{model_name}'

    def load(self, model_path):
        weights_path = os.path.join(model_path, 'weights', 'vectors.bin')
        meta_path = os.path.join(model_path, 'metadata.json')
        self.wv = KeyedVectors.load_word2vec_format(weights_path, binary=True)
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)

    def encode(self, texts, batch_size=32):
        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            token_lists = [[tok for tok in text.lower().split() if tok in self.wv] for text in chunk]
            for toks in token_lists:
                if not toks:
                    outputs.append(np.zeros(self.wv.vector_size, dtype=np.float32))
                else:
                    outputs.append(np.mean([self.wv[tok] for tok in toks], axis=0).astype(np.float32))
        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self):
        return self._name
"""


TRANSFORMER_TEMPLATE = """import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))
from base_embedder import BaseEmbedder


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ffn_dim, dropout, max_length):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        return self.encoder(hidden, src_key_padding_mask=(attention_mask == 0))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class {class_name}(BaseEmbedder):
    def __init__(self):
        self.model = None
        self.projection = None
        self.vocab = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._name = '{model_name}'

    def load(self, model_path):
        meta_path = os.path.join(model_path, 'metadata.json')
        vocab_path = os.path.join(model_path, 'weights', 'vocab.json')
        weights_path = os.path.join(model_path, 'weights', 'transformer.pt')
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)
        with open(vocab_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        tokens = payload['special_tokens'] + payload['tokens']
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        cfg = self.metadata['model_config']
        self.model = TransformerEncoderModel(
            vocab_size=len(tokens),
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            ffn_dim=cfg['ffn_dim'],
            dropout=cfg['dropout'],
            max_length=cfg['max_length'],
        ).to(self.device)
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        proj_path = os.path.join(model_path, 'weights', 'projection.pt')
        if os.path.exists(proj_path):
            self.projection = ProjectionHead(cfg['hidden_size'], self.metadata['projection_dim']).to(self.device)
            self.projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
            self.projection.eval()

    def encode(self, texts, batch_size=32):
        pad_id = self.vocab['[PAD]']
        cls_id = self.vocab['[CLS]']
        unk_id = self.vocab['[UNK]']
        max_length = self.metadata['model_config']['max_length']
        pooling = self.metadata.get('pooling', 'cls')
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                ids = []
                masks = []
                for text in chunk:
                    seq = [cls_id] + [self.vocab.get(tok, unk_id) for tok in text.lower().split()]
                    seq = seq[:max_length]
                    ids.append(seq)
                max_len = max(len(seq) for seq in ids)
                for i, seq in enumerate(ids):
                    pad = max_len - len(seq)
                    masks.append([1] * len(seq) + [0] * pad)
                    ids[i] = seq + [pad_id] * pad
                input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
                hidden = self.model(input_ids, attention_mask)
                if pooling == 'cls':
                    pooled = hidden[:, 0]
                else:
                    mask = attention_mask.unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                if self.projection is not None and self.metadata.get('use_projection_at_inference', False):
                    pooled = self.projection(pooled)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self):
        return self._name
"""


TEST_TEXTS = [
    "diabetes mellitus",
    "insulin therapy",
    "heart failure",
]


class MedBenchIntegrationError(RuntimeError):
    pass


@dataclass
class ModelSpec:
    name: str
    candidates: list[Path]
    class_name: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def model_specs(root: Path) -> list[ModelSpec]:
    return [
        ModelSpec("word2vec", [root / "embedding_training_v2" / "outputs" / "models" / "word2vec"], "Word2VecEmbedder"),
        ModelSpec("word2vec_umls", [root / "embedding_training_v2" / "outputs" / "models" / "word2vec_umls"], "Word2VecUMLSEmbedder"),
        ModelSpec(
            "transformer",
            [
                root / "embedding_training_v2" / "outputs" / "models" / "transformer",
                root / "embedding_training_v2" / "outputs" / "models" / "transformer_fast",
            ],
            "TransformerEmbedder",
        ),
        ModelSpec(
            "transformer_umls",
            [
                root / "embedding_training_v2" / "outputs" / "models" / "transformer_umls",
                root / "embedding_training_v2" / "outputs" / "models" / "transformer_umls_fast",
            ],
            "TransformerUMLSEmbedder",
        ),
    ]


def resolve_source_dir(spec: ModelSpec) -> Path:
    for candidate in spec.candidates:
        if candidate.exists():
            return candidate
    raise MedBenchIntegrationError(
        f"missing source export for {spec.name}; checked: {', '.join(str(path) for path in spec.candidates)}"
    )


def required_files_for(metadata: dict) -> list[str]:
    model_type = metadata["model_type"]
    if model_type.startswith("word2vec"):
        return ["metadata.json", "weights/vectors.bin"]
    if model_type == "transformer":
        return ["metadata.json", "weights/transformer.pt", "weights/vocab.json"]
    if model_type == "transformer_umls":
        return ["metadata.json", "weights/transformer.pt", "weights/vocab.json", "weights/projection.pt"]
    raise MedBenchIntegrationError(f"unsupported model_type: {model_type}")


def wrapper_template_for(metadata: dict) -> str:
    return WORD2VEC_TEMPLATE if metadata["model_type"].startswith("word2vec") else TRANSFORMER_TEMPLATE


def copy_model_to_medbench(source_dir: Path, destination_dir: Path, canonical_name: str, class_name: str) -> dict:
    meta_path = source_dir / "metadata.json"
    if not meta_path.exists():
        raise MedBenchIntegrationError(f"missing metadata.json in {source_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for relative in required_files_for(metadata):
        if not (source_dir / relative).exists():
            raise MedBenchIntegrationError(f"missing required file for {canonical_name}: {source_dir / relative}")

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, destination_dir / "metadata.json")

    weights_src = source_dir / "weights"
    weights_dest = destination_dir / "weights"
    shutil.copytree(weights_src, weights_dest)

    snapshot = source_dir / "config_snapshot.json"
    if snapshot.exists():
        shutil.copy2(snapshot, destination_dir / "config_snapshot.json")

    checkpoints = source_dir / "checkpoints"
    if checkpoints.exists():
        shutil.copytree(checkpoints, destination_dir / "checkpoints")

    template = wrapper_template_for(metadata)
    (destination_dir / "model.py").write_text(
        template.format(class_name=class_name, model_name=canonical_name),
        encoding="utf-8",
    )
    return metadata


def prepare_models(medbench_root: Path, selected_models: list[str]) -> dict[str, Path]:
    root = repo_root()
    specs = {spec.name: spec for spec in model_specs(root)}
    models_dir = medbench_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}

    for name in selected_models:
        spec = specs[name]
        source_dir = resolve_source_dir(spec)
        destination_dir = models_dir / name
        copy_model_to_medbench(source_dir, destination_dir, name, spec.class_name)
        prepared[name] = destination_dir

    return prepared


def install_base_embedder_stub() -> None:
    if "base_embedder" in sys.modules:
        return
    module = types.ModuleType("base_embedder")

    class BaseEmbedder:
        def load(self, model_path):
            raise NotImplementedError

        def encode(self, texts, batch_size=32):
            raise NotImplementedError

        @property
        def name(self):
            return self.__class__.__name__

    module.BaseEmbedder = BaseEmbedder
    sys.modules["base_embedder"] = module


def load_embedder(model_dir: Path):
    model_file = model_dir / "model.py"
    if not model_file.exists():
        raise MedBenchIntegrationError(f"missing model.py in {model_dir}")
    install_base_embedder_stub()
    spec = importlib.util.spec_from_file_location(f"medbench_model_{model_dir.name}", model_file)
    if spec is None or spec.loader is None:
        raise MedBenchIntegrationError(f"unable to import {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and hasattr(obj, "load") and hasattr(obj, "encode"):
            embedder = obj()
            embedder.load(str(model_dir))
            return embedder
    raise MedBenchIntegrationError(f"no embedder class found in {model_file}")


def validate_preflight(prepared_models: dict[str, Path]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    outputs_by_model: dict[str, np.ndarray] = {}

    for name, model_dir in prepared_models.items():
        embedder = load_embedder(model_dir)
        outputs = embedder.encode(TEST_TEXTS, batch_size=3)
        if not isinstance(outputs, np.ndarray):
            raise MedBenchIntegrationError(f"{name}: encode() returned {type(outputs).__name__}, expected numpy.ndarray")
        if outputs.shape[0] != len(TEST_TEXTS) or outputs.ndim != 2:
            raise MedBenchIntegrationError(f"{name}: expected shape (N, D), got {outputs.shape}")
        if np.isnan(outputs).any() or np.isinf(outputs).any():
            raise MedBenchIntegrationError(f"{name}: output contains NaN or Inf values")
        repeated = embedder.encode(TEST_TEXTS, batch_size=3)
        if not np.allclose(outputs, repeated, atol=1e-6, rtol=1e-5):
            raise MedBenchIntegrationError(f"{name}: repeated encode() calls are not stable")
        results[name] = {
            "shape": list(outputs.shape),
            "dtype": str(outputs.dtype),
        }
        outputs_by_model[name] = outputs

    names = list(outputs_by_model)
    for idx, left in enumerate(names):
        for right in names[idx + 1:]:
            left_output = outputs_by_model[left]
            right_output = outputs_by_model[right]
            if left_output.shape == right_output.shape and np.allclose(left_output, right_output, atol=1e-8, rtol=1e-8):
                raise MedBenchIntegrationError(f"preflight outputs for {left} and {right} are unexpectedly identical")

    return results


def run_benchmark(medbench_root: Path, model_name: str, python_bin: str) -> None:
    evaluation_dir = medbench_root / "evaluation"
    run_all = evaluation_dir / "run_all.py"
    if not run_all.exists():
        raise MedBenchIntegrationError(f"missing benchmark entrypoint: {run_all}")
    subprocess.run(
        [python_bin, "run_all.py", "--model", model_name],
        cwd=evaluation_dir,
        check=True,
    )


def flatten_numeric(payload, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(value, next_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}[{index}]"
            flattened.update(flatten_numeric(value, next_prefix))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flattened[prefix] = float(payload)
    return flattened


def collect_json_metrics(result_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not result_dir.exists():
        return metrics
    for json_file in sorted(result_dir.rglob("*.json")):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        flattened = flatten_numeric(payload)
        for key, value in flattened.items():
            metrics[f"{json_file.relative_to(result_dir)}::{key}"] = value
    return metrics


def pick_metric(metrics: dict[str, float], include_terms: list[str], priority_terms: list[str]) -> str:
    candidates = []
    for key, value in metrics.items():
        lowered = key.lower()
        if not all(term in lowered for term in include_terms):
            continue
        score = sum(term in lowered for term in priority_terms)
        candidates.append((score, key, value))
    if not candidates:
        return "n/a"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, key, value = candidates[0]
    return f"{key}={value:.4f}"


def build_summary_row(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "Entity Linking": " | ".join(
            [
                pick_metric(metrics, ["link"], ["acc@1", "acc1", "top1"]),
                pick_metric(metrics, ["link"], ["acc@5", "acc5", "top5"]),
                pick_metric(metrics, ["link"], ["mrr"]),
            ]
        ),
        "STS": pick_metric(metrics, ["sts"], ["pearson"]),
        "NLI": " | ".join(
            [
                pick_metric(metrics, ["nli"], ["accuracy", "acc"]),
                pick_metric(metrics, ["nli"], ["macro", "f1"]),
            ]
        ),
    }


def aggregate_results(medbench_root: Path, selected_models: list[str]) -> dict[str, dict[str, object]]:
    results_root = medbench_root / "results"
    aggregate: dict[str, dict[str, object]] = {}
    unique_metric_sets = set()

    for model_name in selected_models:
        metrics = collect_json_metrics(results_root / model_name)
        aggregate[model_name] = {
            "metrics": metrics,
            "summary": build_summary_row(metrics),
        }
        unique_metric_sets.add(tuple(sorted(metrics.items())))

    if len(selected_models) > 1 and len(unique_metric_sets) == 1:
        raise MedBenchIntegrationError("all benchmark metrics are identical across models; results look degenerate")

    summary_rows = {
        model_name: payload["summary"]
        for model_name, payload in aggregate.items()
    }
    (results_root / "embedding_benchmark_summary.json").write_text(
        json.dumps({"models": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = ["| Model | Entity Linking | STS | NLI |", "| --- | --- | --- | --- |"]
    for model_name, row in summary_rows.items():
        lines.append(f"| {model_name} | {row['Entity Linking']} | {row['STS']} | {row['NLI']} |")
    (results_root / "embedding_benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
