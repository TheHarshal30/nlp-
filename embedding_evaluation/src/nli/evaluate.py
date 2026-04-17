import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.metrics import macro_f1_score
from .data import load_nli_rows


class NLIClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _features(embedder, rows: list[dict], batch_size: int) -> tuple[np.ndarray, list[str]]:
    premise = embedder.encode([row["premise"] for row in rows], batch_size=batch_size)
    hypothesis = embedder.encode([row["hypothesis"] for row in rows], batch_size=batch_size)
    features = np.concatenate([premise, hypothesis, np.abs(premise - hypothesis)], axis=1).astype(np.float32)
    labels = [row["label"] for row in rows]
    return features, labels


def evaluate_nli(embedder, train_path: str, dev_path: str, test_path: str, batch_size: int, epochs: int, lr: float, hidden_dim: int, seed: int, smoke_limit: int | None = None) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_rows = load_nli_rows(train_path)
    dev_rows = load_nli_rows(dev_path)
    test_rows = load_nli_rows(test_path)
    if smoke_limit:
        train_rows = train_rows[:smoke_limit]
        dev_rows = dev_rows[:smoke_limit]
        test_rows = test_rows[:smoke_limit]
    if not train_rows or not test_rows:
        return {"accuracy": 0.0, "f1": 0.0, "examples": 0}

    x_train, y_train_text = _features(embedder, train_rows, batch_size)
    x_dev, y_dev_text = _features(embedder, dev_rows, batch_size)
    x_test, y_test_text = _features(embedder, test_rows, batch_size)
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(y_train_text) | set(y_dev_text) | set(y_test_text)))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    y_train = np.array([label_to_id[label] for label in y_train_text], dtype=np.int64)
    y_dev = np.array([label_to_id[label] for label in y_dev_text], dtype=np.int64)
    y_test = np.array([label_to_id[label] for label in y_test_text], dtype=np.int64)

    model = NLIClassifier(input_dim=x_train.shape[1], hidden_dim=hidden_dim, num_labels=len(label_to_id))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )

    best_state = None
    best_dev = -1.0
    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            optimiser.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
        if len(x_dev):
            model.eval()
            with torch.inference_mode():
                dev_pred = model(torch.from_numpy(x_dev)).argmax(dim=1).numpy()
            dev_acc = float((dev_pred == y_dev).mean())
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(x_test)).argmax(dim=1).numpy()
    predicted_labels = [id_to_label[int(idx)] for idx in predictions]
    gold_labels = [id_to_label[int(idx)] for idx in y_test]
    return {
        "accuracy": float((predictions == y_test).mean()),
        "f1": macro_f1_score(gold_labels, predicted_labels),
        "examples": len(y_test),
    }
