from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from bloodfilm.errors import ConfigError
from bloodfilm.ml import require_torch

ImbalanceStrategy = Literal["none", "weighted_ce"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    imbalance: ImbalanceStrategy = "none"
    seed: int = 42


def compute_class_weights(labels: list[int], num_classes: int) -> Any:
    torch = require_torch("Head training")
    counts = [0] * num_classes
    for label in labels:
        if not 0 <= label < num_classes:
            raise ValueError(f"label {label} outside {num_classes} classes")
        counts[label] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError("Cannot weight an empty label list")
    weights = [total / (num_classes * count) if count else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def _loss_function(torch: Any, config: TrainConfig, labels: list[int], num_classes: int) -> Any:
    nn = torch.nn

    if config.imbalance == "weighted_ce":
        return nn.CrossEntropyLoss(weight=compute_class_weights(labels, num_classes))
    if config.imbalance == "none":
        return nn.CrossEntropyLoss()
    raise ConfigError(f"Unknown imbalance strategy {config.imbalance!r}")


def train_head(
    model: Any,
    embeddings: Any,
    labels: list[int],
    config: TrainConfig,
    num_classes: int,
) -> dict[str, list[float]]:
    torch = require_torch("Head training")
    generator = torch.Generator().manual_seed(config.seed)
    sampler = torch.randperm(len(labels), generator=generator).tolist()
    ordered_embeddings = embeddings[sampler]
    ordered_labels = torch.tensor([labels[index] for index in sampler])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = _loss_function(torch, config, labels, num_classes)
    history: dict[str, list[float]] = {"loss": []}
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(ordered_embeddings), ordered_labels)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
    return history


def head_accuracy(model: Any, embeddings: Any, labels: list[int]) -> float:
    if not labels:
        raise ValueError("Cannot score an empty label list")
    torch = require_torch("Head training")
    model.eval()
    with torch.inference_mode():
        predicted = model(embeddings).argmax(dim=1).tolist()
    hits = sum(1 for guess, truth in zip(predicted, labels) if guess == truth)
    return hits / len(labels)
