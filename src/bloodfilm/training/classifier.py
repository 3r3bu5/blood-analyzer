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


def cache_split_tensors(cache: Any, split: str) -> tuple[Any, list[int]]:
    """Select the embedding rows and labels for one split, skipping failures."""
    ok_entries = [entry for entry in cache.entries if entry.status == "ok"]
    positions = [index for index, entry in enumerate(ok_entries) if entry.split == split]
    if not positions:
        raise ConfigError(f"Embedding cache has no usable rows for split {split!r}")
    return cache.embeddings[positions], [ok_entries[index].label_index for index in positions]


def evaluate_head(
    model: Any, embeddings: Any, labels: list[int], class_names: list[str]
) -> dict[str, Any]:
    """Evaluate a head: report, top-2 accuracy, ECE and raw logits for calibration."""
    from bloodfilm.evaluation.classifier import (
        classification_report,
        expected_calibration_error,
        top_k_accuracy,
    )

    torch = require_torch("Head evaluation")
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
    predictions = logits.argmax(dim=1).tolist()
    probabilities = torch.softmax(logits, dim=1).tolist()
    return {
        "logits": logits,
        "predictions": predictions,
        "report": classification_report(predictions, labels, class_names),
        "top2_accuracy": top_k_accuracy(logits, torch.tensor(labels), k=2),
        "ece": expected_calibration_error(probabilities, labels, bins=15),
    }


def run_head_training(
    cache: Any,
    head_name: str,
    num_classes: int,
    class_names: list[str],
    config: TrainConfig,
    eval_split: str = "validation",
) -> dict[str, Any]:
    """Train one head on cached train embeddings and evaluate once on eval_split."""
    from bloodfilm.classification.heads import build_head
    from bloodfilm.evaluation.classifier import expected_calibration_error, fit_temperature

    torch = require_torch("Head training")
    train_embeddings, train_labels = cache_split_tensors(cache, "train")
    eval_embeddings, eval_labels = cache_split_tensors(cache, eval_split)
    if max(train_labels + eval_labels) >= num_classes:
        raise ConfigError("Cache labels exceed num_classes")
    model = build_head(head_name, num_classes)
    history = train_head(model, train_embeddings, train_labels, config, num_classes)["loss"]
    evaluation = evaluate_head(model, eval_embeddings, eval_labels, class_names)
    temperature = fit_temperature(evaluation["logits"], torch.tensor(eval_labels))
    scaled = torch.softmax(evaluation["logits"] / temperature, dim=1).tolist()
    return {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "train_size": len(train_labels),
        "eval_split": eval_split,
        "eval_size": len(eval_labels),
        "eval_report": evaluation["report"],
        "eval_top2_accuracy": evaluation["top2_accuracy"],
        "eval_ece_before": evaluation["ece"],
        "temperature": temperature,
        "eval_ece_after": expected_calibration_error(scaled, eval_labels, bins=15),
        "final_loss": history[-1],
        "epochs": len(history),
    }
