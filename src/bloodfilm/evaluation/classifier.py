from __future__ import annotations

from typing import Any


def confusion_matrix(predicted: list[int], labels: list[int], num_classes: int) -> list[list[int]]:
    _check_lengths(predicted, labels)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for guess, truth in zip(predicted, labels):
        _check_class_index(guess, num_classes, "predicted")
        _check_class_index(truth, num_classes, "labels")
        matrix[truth][guess] += 1
    return matrix


def classification_report(
    predicted: list[int], labels: list[int], class_names: list[str]
) -> dict[str, Any]:
    _check_lengths(predicted, labels)
    num_classes = len(class_names)
    matrix = confusion_matrix(predicted, labels, num_classes)
    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for class_id, name in enumerate(class_names):
        support = sum(matrix[class_id])
        predicted_as = sum(row[class_id] for row in matrix)
        true_positives = matrix[class_id][class_id]
        precision = true_positives / predicted_as if predicted_as else 0.0
        recall = true_positives / support if support else 0.0
        f1 = _f1(precision, recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        per_class.append(
            {
                "class_id": class_id,
                "class_code": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    total = sum(supports)
    return {
        "accuracy": sum(1 for guess, truth in zip(predicted, labels) if guess == truth)
        / len(labels)
        if labels
        else 0.0,
        "balanced_accuracy": sum(recalls) / num_classes if num_classes else 0.0,
        "macro_f1": sum(f1s) / num_classes if num_classes else 0.0,
        "weighted_f1": sum(score * count for score, count in zip(f1s, supports)) / total
        if total
        else 0.0,
        "per_class": per_class,
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _check_lengths(predicted: list[int], labels: list[int]) -> None:
    if len(predicted) != len(labels):
        raise ValueError(f"predicted ({len(predicted)}) and labels ({len(labels)}) must match")


def _check_class_index(value: int, num_classes: int, field: str) -> None:
    if not 0 <= value < num_classes:
        raise ValueError(f"{field} index {value} outside {num_classes} classes")
