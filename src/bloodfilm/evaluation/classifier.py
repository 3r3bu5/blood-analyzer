from __future__ import annotations

import math
from typing import Any

from bloodfilm.classification.uncertainty import UncertaintyPolicy, decide_prediction


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
        "macro_precision": sum(precisions) / num_classes if num_classes else 0.0,
        "macro_recall": sum(recalls) / num_classes if num_classes else 0.0,
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


def expected_calibration_error(
    probabilities: list[list[float]], labels: list[int], bins: int
) -> float:
    """Binned expected calibration error over the predicted max probability."""
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities ({len(probabilities)}) and labels ({len(labels)}) must match"
        )
    if bins < 1:
        raise ValueError(f"bins must be positive, got {bins}")
    if not labels:
        return 0.0
    confidences = [max(row) for row in probabilities]
    predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    error = 0.0
    for low, high in _bin_edges(bins):
        members = [
            index
            for index, confidence in enumerate(confidences)
            if (low < confidence <= high) or (low == 0.0 and confidence == 0.0)
        ]
        if not members:
            continue
        accuracy = sum(1 for index in members if predictions[index] == labels[index]) / len(members)
        confidence = sum(confidences[index] for index in members) / len(members)
        error += len(members) / len(labels) * abs(accuracy - confidence)
    return error


def _bin_edges(bins: int) -> list[tuple[float, float]]:
    width = 1.0 / bins
    return [(index * width, (index + 1) * width) for index in range(bins)]


def fit_temperature(logits: Any, labels: Any) -> float:
    """Fit Guo et al. temperature scaling on a held-out logit set."""
    from bloodfilm.ml import require_torch

    torch = require_torch("Temperature scaling")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def _nll() -> Any:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(_nll)
    temperature = float(log_temperature.exp().detach())
    if not temperature > 0.0:
        raise ValueError(f"Temperature fitting diverged: {temperature}")
    return temperature


def top_k_accuracy(logits: Any, labels: Any, k: int) -> float:
    """Fraction of samples whose label is among the top-k logits."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if len(labels) == 0:
        return 0.0
    top = logits.topk(min(k, logits.shape[1]), dim=1).indices
    hits = sum(1 for row, truth in zip(top.tolist(), labels.tolist()) if int(truth) in row)
    return hits / len(labels)


def detailed_per_class_metrics(
    probabilities: list[list[float]],
    labels: list[int],
    class_names: list[str],
    *,
    high_confidence_threshold: float = 0.90,
) -> list[dict[str, object]]:
    predicted = predictions_from_probabilities(probabilities)
    matrix = confusion_matrix(predicted, labels, len(class_names))
    rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        support = sum(matrix[class_id])
        predicted_as = sum(row[class_id] for row in matrix)
        true_positives = matrix[class_id][class_id]
        false_positives = predicted_as - true_positives
        false_negatives = support - true_positives
        precision = true_positives / predicted_as if predicted_as else 0.0
        recall = true_positives / support if support else 0.0
        incorrect_confusions = [
            (other_id, count)
            for other_id, count in enumerate(matrix[class_id])
            if other_id != class_id and count > 0
        ]
        most_confused = max(incorrect_confusions, key=lambda item: item[1], default=None)
        correct_conf = [
            probabilities[index][class_id]
            for index, truth in enumerate(labels)
            if truth == class_id and predicted[index] == class_id
        ]
        incorrect_conf = [
            max(probabilities[index])
            for index, truth in enumerate(labels)
            if truth == class_id and predicted[index] != class_id
        ]
        high_conf_errors = sum(
            1
            for index, truth in enumerate(labels)
            if truth == class_id
            and predicted[index] != class_id
            and max(probabilities[index]) >= high_confidence_threshold
        )
        rows.append(
            {
                "class_id": class_id,
                "class_code": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "most_frequent_confusion_class": class_names[most_confused[0]]
                if most_confused
                else None,
                "most_frequent_confusion_count": most_confused[1] if most_confused else 0,
                "mean_calibrated_confidence_correct": _mean(correct_conf),
                "mean_calibrated_confidence_incorrect": _mean(incorrect_conf),
                "high_confidence_error_count": high_conf_errors,
                "limitation": "low_support" if support < 30 else "",
            }
        )
    return rows


def calibration_summary(
    probabilities: list[list[float]], labels: list[int], *, bins: int = 15
) -> dict[str, object]:
    _check_probability_inputs(probabilities, labels)
    reliability_bins = reliability_bin_rows(probabilities, labels, bins=bins)
    ece = sum(row["weight"] * row["absolute_gap"] for row in reliability_bins)
    mce = max((row["absolute_gap"] for row in reliability_bins), default=0.0)
    nll = -sum(math.log(max(probabilities[i][label], 1e-12)) for i, label in enumerate(labels))
    nll = nll / len(labels) if labels else 0.0
    brier = 0.0
    if labels:
        for row, truth in zip(probabilities, labels):
            brier += sum(
                (prob - (1.0 if index == truth else 0.0)) ** 2 for index, prob in enumerate(row)
            )
        brier /= len(labels)
    return {
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "negative_log_likelihood": nll,
        "brier_score": brier,
        "reliability_bins": reliability_bins,
    }


def reliability_bin_rows(
    probabilities: list[list[float]], labels: list[int], *, bins: int = 15
) -> list[dict[str, float | int]]:
    _check_probability_inputs(probabilities, labels)
    confidences = [max(row) for row in probabilities]
    predicted = predictions_from_probabilities(probabilities)
    rows: list[dict[str, float | int]] = []
    for index, (low, high) in enumerate(_bin_edges(bins)):
        members = [
            item
            for item, confidence in enumerate(confidences)
            if (low < confidence <= high) or (low == 0.0 and confidence == 0.0)
        ]
        count = len(members)
        accuracy = (
            sum(1 for item in members if predicted[item] == labels[item]) / count if count else 0.0
        )
        confidence = sum(confidences[item] for item in members) / count if count else 0.0
        rows.append(
            {
                "bin_index": index,
                "low": low,
                "high": high,
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "absolute_gap": abs(accuracy - confidence),
                "weight": count / len(labels) if labels else 0.0,
            }
        )
    return rows


def coverage_accuracy_rows(
    probabilities: list[list[float]],
    labels: list[int],
    class_names: list[str],
    policy: UncertaintyPolicy,
) -> list[dict[str, object]]:
    predicted = predictions_from_probabilities(probabilities)
    decisions = [decide_prediction(row, class_names, policy) for row in probabilities]
    rows: list[dict[str, object]] = []
    total = len(labels)
    for status in ("accepted", "review_required", "unknown"):
        indices = [index for index, decision in enumerate(decisions) if decision.status == status]
        correct = sum(1 for index in indices if predicted[index] == labels[index])
        errors = sum(1 for index in indices if predicted[index] != labels[index])
        total_errors = sum(1 for guess, truth in zip(predicted, labels) if guess != truth)
        rows.append(
            {
                "status": status,
                "count": len(indices),
                "coverage": len(indices) / total if total else 0.0,
                "accuracy": correct / len(indices) if indices else 0.0,
                "error_count": errors,
                "error_coverage": errors / total_errors if total_errors else 0.0,
            }
        )
    return rows


def predictions_from_probabilities(probabilities: list[list[float]]) -> list[int]:
    """Argmax predictions shared by evaluation and reporting seams."""
    return [max(range(len(row)), key=row.__getitem__) for row in probabilities]


def _predictions(probabilities: list[list[float]]) -> list[int]:
    return predictions_from_probabilities(probabilities)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _check_probability_inputs(probabilities: list[list[float]], labels: list[int]) -> None:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must match")
    if any(not row for row in probabilities):
        raise ValueError("probability rows must not be empty")
    for row, label in zip(probabilities, labels):
        if not 0 <= label < len(row):
            raise ValueError(f"label index {label} outside probability row width {len(row)}")
