from __future__ import annotations

from typing import Any

from bloodfilm.detection.detector import Detection
from bloodfilm.errors import ConfigError


def evaluate_detections(
    *,
    predictions_by_image: dict[str, list[Detection]],
    truth_by_image: dict[str, list[Detection]],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
    target_label: str = "wbc_candidate",
) -> dict[str, Any]:
    predictions_by_image = _filter_label(predictions_by_image, target_label)
    truth_by_image = _filter_label(truth_by_image, target_label)
    image_ids = sorted(set(predictions_by_image) | set(truth_by_image))
    matches = _match_dataset(
        predictions_by_image,
        truth_by_image,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    truth_count = sum(len(truth_by_image.get(image_id, [])) for image_id in image_ids)
    predicted_count = matches["true_positives"] + matches["false_positives"]
    recall = matches["true_positives"] / truth_count if truth_count else 0.0
    precision = matches["true_positives"] / predicted_count if predicted_count else 0.0
    false_positives_per_image = matches["false_positives"] / len(image_ids) if image_ids else 0.0
    missed_wbc_rate = matches["false_negatives"] / truth_count if truth_count else 0.0
    ap_by_iou = {
        f"ap{round(threshold * 100)}": average_precision(
            predictions_by_image,
            truth_by_image,
            confidence_threshold=0.0,
            iou_threshold=threshold,
        )
        for threshold in _map_thresholds()
    }
    ap50 = ap_by_iou["ap50"]
    map50_95 = sum(ap_by_iou.values()) / len(ap_by_iou) if ap_by_iou else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_count": len(image_ids),
        "truth_count": truth_count,
        "predicted_count": predicted_count,
        "true_positives": matches["true_positives"],
        "false_positives": matches["false_positives"],
        "missed_wbc_count": matches["false_negatives"],
        "recall": recall,
        "precision": precision,
        "false_positives_per_image": false_positives_per_image,
        "missed_wbc_rate": missed_wbc_rate,
        "ap50": ap50,
        "mAP50": ap50,
        "mAP50_95": map50_95,
        "ap_by_iou": ap_by_iou,
        "missed_wbc": matches["missed_wbc"],
    }


def average_precision(
    predictions_by_image: dict[str, list[Detection]],
    truth_by_image: dict[str, list[Detection]],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> float:
    truth_count = sum(len(rows) for rows in truth_by_image.values())
    if truth_count == 0:
        return 0.0
    predictions = sorted(
        (
            (image_id, detection)
            for image_id, rows in predictions_by_image.items()
            for detection in rows
            if detection.score >= confidence_threshold
        ),
        key=lambda item: item[1].score,
        reverse=True,
    )
    matched: dict[str, set[int]] = {image_id: set() for image_id in truth_by_image}
    true_positives: list[int] = []
    false_positives: list[int] = []
    for image_id, prediction in predictions:
        match_index = _best_unmatched_truth(
            prediction,
            truth_by_image.get(image_id, []),
            matched.setdefault(image_id, set()),
            iou_threshold,
        )
        if match_index is None:
            true_positives.append(0)
            false_positives.append(1)
        else:
            matched[image_id].add(match_index)
            true_positives.append(1)
            false_positives.append(0)
    if not true_positives:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    precision_recall: list[tuple[float, float]] = []
    for tp, fp in zip(true_positives, false_positives, strict=True):
        cumulative_tp += tp
        cumulative_fp += fp
        recall = cumulative_tp / truth_count
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        precision_recall.append((recall, precision))
    return _interpolated_ap(precision_recall)


def select_confidence_threshold(
    candidates: list[dict[str, float]],
    *,
    target_recall: float,
    max_false_positives_per_image: float,
) -> dict[str, float | str]:
    if not candidates:
        raise ConfigError("No detector threshold candidates supplied")
    viable = [
        row
        for row in candidates
        if row["recall"] >= target_recall
        and row["false_positives_per_image"] <= max_false_positives_per_image
    ]
    if viable:
        selected = max(
            viable,
            key=lambda row: (
                row["recall"],
                -row["false_positives_per_image"],
                row["confidence_threshold"],
            ),
        )
        return {**selected, "selection_reason": "meets_recall_and_false_crop_budget"}
    selected = max(candidates, key=lambda row: (row["recall"], -row["false_positives_per_image"]))
    return {**selected, "selection_reason": "best_available_recall"}


def _match_dataset(
    predictions_by_image: dict[str, list[Detection]],
    truth_by_image: dict[str, list[Detection]],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    missed_wbc: list[dict[str, object]] = []
    for image_id in sorted(set(predictions_by_image) | set(truth_by_image)):
        truths = truth_by_image.get(image_id, [])
        matched: set[int] = set()
        predictions = sorted(
            [
                row
                for row in predictions_by_image.get(image_id, [])
                if row.score >= confidence_threshold
            ],
            key=lambda row: row.score,
            reverse=True,
        )
        for prediction in predictions:
            match_index = _best_unmatched_truth(prediction, truths, matched, iou_threshold)
            if match_index is None:
                false_positives += 1
            else:
                matched.add(match_index)
                true_positives += 1
        for index, truth in enumerate(truths):
            if index not in matched:
                false_negatives += 1
                missed_wbc.append({"image_id": image_id, "box": truth.to_dict()})
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "missed_wbc": missed_wbc,
    }


def _best_unmatched_truth(
    prediction: Detection,
    truths: list[Detection],
    matched: set[int],
    iou_threshold: float,
) -> int | None:
    best_index = None
    best_iou = 0.0
    for index, truth in enumerate(truths):
        if index in matched:
            continue
        overlap = iou(prediction, truth)
        if overlap > best_iou:
            best_iou = overlap
            best_index = index
    return best_index if best_iou >= iou_threshold else None


def iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _interpolated_ap(precision_recall: list[tuple[float, float]]) -> float:
    total = 0.0
    for threshold in [index / 10 for index in range(11)]:
        precisions = [precision for recall, precision in precision_recall if recall >= threshold]
        total += max(precisions, default=0.0)
    return total / 11


def _map_thresholds() -> list[float]:
    return [threshold / 100 for threshold in range(50, 100, 5)]


def _filter_label(
    detections_by_image: dict[str, list[Detection]], target_label: str
) -> dict[str, list[Detection]]:
    return {
        image_id: [detection for detection in detections if detection.label == target_label]
        for image_id, detections in detections_by_image.items()
    }
