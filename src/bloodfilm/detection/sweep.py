from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bloodfilm.detection.detector import Detection
from bloodfilm.detection.metrics import evaluate_detections, threshold_selection_report
from bloodfilm.documents import write_json
from bloodfilm.errors import ConfigError, InputNotFoundError, ModelLoadError
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, probe_image

DEFAULT_SWEEP_THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)


def parse_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated confidence-threshold list into sorted unique floats."""
    values: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise ConfigError(f"Invalid confidence threshold {text!r}") from exc
        if not 0.0 < value < 1.0:
            raise ConfigError(f"Confidence threshold {value} is outside (0, 1)")
        values.append(round(value, 4))
    unique = sorted(set(values))
    if not unique:
        raise ConfigError("No confidence thresholds supplied")
    return unique


def collect_images(input_path: Path | str) -> list[Path]:
    """Collect supported image files from a file or directory path."""
    path = Path(input_path)
    if not path.exists():
        raise InputNotFoundError(f"Sweep input not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ConfigError(f"Unsupported sweep image extension: {path.suffix}")
        return [path]
    images = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise InputNotFoundError(f"No supported images under {path}")
    return images


def load_yolo_truth(labels_dir: Path | str, images: list[Path]) -> dict[str, list[Detection]]:
    """Load YOLO txt ground-truth boxes as pixel-space WBC detections.

    Label files are expected beside ``labels_dir`` with the image stem and a
    ``.txt`` suffix. Class id 0 is treated as the WBC candidate class. Images
    without a label file contribute an empty truth list.
    """
    root = Path(labels_dir)
    truth: dict[str, list[Detection]] = {}
    for image in images:
        metadata = probe_image(image)
        label_path = root / f"{image.stem}.txt"
        detections: list[Detection] = []
        if label_path.exists():
            detections = _parse_yolo_label_file(
                label_path, image_width=metadata.width, image_height=metadata.height
            )
        truth[image.stem] = detections
    return truth


def score_thresholds_with_truth(
    predictions_by_image: dict[str, list[Detection]],
    truth_by_image: dict[str, list[Detection]],
    thresholds: list[float],
    *,
    iou_threshold: float,
    target_recall: float,
    max_false_positives_per_image: float,
) -> dict[str, Any]:
    """Score one prediction set at every threshold and select the best one."""
    candidates = [
        {
            "confidence_threshold": threshold,
            **_candidate_metrics(
                predictions_by_image,
                truth_by_image,
                confidence_threshold=threshold,
                iou_threshold=iou_threshold,
            ),
        }
        for threshold in thresholds
    ]
    return threshold_selection_report(
        candidates,
        target_recall=target_recall,
        max_false_positives_per_image=max_false_positives_per_image,
    )


def run_prediction_sweep(
    weights: Path | str,
    images: list[Path],
    thresholds: list[float],
    *,
    iou_threshold: float = 0.5,
    image_size: int = 640,
) -> dict[str, Any]:
    """Run a YOLO bundle's weights over images once per threshold.

    Returns per-threshold detection counts, boxes, and latency without
    requiring ground-truth labels. Pair with :func:`score_thresholds_with_truth`
    when labels are available.
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise InputNotFoundError(f"Detector weights not found: {weights_path}")
    if not images:
        raise ConfigError("No sweep images supplied")
    if not thresholds:
        raise ConfigError("No sweep thresholds supplied")
    model = _load_yolo(weights_path)
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        started = time.perf_counter()
        boxes_by_image: dict[str, list[dict[str, float]]] = {}
        total = 0
        for image in images:
            boxes = _predict_boxes(
                model, image, confidence=threshold, iou=iou_threshold, image_size=image_size
            )
            boxes_by_image[image.stem] = [
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "score": box.score,
                }
                for box in boxes
            ]
            total += len(boxes)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        results.append(
            {
                "confidence_threshold": threshold,
                "image_count": len(images),
                "detection_count": total,
                "detections_per_image": total / len(images),
                "latency_ms": elapsed_ms,
                "boxes_by_image": boxes_by_image,
            }
        )
    return {
        "status": "ok",
        "research_only": True,
        "weights": str(weights_path),
        "iou_threshold": iou_threshold,
        "image_size": image_size,
        "images": [str(image) for image in images],
        "candidates": results,
    }


def write_sweep_report(report: dict[str, Any], output: Path | str) -> Path:
    """Write a sweep report to JSON and return the output path."""
    output_path = Path(output)
    write_json(output_path, report)
    return output_path


def _candidate_metrics(
    predictions_by_image: dict[str, list[Detection]],
    truth_by_image: dict[str, list[Detection]],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, float]:
    report = evaluate_detections(
        predictions_by_image=predictions_by_image,
        truth_by_image=truth_by_image,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    return {
        "recall": report["recall"],
        "precision": report["precision"],
        "false_positives_per_image": report["false_positives_per_image"],
        "missed_wbc_rate": report["missed_wbc_rate"],
    }


def _parse_yolo_label_file(
    label_path: Path, *, image_width: int, image_height: int
) -> list[Detection]:
    detections: list[Detection] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ConfigError(f"Malformed YOLO label {label_path}:{line_number}")
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = (float(part) for part in parts[1:])
        except ValueError as exc:
            raise ConfigError(f"Malformed YOLO label {label_path}:{line_number}") from exc
        if class_id != 0:
            continue
        box_width = width * image_width
        box_height = height * image_height
        x1 = center_x * image_width - box_width / 2
        y1 = center_y * image_height - box_height / 2
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x1 + box_width,
                y2=y1 + box_height,
                score=1.0,
                label="wbc_candidate",
            )
        )
    return detections


def _load_yolo(weights_path: Path) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModelLoadError(
            "Detector sweep requires ultralytics; install the ml extras first"
        ) from exc
    try:
        return YOLO(str(weights_path))
    except Exception as exc:
        raise ModelLoadError(f"Detector weights are not loadable: {weights_path}") from exc


def _predict_boxes(
    model: Any, image: Path, *, confidence: float, iou: float, image_size: int
) -> list[Detection]:
    try:
        results = model.predict(
            str(image), conf=confidence, iou=iou, imgsz=image_size, verbose=False
        )
    except Exception as exc:
        raise ModelLoadError(f"Detector sweep failed on image {image}: {exc}") from exc
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []
    detections: list[Detection] = []
    for box in boxes:
        x1, y1, x2, y2 = _to_floats(box.xyxy[0], expected=4)
        score = _to_float(box.conf[0])
        detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label="wbc_candidate"))
    return detections


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return float(value.tolist())
    return float(value)


def _to_floats(value: Any, *, expected: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        items = [float(item) for item in value.tolist()]
    else:
        items = [float(item) for item in value]
    if len(items) != expected:
        raise ModelLoadError(f"Detector box has {len(items)} coordinates, expected {expected}")
    return items
