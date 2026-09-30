from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, TypedDict

from bloodfilm.classification.png import write_rgb_png
from bloodfilm.detection import CandidateCellDetector, Detection, validate_detections
from bloodfilm.errors import ConfigError
from bloodfilm.imaging.io import ImageData, load_image


class CropClassifier(TypedDict, total=False):
    decision: dict[str, Any]
    probabilities: list[dict[str, Any]]


class CropClassifierProtocol(Protocol):
    version: str

    def classify_crop(self, image_path: Path) -> CropClassifier: ...


def analyze_field_image(
    image_path: Path | str,
    *,
    detector: CandidateCellDetector,
    classifier: CropClassifierProtocol,
    crop_dir: Path | str,
    crop_padding_ratio: float = 0.10,
) -> dict[str, Any]:
    started = time.perf_counter()
    image = load_image(image_path)
    detections = validate_detections(
        detector.detect(image), image_width=image.width, image_height=image.height
    )
    crop_root = Path(crop_dir)
    crop_root.mkdir(parents=True, exist_ok=True)
    cells = []
    source_stem = Path(image_path).stem
    for index, detection in enumerate(detections):
        cell_id = f"{source_stem}_d{index:04d}"
        crop_box = _expanded_box(detection, image.width, image.height, crop_padding_ratio)
        crop_path = crop_root / f"{cell_id}.png"
        _write_crop(image, crop_box, crop_path)
        prediction = classifier.classify_crop(crop_path)
        probabilities = prediction.get("probabilities", [])
        if len(probabilities) != 18:
            raise ConfigError("Classifier field analysis output must contain 18 probabilities")
        decision = prediction.get("decision", {})
        box_status = detection.box_status
        box_review_reasons = list(detection.box_review_reasons)
        classifier_status = decision.get("status", "unknown")
        classification_status = "review" if box_status == "review" else classifier_status
        uncertainty_reasons = list(decision.get("uncertainty_reasons", []))
        if box_status == "review" and "localization_quality" not in uncertainty_reasons:
            uncertainty_reasons.append("localization_quality")
        cells.append(
            {
                "cell_id": cell_id,
                "box": _box_dict(detection),
                "box_status": box_status,
                "box_review_reasons": box_review_reasons,
                "crop_box": crop_box,
                "detector_confidence": detection.score,
                "predicted_class": decision.get("label"),
                "classification_confidence": decision.get("confidence"),
                "classification_status": classification_status,
                "decision_status": classification_status,
                "uncertainty_reasons": uncertainty_reasons,
                "probabilities": probabilities,
                "crop_path": str(crop_path),
                "detector_version": detector.version,
                "classifier_version": classifier.version,
            }
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "schema_version": 1,
        "status": "zero_detections" if not cells else "succeeded",
        "source_image": str(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "detector_version": detector.version,
        "classifier_version": classifier.version,
        "latency_ms": latency_ms,
        "missed_cell_rate": None,
        "cell_count": len(cells),
        "cells": cells,
    }


def _expanded_box(
    detection: Detection, image_width: int, image_height: int, padding_ratio: float
) -> dict[str, int]:
    width = detection.x2 - detection.x1
    height = detection.y2 - detection.y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return {
        "x1": max(0, int(detection.x1 - pad_x)),
        "y1": max(0, int(detection.y1 - pad_y)),
        "x2": min(image_width, int(detection.x2 + pad_x)),
        "y2": min(image_height, int(detection.y2 + pad_y)),
    }


def _write_crop(image: ImageData, box: dict[str, int], output: Path) -> None:
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    pixels: list[tuple[int, int, int]] = []
    for y in range(box["y1"], box["y2"]):
        for x in range(box["x1"], box["x2"]):
            offset = (y * image.width + x) * 3
            red, green, blue = image.pixels[offset : offset + 3]
            pixels.append((red, green, blue))
    write_rgb_png(output, width, height, pixels)


def _box_dict(detection: Detection) -> dict[str, int]:
    return {
        "x1": int(detection.x1),
        "y1": int(detection.y1),
        "x2": int(detection.x2),
        "y2": int(detection.y2),
    }
