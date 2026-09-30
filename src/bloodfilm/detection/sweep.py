from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from bloodfilm.classification.png import write_rgb_png
from bloodfilm.detection.detector import Detection
from bloodfilm.detection.metrics import evaluate_detections, threshold_selection_report
from bloodfilm.detection.roi import FieldROI, detect_field_roi
from bloodfilm.detection.tiling import (
    detection_center_in_tile_core,
    generate_tiles,
    global_deduplicate,
    map_tile_detection,
)
from bloodfilm.documents import write_json
from bloodfilm.errors import ConfigError, InputNotFoundError, ModelLoadError
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, ImageData, load_image, probe_image

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
    preprocessing_mode: str = "full_field",
    tile_size: int = 512,
    overlap_ratio: float = 0.20,
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
    if preprocessing_mode not in {"full_field", "field_roi", "tiled", "field_roi_tiled"}:
        raise ConfigError(f"Unsupported preprocessing mode: {preprocessing_mode}")
    if tile_size <= 0:
        raise ConfigError("tile_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ConfigError("overlap_ratio must be in [0, 1)")
    model = _load_yolo(weights_path)
    roi_by_image = _roi_by_image(images, preprocessing_mode=preprocessing_mode)
    use_tiling = preprocessing_mode in {"tiled", "field_roi_tiled"}
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        started = time.perf_counter()
        boxes_by_image: dict[str, list[dict[str, float]]] = {}
        box_quality_flags_by_image: dict[str, list[list[str]]] = {}
        counts_by_image: dict[str, dict[str, int | bool]] = {}
        total = 0
        with tempfile.TemporaryDirectory(prefix="bloodfilm-detector-sweep-") as temp_dir:
            temp_root = Path(temp_dir)
            for image in images:
                prediction_image = image
                roi = roi_by_image.get(image.stem)
                if roi is not None:
                    prediction_image = _write_roi_crop(image, roi, temp_root / f"{image.stem}.png")
                tile_flags_by_box: dict[tuple[float, float, float, float], list[str]] = {}
                if use_tiling:
                    boxes, stage_counts, tile_flags_by_box = _predict_tiled_boxes(
                        model,
                        prediction_image,
                        temp_root / image.stem,
                        confidence=threshold,
                        iou=iou_threshold,
                        image_size=image_size,
                        tile_size=tile_size,
                        overlap_ratio=overlap_ratio,
                    )
                else:
                    boxes = _predict_boxes(
                        model,
                        prediction_image,
                        confidence=threshold,
                        iou=iou_threshold,
                        image_size=image_size,
                    )
                    stage_counts = _single_image_stage_counts(len(boxes))
                if roi is not None:
                    tile_flags_by_box = _map_detection_flags_to_roi(tile_flags_by_box, roi)
                    boxes = [_map_roi_detection(box, roi) for box in boxes]
                quality_flags_by_image = _quality_flags_by_detection(
                    boxes,
                    image_width=probe_image(image).width,
                    image_height=probe_image(image).height,
                    inherited_flags=tile_flags_by_box,
                )
                boxes = _mark_review_boxes(boxes, quality_flags_by_image)
                boxes_by_image[image.stem] = [_box_dict(box) for box in boxes]
                box_quality_flags_by_image[image.stem] = quality_flags_by_image
                counts_by_image[image.stem] = {
                    **stage_counts,
                    "sent_to_classifier_count": len(boxes),
                    "box_review_count": sum(1 for flags in quality_flags_by_image if flags),
                }
                total += len(boxes)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        results.append(
            {
                "confidence_threshold": threshold,
                "image_count": len(images),
                "detection_count": total,
                "detections_per_image": total / len(images),
                "latency_ms": elapsed_ms,
                "counts_by_image": counts_by_image,
                "box_quality_flags_by_image": box_quality_flags_by_image,
                "boxes_by_image": boxes_by_image,
            }
        )
    return {
        "status": "ok",
        "research_only": True,
        "weights": str(weights_path),
        "iou_threshold": iou_threshold,
        "image_size": image_size,
        "preprocessing_mode": preprocessing_mode,
        "tile_size": tile_size if use_tiling else None,
        "overlap_ratio": overlap_ratio if use_tiling else None,
        "maximum_detection_limit": None,
        "images": [str(image) for image in images],
        "roi_by_image": {image_id: roi.to_dict() for image_id, roi in roi_by_image.items()},
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


def _roi_by_image(images: list[Path], *, preprocessing_mode: str) -> dict[str, FieldROI]:
    if preprocessing_mode not in {"field_roi", "field_roi_tiled"}:
        return {}
    return {image.stem: detect_field_roi(image) for image in images}


def _write_roi_crop(image_path: Path, roi: FieldROI, output_path: Path) -> Path:
    image = load_image(image_path)
    pixels: list[tuple[int, int, int]] = []
    for y in range(roi.y1, roi.y2):
        for x in range(roi.x1, roi.x2):
            offset = (y * image.width + x) * image.channels
            channels = image.pixels[offset : offset + image.channels]
            if len(channels) == 1:
                pixels.append((channels[0], channels[0], channels[0]))
            else:
                pixels.append((channels[0], channels[1], channels[2]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb_png(output_path, roi.x2 - roi.x1, roi.y2 - roi.y1, pixels)
    return output_path


def _map_roi_detection(detection: Detection, roi: FieldROI) -> Detection:
    return Detection(
        x1=detection.x1 + roi.x1,
        y1=detection.y1 + roi.y1,
        x2=detection.x2 + roi.x1,
        y2=detection.y2 + roi.y1,
        score=detection.score,
        label=detection.label,
        box_status=detection.box_status,
        box_review_reasons=detection.box_review_reasons,
    )


def _mark_review_boxes(
    detections: list[Detection], quality_flags_by_detection: list[list[str]]
) -> list[Detection]:
    marked: list[Detection] = []
    for detection, flags in zip(detections, quality_flags_by_detection):
        if not flags:
            marked.append(detection)
            continue
        reasons = tuple(dict.fromkeys((*detection.box_review_reasons, "localization_quality")))
        marked.append(
            Detection(
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                score=detection.score,
                label=detection.label,
                box_status="review",
                box_review_reasons=reasons,
            )
        )
    return marked


def _single_image_stage_counts(count: int) -> dict[str, int | bool]:
    return {
        "raw_yolo_proposals_count": count,
        "post_confidence_count": count,
        "post_tile_merge_count": count,
        "post_global_nms_count": count,
        "post_box_sanity_count": count,
        "maximum_detection_limit_applied": False,
    }


def _predict_tiled_boxes(
    model: Any,
    image: Path,
    output_root: Path,
    *,
    confidence: float,
    iou: float,
    image_size: int,
    tile_size: int,
    overlap_ratio: float,
) -> tuple[
    list[Detection],
    dict[str, int | bool],
    dict[tuple[float, float, float, float], list[str]],
]:
    image_data = load_image(image)
    tiles = generate_tiles(
        image_data.width,
        image_data.height,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )
    merged: list[Detection] = []
    tile_flags_by_box: dict[tuple[float, float, float, float], list[str]] = {}
    raw_count = 0
    output_root.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        tile_path = output_root / f"{tile.tile_id}.png"
        _write_tile_crop(image_data, tile.x1, tile.y1, tile.x2, tile.y2, tile_path)
        tile_boxes = _predict_boxes(
            model,
            tile_path,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
        )
        raw_count += len(tile_boxes)
        for box in tile_boxes:
            mapped = map_tile_detection(
                box, tile, image_width=image_data.width, image_height=image_data.height
            )
            if detection_center_in_tile_core(
                mapped,
                tile,
                tiles,
                image_width=image_data.width,
                image_height=image_data.height,
            ):
                merged.append(mapped)
                flags = _tile_boundary_flags(
                    box, tile, image_width=image_data.width, image_height=image_data.height
                )
                if flags:
                    tile_flags_by_box[_detection_key(mapped)] = flags
    kept = global_deduplicate(merged, iou_threshold=iou)
    kept_flags = {
        _detection_key(detection): tile_flags_by_box[_detection_key(detection)]
        for detection in kept
        if _detection_key(detection) in tile_flags_by_box
    }
    return (
        kept,
        {
            "tile_count": len(tiles),
            "raw_yolo_proposals_count": raw_count,
            "post_confidence_count": raw_count,
            "post_tile_core_count": len(merged),
            "post_tile_merge_count": len(merged),
            "post_global_nms_count": len(kept),
            "post_global_dedup_count": len(kept),
            "post_box_sanity_count": len(kept),
            "maximum_detection_limit_applied": False,
        },
        kept_flags,
    )


def _write_tile_crop(
    image: ImageData, x1: int, y1: int, x2: int, y2: int, output_path: Path
) -> None:
    pixels: list[tuple[int, int, int]] = []
    for y in range(y1, y2):
        for x in range(x1, x2):
            offset = (y * image.width + x) * image.channels
            channels = image.pixels[offset : offset + image.channels]
            if len(channels) == 1:
                pixels.append((channels[0], channels[0], channels[0]))
            else:
                pixels.append((channels[0], channels[1], channels[2]))
    write_rgb_png(output_path, x2 - x1, y2 - y1, pixels)


def _quality_flags_by_detection(
    detections: list[Detection],
    *,
    image_width: int,
    image_height: int,
    inherited_flags: dict[tuple[float, float, float, float], list[str]],
) -> list[list[str]]:
    if not detections:
        return []
    areas = sorted(_area(detection) for detection in detections if _area(detection) > 0)
    median_area = areas[len(areas) // 2] if areas else 0.0
    flags_by_index: list[list[str]] = []
    for index, detection in enumerate(detections):
        flags = list(inherited_flags.get(_detection_key(detection), []))
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        area = width * height if width > 0 and height > 0 else 0.0
        if median_area > 0 and area > median_area * 2.5:
            flags.append("large_area_vs_field_median")
        if width > image_width * 0.20 or height > image_height * 0.20:
            flags.append("large_axis_vs_field")
        if width > 0 and height > 0 and max(width / height, height / width) > 3.0:
            flags.append("extreme_aspect_ratio")
        if _contains_other_detection(detection, detections, current_index=index):
            flags.append("contains_smaller_candidate")
        flags_by_index.append(sorted(set(flags)))
    return flags_by_index


def _tile_boundary_flags(
    detection: Detection, tile: Any, *, image_width: int, image_height: int
) -> list[str]:
    flags: list[str] = []
    tolerance = 2.0
    tile_width = float(tile.x2 - tile.x1)
    tile_height = float(tile.y2 - tile.y1)
    touches_left = detection.x1 <= tolerance and tile.x1 > 0
    touches_top = detection.y1 <= tolerance and tile.y1 > 0
    touches_right = detection.x2 >= tile_width - tolerance and tile.x2 < image_width
    touches_bottom = detection.y2 >= tile_height - tolerance and tile.y2 < image_height
    if touches_left or touches_top or touches_right or touches_bottom:
        flags.append("internal_tile_boundary_truncation")
    return flags


def _contains_other_detection(
    detection: Detection, detections: list[Detection], *, current_index: int
) -> bool:
    detection_area = _area(detection)
    if detection_area <= 0:
        return False
    for index, other in enumerate(detections):
        if index == current_index:
            continue
        other_area = _area(other)
        if other_area <= 0 or other_area >= detection_area:
            continue
        intersection = _intersection_area(detection, other)
        if intersection / other_area >= 0.85:
            return True
    return False


def _intersection_area(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def _area(detection: Detection) -> float:
    return max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)


def _detection_key(detection: Detection) -> tuple[float, float, float, float]:
    return (
        round(detection.x1, 3),
        round(detection.y1, 3),
        round(detection.x2, 3),
        round(detection.y2, 3),
    )


def _map_detection_flags_to_roi(
    flags_by_box: dict[tuple[float, float, float, float], list[str]], roi: FieldROI
) -> dict[tuple[float, float, float, float], list[str]]:
    mapped: dict[tuple[float, float, float, float], list[str]] = {}
    for key, flags in flags_by_box.items():
        x1, y1, x2, y2 = key
        detection = Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0)
        mapped[_detection_key(_map_roi_detection(detection, roi))] = flags
    return mapped


def _box_dict(box: Detection) -> dict[str, Any]:
    data: dict[str, Any] = {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
        "score": box.score,
    }
    if box.box_status != "accepted" or box.box_review_reasons:
        data["box_status"] = box.box_status
        data["box_review_reasons"] = list(box.box_review_reasons)
    return data


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
