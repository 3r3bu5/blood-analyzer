from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from bloodfilm.errors import BloodFilmError, ConfigError, InputNotFoundError
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, probe_image


class AnnotationCompleteness(StrEnum):
    FULLY_ANNOTATED = "fully_annotated"
    SPARSELY_ANNOTATED = "sparsely_annotated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class AnnotationRecord:
    box: Box
    source_class: str
    canonical_class: str
    source_dataset: str = "unknown"
    attributes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def require_fully_annotated(completeness: AnnotationCompleteness | str) -> None:
    value = AnnotationCompleteness(completeness)
    if value != AnnotationCompleteness.FULLY_ANNOTATED:
        raise ConfigError("Detector training preparation requires fully annotated samples")


def parse_yolo_annotations(
    label_path: Path | str,
    *,
    image_width: int,
    image_height: int,
    class_names: dict[int, str],
    class_mapping: dict[str, str],
    source_dataset: str = "LeukemiaAttri",
) -> list[AnnotationRecord]:
    path = Path(label_path)
    if not path.exists():
        raise InputNotFoundError(f"YOLO label file not found: {path}")
    records: list[AnnotationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ConfigError(f"Malformed YOLO annotation {path.name}:{line_number}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError as exc:
            raise ConfigError(f"Malformed YOLO annotation {path.name}:{line_number}") from exc
        _validate_normalized_geometry(path, line_number, x_center, y_center, width, height)
        source_class = class_names.get(class_id, f"class_{class_id}")
        canonical_class, warnings = _map_source_class(source_class, class_mapping)
        box_width = width * image_width
        box_height = height * image_height
        x1 = x_center * image_width - box_width / 2
        y1 = y_center * image_height - box_height / 2
        attributes = parts[5:]
        metadata = {
            "source_class": source_class,
            "source_dataset": source_dataset,
            "line_number": line_number,
        }
        if attributes:
            metadata["attributes"] = attributes
        records.append(
            AnnotationRecord(
                box=Box(x1=x1, y1=y1, x2=x1 + box_width, y2=y1 + box_height),
                source_class=source_class,
                canonical_class=canonical_class,
                source_dataset=source_dataset,
                attributes=attributes,
                metadata=metadata,
                warnings=warnings,
            )
        )
    return records


def parse_coco_annotations(
    annotation_path: Path | str,
    *,
    class_mapping: dict[str, str],
    source_dataset: str = "LeukemiaAttri",
) -> dict[str, list[AnnotationRecord]]:
    path = Path(annotation_path)
    if not path.exists():
        raise InputNotFoundError(f"COCO annotation file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in raw.get("images", [])}
    categories = {int(row["id"]): str(row["name"]) for row in raw.get("categories", [])}
    records: dict[str, list[AnnotationRecord]] = {
        str(row["file_name"]): [] for row in raw.get("images", [])
    }
    for annotation in raw.get("annotations", []):
        image = images.get(int(annotation.get("image_id")))
        if image is None:
            raise ConfigError(
                f"COCO annotation references unknown image_id {annotation.get('image_id')}"
            )
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ConfigError(f"COCO annotation {annotation.get('id')} has malformed bbox")
        x, y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            raise ConfigError(f"COCO annotation {annotation.get('id')} has non-positive bbox")
        source_class = categories.get(int(annotation.get("category_id")), "unknown")
        canonical_class, warnings = _map_source_class(source_class, class_mapping)
        metadata = {
            "source_class": source_class,
            "source_dataset": source_dataset,
            "annotation_id": annotation.get("id"),
            "attributes": annotation.get("attributes", {}),
        }
        records[str(image["file_name"])].append(
            AnnotationRecord(
                box=Box(x1=x, y1=y, x2=x + width, y2=y + height),
                source_class=source_class,
                canonical_class=canonical_class,
                source_dataset=source_dataset,
                metadata=metadata,
                warnings=warnings,
            )
        )
    return records


def audit_yolo_directory(
    *,
    image_root: Path | str,
    label_root: Path | str,
    class_names: dict[int, str],
    class_mapping: dict[str, str],
    annotation_completeness: str,
    field_sized_area_ratio: float = 0.35,
) -> dict[str, Any]:
    images = _collect_images(Path(image_root))
    labels = Path(label_root)
    completeness = AnnotationCompleteness(annotation_completeness)
    source_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    malformed: list[str] = []
    unreadable: list[str] = []
    missing_labels: list[str] = []
    orphan_labels = [str(path) for path in sorted(labels.rglob("*.txt"))]
    field_sized_boxes = 0
    total_boxes = 0
    for image in images:
        try:
            metadata = probe_image(image)
        except BloodFilmError as exc:
            unreadable.append(f"{image}: {exc}")
            continue
        label_path = labels / f"{image.stem}.txt"
        if label_path.exists():
            if str(label_path) in orphan_labels:
                orphan_labels.remove(str(label_path))
        else:
            missing_labels.append(str(image))
            continue
        try:
            records = parse_yolo_annotations(
                label_path,
                image_width=metadata.width,
                image_height=metadata.height,
                class_names=class_names,
                class_mapping=class_mapping,
            )
        except ConfigError as exc:
            malformed.append(str(exc))
            continue
        image_area = metadata.width * metadata.height
        for record in records:
            total_boxes += 1
            source_counts[record.source_class] += 1
            canonical_counts[record.canonical_class] += 1
            if image_area and record.box.area / image_area >= field_sized_area_ratio:
                field_sized_boxes += 1
    status = "ok"
    if malformed or unreadable or missing_labels or orphan_labels or field_sized_boxes:
        status = "review_required"
    return {
        "schema_version": 1,
        "status": status,
        "annotation_completeness": completeness.value,
        "image_count": len(images),
        "box_count": total_boxes,
        "source_class_counts": dict(sorted(source_counts.items())),
        "canonical_class_counts": dict(sorted(canonical_counts.items())),
        "malformed_annotations": malformed,
        "unreadable_images": unreadable,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "field_sized_boxes": field_sized_boxes,
        "visual_review_required": True,
    }


def _collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise InputNotFoundError(f"Image root not found: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _validate_normalized_geometry(
    path: Path,
    line_number: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> None:
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise ConfigError(f"YOLO center outside [0, 1] in {path.name}:{line_number}")
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        raise ConfigError(f"YOLO size outside (0, 1] in {path.name}:{line_number}")


def _map_source_class(source_class: str, class_mapping: dict[str, str]) -> tuple[str, list[str]]:
    canonical_class = class_mapping.get(source_class)
    if canonical_class is None:
        return "unknown", [f"unmapped_source_class:{source_class}"]
    return canonical_class, []
