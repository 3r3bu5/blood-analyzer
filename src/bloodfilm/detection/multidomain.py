from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from bloodfilm.detection.annotations import (
    AnnotationCompleteness,
    AnnotationRecord,
    Box,
    parse_yolo_annotations,
    require_fully_annotated,
)
from bloodfilm.documents import write_json
from bloodfilm.errors import ConfigError, InputNotFoundError
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, probe_image
from bloodfilm.util import sha256_file

MANIFEST_COLUMNS = [
    "sample_id",
    "image_path",
    "label_path",
    "image_sha256",
    "source_dataset",
    "source_split",
    "unified_split",
    "patient_id",
    "slide_id",
    "acquisition_domain",
    "microscope",
    "camera",
    "magnification",
    "annotation_completeness",
    "source_classes",
    "canonical_classes",
    "locked_for_validation",
    "license_id",
]


def prepare_multidomain_yolo_dataset(
    *,
    txl_pbc_root: Path | str,
    leukemia_attri_root: Path | str,
    output_root: Path | str,
    manifest_output: Path | str,
    splits_output: Path | str,
    leakage_output: Path | str,
    report_output: Path | str,
    txl_class_names: dict[int, str],
    txl_class_mapping: dict[str, str],
    leukemia_class_names: dict[int, str],
    leukemia_class_mapping: dict[str, str],
    locked_hashes: set[str],
    txl_annotation_completeness: str = "fully_annotated",
    leukemia_annotation_completeness: str = "fully_annotated",
    leukemia_annotation_format: str = "yolo",
) -> dict[str, Any]:
    """Create a unified candidate-WBC YOLO dataset from TXL-PBC and LeukemiaAttri."""
    require_fully_annotated(txl_annotation_completeness)
    require_fully_annotated(leukemia_annotation_completeness)
    output = Path(output_root)
    samples = [
        *_collect_source_samples(
            Path(txl_pbc_root),
            source_dataset="TXL-PBC",
            output_prefix="txl_pbc",
            class_names=txl_class_names,
            class_mapping=txl_class_mapping,
            annotation_completeness=AnnotationCompleteness(txl_annotation_completeness),
        ),
        *_collect_leukemia_samples(
            Path(leukemia_attri_root),
            class_names=leukemia_class_names,
            class_mapping=leukemia_class_mapping,
            annotation_completeness=AnnotationCompleteness(leukemia_annotation_completeness),
            annotation_format=leukemia_annotation_format,
        ),
    ]
    if not samples:
        raise ConfigError("No detector samples found for multidomain preparation")

    leakage_violations = [sample for sample in samples if sample["image_sha256"] in locked_hashes]
    leakage_report = {
        "schema_version": 1,
        "locked_target_violations": [sample["sample_id"] for sample in leakage_violations],
        "locked_hash_count": len(locked_hashes),
    }
    write_json(leakage_output, leakage_report)
    if leakage_violations:
        raise ConfigError("locked target image would leak into detector training data")

    _reset_yolo_tree(output)
    manifest_rows = []
    split_rows = []
    source_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    for sample in sorted(samples, key=lambda row: row["sample_id"]):
        split = str(sample["unified_split"])
        target_image = output / "images" / split / f"{sample['sample_id']}{sample['image_suffix']}"
        target_label = output / "labels" / split / f"{sample['sample_id']}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(str(sample["image_path"])), target_image)
        target_label.write_text(str(sample["label_text"]), encoding="utf-8")
        manifest_row = {key: str(sample.get(key, "")) for key in MANIFEST_COLUMNS}
        manifest_row["image_path"] = str(target_image)
        manifest_row["label_path"] = str(target_label)
        manifest_rows.append(manifest_row)
        split_rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "source_dataset": str(sample["source_dataset"]),
                "split": split,
            }
        )
        source_counts[str(sample["source_dataset"])] += 1
        box_counts[str(sample["source_dataset"])] += int(sample["candidate_wbc_count"])

    _write_csv(Path(manifest_output), MANIFEST_COLUMNS, manifest_rows)
    _write_csv(Path(splits_output), ["sample_id", "source_dataset", "split"], split_rows)
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {"0": "candidate_wbc"},
    }
    write_json(output / "data.yaml", data_yaml)
    report = {
        "schema_version": 1,
        "status": "ok",
        "research_only": True,
        "output_root": str(output),
        "data_yaml": str(output / "data.yaml"),
        "manifest": str(manifest_output),
        "splits": str(splits_output),
        "leakage_report": str(leakage_output),
        "sample_count": len(manifest_rows),
        "source_sample_counts": dict(sorted(source_counts.items())),
        "candidate_wbc_box_counts": dict(sorted(box_counts.items())),
        "notes": [
            "Only canonical candidate_wbc boxes are written as YOLO class 0.",
            "Artifact, RBC, platelet, unknown, and unmapped source classes are excluded from detector targets.",
            "Patient-level independence is not claimed unless source metadata supplies patient IDs.",
        ],
    }
    write_json(report_output, report)
    return report


def _collect_leukemia_samples(
    root: Path,
    *,
    class_names: dict[int, str],
    class_mapping: dict[str, str],
    annotation_completeness: AnnotationCompleteness,
    annotation_format: str,
) -> list[dict[str, Any]]:
    if annotation_format == "yolo":
        return _collect_source_samples(
            root,
            source_dataset="LeukemiaAttri",
            output_prefix="leukemia_attri",
            class_names=class_names,
            class_mapping=class_mapping,
            annotation_completeness=annotation_completeness,
        )
    if annotation_format == "coco_domain":
        return _collect_leukemia_coco_domain_samples(
            root,
            class_mapping=class_mapping,
            annotation_completeness=annotation_completeness,
        )
    raise ConfigError(f"Unsupported LeukemiaAttri annotation format: {annotation_format}")


def _collect_source_samples(
    root: Path,
    *,
    source_dataset: str,
    output_prefix: str,
    class_names: dict[int, str],
    class_mapping: dict[str, str],
    annotation_completeness: AnnotationCompleteness,
) -> list[dict[str, Any]]:
    image_root = root / "images"
    label_root = root / "labels"
    if not image_root.exists() or not label_root.exists():
        raise InputNotFoundError(f"YOLO images/labels not found under {root}")
    samples: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for image in sorted(_iter_images(image_root)):
        relative = image.relative_to(image_root)
        label = label_root / relative.with_suffix(".txt")
        if not label.exists():
            missing_labels.append(str(relative))
            continue
        metadata = probe_image(image)
        records = parse_yolo_annotations(
            label,
            image_width=metadata.width,
            image_height=metadata.height,
            class_names=class_names,
            class_mapping=class_mapping,
            source_dataset=source_dataset,
        )
        candidate_records = [
            record for record in records if record.canonical_class == "candidate_wbc"
        ]
        split = _split_from_relative(relative)
        sample_id = f"{output_prefix}__{_sample_key(relative)}"
        source_classes = sorted({record.source_class for record in records})
        canonical_classes = sorted({record.canonical_class for record in candidate_records})
        samples.append(
            {
                "sample_id": sample_id,
                "image_path": str(image),
                "label_path": str(label),
                "image_suffix": image.suffix,
                "image_sha256": sha256_file(image),
                "source_dataset": source_dataset,
                "source_split": split,
                "unified_split": split,
                "patient_id": "",
                "slide_id": "",
                "acquisition_domain": source_dataset,
                "microscope": "",
                "camera": "",
                "magnification": "",
                "annotation_completeness": annotation_completeness.value,
                "source_classes": ";".join(source_classes),
                "canonical_classes": ";".join(canonical_classes),
                "locked_for_validation": "false",
                "license_id": "verify_before_use"
                if source_dataset == "LeukemiaAttri"
                else "source_license",
                "label_text": "".join(
                    _record_to_yolo_line(
                        record, image_width=metadata.width, image_height=metadata.height
                    )
                    for record in candidate_records
                ),
                "candidate_wbc_count": len(candidate_records),
            }
        )
    if missing_labels:
        raise ConfigError(
            f"{source_dataset} has images without labels; fully annotated YOLO preparation requires labels for every image: "
            + ", ".join(missing_labels[:10])
        )
    return samples


def _collect_leukemia_coco_domain_samples(
    root: Path,
    *,
    class_mapping: dict[str, str],
    annotation_completeness: AnnotationCompleteness,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    annotation_paths = sorted(root.glob("*/json_labels/*.json"))
    if not annotation_paths:
        raise InputNotFoundError(f"LeukemiaAttri COCO json_labels not found under {root}")
    for annotation_path in annotation_paths:
        domain = annotation_path.parents[1].name
        source_split = annotation_path.stem
        unified_split = (
            "val" if source_split == "test" else _split_from_relative(Path(source_split))
        )
        image_root = root / domain / "Images" / source_split
        if not image_root.exists():
            image_root = root / domain / "images" / source_split
        if not image_root.exists():
            raise InputNotFoundError(
                f"LeukemiaAttri images not found for {annotation_path}: expected Images/{source_split}"
            )
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_rows = {int(row["id"]): row for row in raw.get("images", [])}
        categories = {int(row["id"]): str(row["name"]) for row in raw.get("categories", [])}
        unmapped_categories = sorted(set(categories.values()) - set(class_mapping))
        if unmapped_categories:
            raise ConfigError(
                f"LeukemiaAttri COCO file has unmapped COCO categories in {annotation_path}: "
                + ", ".join(unmapped_categories)
            )
        records_by_image: dict[int, list[AnnotationRecord]] = {
            image_id: [] for image_id in image_rows
        }
        for annotation in raw.get("annotations", []):
            image_id = int(annotation["image_id"])
            image_row = image_rows.get(image_id)
            if image_row is None:
                raise ConfigError(
                    f"{annotation_path} references unknown image_id {annotation['image_id']}"
                )
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ConfigError(
                    f"{annotation_path} has malformed bbox for annotation {annotation.get('id')}"
                )
            x, y, width, height = [float(value) for value in bbox]
            if width <= 0 or height <= 0:
                raise ConfigError(
                    f"{annotation_path} has non-positive bbox for annotation {annotation.get('id')}"
                )
            source_class = categories.get(int(annotation["category_id"]), "unknown")
            canonical_class = class_mapping.get(source_class, "unknown")
            records_by_image[image_id].append(
                AnnotationRecord(
                    box=Box(x1=x, y1=y, x2=x + width, y2=y + height),
                    source_class=source_class,
                    canonical_class=canonical_class,
                    source_dataset="LeukemiaAttri",
                    metadata={"annotation_id": annotation.get("id")},
                    warnings=[]
                    if canonical_class != "unknown"
                    else [f"unmapped_source_class:{source_class}"],
                )
            )
        for image_id, image_row in sorted(image_rows.items()):
            image_path = image_root / str(image_row["file_name"])
            if not image_path.exists():
                raise InputNotFoundError(f"LeukemiaAttri image missing: {image_path}")
            width = int(image_row["width"])
            height = int(image_row["height"])
            records = records_by_image[image_id]
            candidate_records = [
                record for record in records if record.canonical_class == "candidate_wbc"
            ]
            sample_id = f"leukemia_attri__{domain}__{source_split}__{Path(str(image_row['file_name'])).stem}"
            source_classes = sorted({record.source_class for record in records})
            canonical_classes = sorted({record.canonical_class for record in candidate_records})
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "label_path": str(annotation_path),
                    "image_suffix": image_path.suffix,
                    "image_sha256": sha256_file(image_path),
                    "source_dataset": "LeukemiaAttri",
                    "source_split": source_split,
                    "unified_split": unified_split,
                    "patient_id": "",
                    "slide_id": "",
                    "acquisition_domain": domain,
                    "microscope": domain.split("_")[0] if "_" in domain else "",
                    "camera": domain.split("_")[-1] if "_" in domain else "",
                    "magnification": domain.split("_")[1] if len(domain.split("_")) >= 2 else "",
                    "annotation_completeness": annotation_completeness.value,
                    "source_classes": ";".join(source_classes),
                    "canonical_classes": ";".join(canonical_classes),
                    "locked_for_validation": "false",
                    "license_id": "verify_before_use",
                    "label_text": "".join(
                        _record_to_yolo_line(record, image_width=width, image_height=height)
                        for record in candidate_records
                    ),
                    "candidate_wbc_count": len(candidate_records),
                }
            )
    return samples


def _iter_images(image_root: Path) -> list[Path]:
    return [
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def _split_from_relative(relative: Path) -> str:
    first = relative.parts[0] if len(relative.parts) > 1 else "train"
    if first in {"train", "val", "test"}:
        return first
    if first == "validation":
        return "val"
    return "train"


def _sample_key(relative: Path) -> str:
    return "__".join(relative.with_suffix("").parts)


def _record_to_yolo_line(record: AnnotationRecord, *, image_width: int, image_height: int) -> str:
    x_center = ((record.box.x1 + record.box.x2) / 2) / image_width
    y_center = ((record.box.y1 + record.box.y2) / 2) / image_height
    width = (record.box.x2 - record.box.x1) / image_width
    height = (record.box.y2 - record.box.y1) / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def _reset_yolo_tree(output: Path) -> None:
    shutil.rmtree(output / "images", ignore_errors=True)
    shutil.rmtree(output / "labels", ignore_errors=True)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
