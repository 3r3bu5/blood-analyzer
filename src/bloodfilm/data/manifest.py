from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bloodfilm.data.mappings import load_label_mapping
from bloodfilm.documents import write_json
from bloodfilm.errors import (
    ImageDecodeError,
    InputNotFoundError,
    MappingError,
    UnsupportedImageFormatError,
)
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, load_image
from bloodfilm.schemas import (
    MLL23_CANONICAL_CLASSES,
    UNGROUPED_GROUP_PREFIX,
    InvalidManifestRow,
    ManifestRow,
)
from bloodfilm.util import sha256_file, write_csv_rows


@dataclass(frozen=True)
class ManifestBuildResult:
    valid_rows: list[ManifestRow]
    invalid_rows: list[InvalidManifestRow]
    class_distribution: dict[str, int]


def build_mll23_manifest(dataset_root: Path | str, mapping_path: Path | str) -> ManifestBuildResult:
    root = Path(dataset_root)
    mapping = load_label_mapping(mapping_path)
    if not root.exists():
        raise InputNotFoundError(f"MLL23 dataset root not found: {root}")
    if mapping.canonical_classes != MLL23_CANONICAL_CLASSES:
        raise MappingError("MLL23 mapping must contain the exact ordered 18 canonical classes")
    valid: list[ManifestRow] = []
    invalid: list[InvalidManifestRow] = []
    for image_path in _iter_dataset_files(root):
        source_folder = _source_folder(root, image_path)
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            invalid.append(
                InvalidManifestRow(
                    image_path=str(image_path),
                    source_folder=source_folder,
                    reason="UNSUPPORTED_IMAGE_FORMAT",
                    detail=f"Unsupported image extension: {image_path.suffix.lower()}",
                )
            )
            continue
        canonical = mapping.canonical_for(source_folder)
        if canonical is None:
            invalid.append(
                InvalidManifestRow(
                    image_path=str(image_path),
                    source_folder=source_folder,
                    reason="unmapped_label",
                    detail=f"No mapping for source folder {source_folder!r}",
                )
            )
            continue
        try:
            image = load_image(image_path)
        except (ImageDecodeError, UnsupportedImageFormatError) as exc:
            invalid.append(
                InvalidManifestRow(
                    image_path=str(image_path),
                    source_folder=source_folder,
                    reason=exc.code,
                    detail=str(exc),
                )
            )
            continue
        checksum = sha256_file(image_path)
        relative_path = image_path.relative_to(root).as_posix()
        valid.append(
            ManifestRow(
                image_id=checksum,
                image_path=relative_path,
                source_folder=source_folder,
                canonical_label=canonical,
                sha256=checksum,
                width=image.width,
                height=image.height,
                mode=image.mode,
                patient_or_source_group=f"{UNGROUPED_GROUP_PREFIX}{checksum}",
            )
        )
    counts = Counter(row.canonical_label for row in valid)
    distribution = {class_name: counts.get(class_name, 0) for class_name in MLL23_CANONICAL_CLASSES}
    return ManifestBuildResult(
        valid_rows=valid, invalid_rows=invalid, class_distribution=distribution
    )


def write_manifest(path: Path | str, rows: list[ManifestRow]) -> None:
    write_csv_rows(path, list(ManifestRow.__dataclass_fields__), [row.__dict__ for row in rows])


def write_invalid_manifest(path: Path | str, rows: list[InvalidManifestRow]) -> None:
    write_csv_rows(
        path, list(InvalidManifestRow.__dataclass_fields__), [row.__dict__ for row in rows]
    )


def write_checksum_manifest(path: Path | str, rows: list[ManifestRow]) -> None:
    write_csv_rows(
        path,
        ["image_id", "image_path", "sha256"],
        [
            {"image_id": row.image_id, "image_path": row.image_path, "sha256": row.sha256}
            for row in rows
        ],
    )


def read_manifest(path: Path | str) -> list[ManifestRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [
            ManifestRow(
                image_id=row["image_id"],
                image_path=row["image_path"],
                source_folder=row["source_folder"],
                canonical_label=row["canonical_label"],
                sha256=row["sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                mode=row["mode"],
                patient_or_source_group=row["patient_or_source_group"],
            )
            for row in csv.DictReader(handle)
        ]


def _iter_dataset_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _source_folder(root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root)
    if len(relative.parts) < 2:
        return ""
    return relative.parts[0]


def manifest_audit_report(result: ManifestBuildResult) -> dict[str, object]:
    invalid_by_reason: dict[str, int] = {}
    for row in result.invalid_rows:
        invalid_by_reason[row.reason] = invalid_by_reason.get(row.reason, 0) + 1
    return {
        "valid_image_count": len(result.valid_rows),
        "invalid_item_count": len(result.invalid_rows),
        "class_distribution": result.class_distribution,
        "invalid_by_reason": dict(sorted(invalid_by_reason.items())),
        "grouping_status": "unverified_image_level_surrogate",
        "independence_claim": False,
        "leakage_note": (
            "M0/M1 scaffold assigns one surrogate group per image until official MLL23 grouping "
            "metadata is available. Random image-level independence is not claimed."
        ),
    }


def run_build_manifest(
    dataset_root: Path | str,
    mapping_path: Path | str,
    *,
    output: Path | str,
    invalid_output: Path | str,
    checksum_output: Path | str,
    report_output: Path | str,
) -> Path:
    result = build_mll23_manifest(dataset_root, mapping_path)
    write_manifest(output, result.valid_rows)
    write_invalid_manifest(invalid_output, result.invalid_rows)
    write_checksum_manifest(checksum_output, result.valid_rows)
    report_path = Path(report_output)
    write_json(report_path, manifest_audit_report(result))
    return report_path
