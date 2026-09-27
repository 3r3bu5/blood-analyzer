from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from bloodfilm.documents import write_json
from bloodfilm.errors import InputNotFoundError
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS


def prepare_wbc_yolo_dataset(
    dataset_root: Path | str,
    output_root: Path | str,
    *,
    wbc_class_id: int,
) -> dict[str, Any]:
    source = Path(dataset_root)
    output = Path(output_root)
    image_root = source / "images"
    label_root = source / "labels"
    if not image_root.exists() or not label_root.exists():
        raise InputNotFoundError(f"TXL-PBC YOLO images/labels not found under {source}")
    images_copied = 0
    labels_written = 0
    wbc_boxes = 0
    for image in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ):
        relative = image.relative_to(image_root)
        target_image = output / "images" / relative
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        images_copied += 1

        source_label = label_root / relative.with_suffix(".txt")
        target_label = output / "labels" / relative.with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)
        kept = _wbc_label_lines(source_label, wbc_class_id=wbc_class_id)
        target_label.write_text("".join(kept), encoding="utf-8")
        labels_written += 1
        wbc_boxes += len(kept)
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {"0": "wbc_candidate"},
    }
    write_json(output / "data.yaml", data_yaml)
    report = {
        "status": "ok",
        "source_root": str(source),
        "output_root": str(output),
        "wbc_class_id": wbc_class_id,
        "images_copied": images_copied,
        "labels_written": labels_written,
        "wbc_boxes": wbc_boxes,
        "data_yaml": str(output / "data.yaml"),
    }
    write_json(output / "wbc_derivative_report.json", report)
    return report


def _wbc_label_lines(label_path: Path, *, wbc_class_id: int) -> list[str]:
    if not label_path.exists():
        return []
    kept = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == str(wbc_class_id):
            kept.append("0 " + " ".join(parts[1:]) + "\n")
    return kept
