from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path
from typing import Any

from bloodfilm.errors import (
    ConfigError,
    ImageDecodeError,
    InputNotFoundError,
    UnsupportedImageFormatError,
)
from bloodfilm.imaging.io import SUPPORTED_IMAGE_EXTENSIONS, probe_image


def audit_txl_pbc_dataset(dataset_root: Path | str) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.exists():
        raise InputNotFoundError(f"TXL-PBC dataset root not found: {root}")
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise InputNotFoundError(f"TXL-PBC data.yaml not found: {data_yaml}")

    class_names = _load_yolo_names(data_yaml)
    wbc_class_id = _find_wbc_class_id(class_names)
    image_paths = sorted(
        path
        for path in (root / "images").rglob("*")
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    label_paths = sorted((root / "labels").rglob("*.txt")) if (root / "labels").exists() else []
    class_counts: Counter[int] = Counter()
    invalid_labels: list[dict[str, object]] = []

    for label_path in label_paths:
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parsed = _parse_yolo_label_line(line)
            if parsed is None:
                invalid_labels.append(
                    {"path": str(label_path), "line": line_number, "reason": "malformed_yolo_label"}
                )
                continue
            class_id, x_center, y_center, width, height = parsed
            if class_id not in class_names:
                invalid_labels.append(
                    {"path": str(label_path), "line": line_number, "reason": "unknown_class_id"}
                )
                continue
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                invalid_labels.append(
                    {"path": str(label_path), "line": line_number, "reason": "box_out_of_range"}
                )
                continue
            if width <= 0 or height <= 0:
                invalid_labels.append(
                    {"path": str(label_path), "line": line_number, "reason": "non_positive_box"}
                )
                continue
            class_counts[class_id] += 1

    dimensions: list[tuple[int, int]] = []
    unreadable_images: list[str] = []
    for path in image_paths:
        try:
            dimensions.append(_safe_probe(path))
        except (ImageDecodeError, UnsupportedImageFormatError):
            unreadable_images.append(str(path))
    return {
        "dataset": "TXL-PBC",
        "status": "ok"
        if not invalid_labels and not unreadable_images and image_paths and label_paths
        else "review_required",
        "dataset_root": str(root),
        "data_yaml": str(data_yaml),
        "source_revision": _git_revision(root),
        "images": {
            "count": len(image_paths),
            "splits": _split_counts(image_paths, root / "images"),
        },
        "labels": {
            "count": len(label_paths),
            "splits": _split_counts(label_paths, root / "labels"),
        },
        "classes": {
            _normalize_label(name): {"class_id": class_id, "box_count": class_counts[class_id]}
            for class_id, name in sorted(class_names.items())
        },
        "wbc_class_id": wbc_class_id,
        "dimensions": {
            "measured_images": len(dimensions),
            "min_width": min((item[0] for item in dimensions), default=None),
            "max_width": max((item[0] for item in dimensions), default=None),
            "min_height": min((item[1] for item in dimensions), default=None),
            "max_height": max((item[1] for item in dimensions), default=None),
        },
        "unreadable_images": unreadable_images,
        "invalid_labels": invalid_labels,
        "notes": [
            "TXL-PBC is a broad detection dataset; MLL23 remains the 18-class classifier source.",
            "Verify downloaded data.yaml class IDs before training; do not assume WBC numeric ID.",
        ],
    }


def _load_yolo_names(path: Path) -> dict[int, str]:
    parsed = _load_yolo_names_with_yaml(path)
    if parsed:
        return parsed
    names: dict[int, str] = {}
    in_names = False
    next_list_index = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            inline = line.removeprefix("names:").strip()
            if inline.startswith("[") and inline.endswith("]"):
                for index, value in enumerate(inline.strip("[]").split(",")):
                    names[index] = value.strip().strip("'\"")
            continue
        if in_names and line.startswith("-"):
            names[next_list_index] = line.removeprefix("-").strip().strip("'\"")
            next_list_index += 1
            continue
        if in_names and ":" in line:
            key, value = line.split(":", 1)
            if key.strip().isdigit():
                names[int(key.strip())] = value.strip().strip("'\"")
            elif not raw_line.startswith((" ", "\t")):
                in_names = False
    if not names:
        raise ConfigError(f"Could not parse YOLO class names from {path}")
    return names


def _load_yolo_names_with_yaml(path: Path) -> dict[int, str]:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(index): str(value) for index, value in raw_names.items()}
    return {}


def _find_wbc_class_id(class_names: dict[int, str]) -> int:
    matches = [
        class_id for class_id, name in class_names.items() if _normalize_label(name) == "wbc"
    ]
    if len(matches) != 1:
        raise ConfigError("TXL-PBC data.yaml must contain exactly one WBC class")
    return matches[0]


def _parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(part) for part in parts[1:])
    except ValueError:
        return None
    return class_id, x_center, y_center, width, height


def _split_counts(paths: list[Path], root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        relative = path.relative_to(root)
        split = relative.parts[0] if len(relative.parts) > 1 else "top_level"
        counts[split] += 1
    return dict(sorted(counts.items()))


def _safe_probe(path: Path) -> tuple[int, int]:
    metadata = probe_image(path)
    return metadata.width, metadata.height


def _git_revision(root: Path) -> str | None:
    git_dir = root / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref:"):
        return value
    ref_path = git_dir / value.removeprefix("ref:").strip()
    return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")
