from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from bloodfilm.data.integrity import find_duplicates
from bloodfilm.data.manifest import manifest_audit_report
from bloodfilm.errors import ImageDecodeError, InputNotFoundError, UnsupportedImageFormatError
from bloodfilm.imaging.io import load_image

__all__ = ["audit_dataset", "manifest_audit_report"]

TOP_LEVEL_CLASS = "top_level"


def audit_dataset(dataset_root: Path | str) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.exists():
        raise InputNotFoundError(f"Dataset root not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    class_counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    unreadable: list[str] = []
    hashable: list[Path] = []
    for path in files:
        relative = path.relative_to(root)
        class_counts[relative.parts[0] if len(relative.parts) > 1 else TOP_LEVEL_CLASS] += 1
        try:
            image = load_image(path)
        except (ImageDecodeError, UnsupportedImageFormatError):
            unreadable.append(str(path))
            continue
        widths.append(image.width)
        heights.append(image.height)
        hashable.append(path)
    return {
        "dataset_root": str(root),
        "total_files": len(files),
        "class_counts": dict(sorted(class_counts.items())),
        "dimensions": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
            "measured_images": len(widths),
        },
        "duplicates": find_duplicates(hashable),
        "unreadable": sorted(unreadable),
    }
