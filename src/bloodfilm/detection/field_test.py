from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError, InputNotFoundError, ModelLoadError


def build_source_test_subset(
    *,
    manifest_path: Path | str,
    data_yaml_path: Path | str,
    output_root: Path | str,
    source_dataset: str = "LeukemiaAttri",
    source_split: str = "test",
) -> dict[str, Any]:
    """Copy one source's held-out images into a standalone YOLO test folder.

    The unified ``test`` split is TXL-PBC-only by design (LeukemiaAttri's own
    test fold is remapped to unified ``val``), so field-level generalization
    needs this separate scoring subset. No training data is touched.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise InputNotFoundError(f"Detector manifest not found: {manifest}")
    base_yaml = _resolve_base_yaml(Path(data_yaml_path))
    rows = [
        row
        for row in _read_csv(manifest)
        if row.get("source_dataset") == source_dataset and row.get("source_split") == source_split
    ]
    if not rows:
        raise ConfigError(
            f"No manifest rows for source_dataset={source_dataset!r} "
            f"with source_split={source_split!r} in {manifest}"
        )
    output = Path(output_root)
    images_dir = output / "images" / "test"
    labels_dir = output / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied = 0
    for row in sorted(rows, key=lambda item: item.get("sample_id", "")):
        image = Path(str(row.get("image_path", "")))
        label = Path(str(row.get("label_path", "")))
        if not image.exists() or not label.exists():
            missing.append(str(row.get("sample_id", "")))
            continue
        shutil.copy2(image, images_dir / image.name)
        shutil.copy2(label, labels_dir / f"{image.stem}.txt")
        copied += 1
    if missing:
        raise InputNotFoundError(
            f"{len(missing)} subset files are missing under the unified dataset "
            f"(first: {', '.join(missing[:5])})"
        )
    data_yaml = {
        "path": str(output.resolve()),
        "train": base_yaml["train"],
        "val": base_yaml["val"],
        "test": "images/test",
        "names": {"0": "candidate_wbc"},
    }
    yaml_path = output / "data.yaml"
    write_json(yaml_path, data_yaml)
    domains = Counter(str(row.get("acquisition_domain", "")) for row in rows)
    return {
        "schema_version": 1,
        "research_only": True,
        "status": "ok",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "sample_count": copied,
        "domain_counts": dict(sorted(domains.items())),
        "output_root": str(output),
        "data_yaml": str(yaml_path),
    }


def score_weights_on_subset(
    *,
    weights: Path | str,
    data_yaml: Path | str,
    split: str = "test",
    image_size: int = 640,
) -> dict[str, Any]:
    """Score one detector checkpoint on a prepared subset without training."""
    weights_path = Path(weights)
    if not weights_path.exists():
        raise InputNotFoundError(f"Detector weights not found: {weights_path}")
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise InputNotFoundError(f"Subset data.yaml not found: {yaml_path}")
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModelLoadError(
            "Subset scoring requires ultralytics; install the ml extras first"
        ) from exc
    try:
        model = YOLO(str(weights_path))
    except Exception as exc:
        raise ModelLoadError(f"Detector weights are not loadable: {weights_path}") from exc
    try:
        results = model.val(data=str(yaml_path), split=split, imgsz=image_size, verbose=False)
    except Exception as exc:
        raise ModelLoadError(f"Detector val failed for {weights_path}: {exc}") from exc
    metrics = {str(key): float(value) for key, value in dict(results.results_dict).items()}
    return {
        "weights": str(weights_path),
        "data_yaml": str(yaml_path),
        "split": split,
        "image_size": image_size,
        "metrics": metrics,
        "save_dir": str(results.save_dir),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_base_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        raise InputNotFoundError(f"Base data.yaml not found: {path}")
    raw = load_config_document(path)
    base = Path(str(raw.get("path", path.parent)))
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    resolved = {}
    for key in ("train", "val"):
        value = Path(str(raw.get(key, "")))
        resolved[key] = str((base / value).resolve()) if not value.is_absolute() else str(value)
    return resolved
