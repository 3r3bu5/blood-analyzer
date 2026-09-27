from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from bloodfilm.classification.dinobloom import PREPROCESSING
from bloodfilm.classification.uncertainty import UncertaintyPolicy
from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError, InputNotFoundError
from bloodfilm.util import sha256_file

REQUIRED_BUNDLE_FILES = (
    "head.pt",
    "bundle.json",
    "taxonomy.json",
    "preprocessing.json",
    "calibration.json",
    "thresholds.json",
    "metrics.json",
    "model_card.md",
    "sha256sums.txt",
)


def write_bundle_metadata(
    bundle_dir: Path | str,
    *,
    head_checkpoint: Path | str,
    class_names: list[str],
    head_name: str,
    backbone_sha256: str,
    preprocessing_sha256: str,
    temperature: float,
    validation_metrics: dict[str, Any],
    uncertainty_policy: UncertaintyPolicy,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    head_source = Path(head_checkpoint)
    head_target = bundle / "head.pt"
    if head_source.resolve() != head_target.resolve():
        shutil.copyfile(head_source, head_target)

    bundle_doc = {
        "schema_version": 1,
        "name": bundle.name,
        "research_only": True,
        "backbone": "DinoBloom-B",
        "backbone_sha256": backbone_sha256,
        "head": head_name,
        "num_classes": len(class_names),
        "head_file": "head.pt",
        "preprocessing_sha256": preprocessing_sha256,
    }
    write_json(bundle / "bundle.json", bundle_doc)
    write_json(
        bundle / "taxonomy.json",
        {"schema_version": 1, "class_names": class_names, "class_count": len(class_names)},
    )
    write_json(bundle / "preprocessing.json", {"recipe": PREPROCESSING})
    write_json(
        bundle / "calibration.json", {"temperature": temperature, "method": "temperature_scaling"}
    )
    write_json(bundle / "thresholds.json", uncertainty_policy.to_dict())
    write_json(bundle / "metrics.json", validation_metrics)
    (bundle / "model_card.md").write_text(_model_card(bundle_doc), encoding="utf-8")
    _write_sha256sums(bundle)
    return bundle


def inspect_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    if not bundle.exists():
        raise InputNotFoundError(f"Classifier bundle not found: {bundle}")
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in REQUIRED_BUNDLE_FILES:
        path = bundle / name
        exists = path.exists()
        if not exists:
            missing.append(name)
        files[name] = {
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": sha256_file(path) if exists and path.is_file() else None,
        }

    bundle_doc: dict[str, Any] = {}
    taxonomy: dict[str, Any] = {}
    if (bundle / "bundle.json").exists():
        bundle_doc = load_config_document(bundle / "bundle.json")
    if (bundle / "taxonomy.json").exists():
        taxonomy = load_config_document(bundle / "taxonomy.json")

    return {
        "status": "incomplete" if missing else "ok",
        "bundle_dir": str(bundle),
        "bundle": bundle_doc,
        "taxonomy": taxonomy,
        "files": files,
        "missing_files": missing,
    }


def load_bundle_documents(bundle_dir: Path | str) -> dict[str, Any]:
    report = inspect_bundle(bundle_dir)
    if report["missing_files"]:
        raise ConfigError(f"Classifier bundle is incomplete: {report['missing_files']}")
    bundle = Path(bundle_dir)
    return {
        "bundle": load_config_document(bundle / "bundle.json"),
        "taxonomy": load_config_document(bundle / "taxonomy.json"),
        "preprocessing": load_config_document(bundle / "preprocessing.json"),
        "calibration": load_config_document(bundle / "calibration.json"),
        "thresholds": load_config_document(bundle / "thresholds.json"),
        "metrics": load_config_document(bundle / "metrics.json"),
    }


def _write_sha256sums(bundle: Path) -> None:
    rows = []
    for name in REQUIRED_BUNDLE_FILES:
        if name == "sha256sums.txt":
            continue
        path = bundle / name
        if path.exists():
            rows.append(f"{sha256_file(path)}  {name}")
    (bundle / "sha256sums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _model_card(bundle_doc: dict[str, Any]) -> str:
    return (
        f"# {bundle_doc['name']}\n\n"
        "Research-only WBC crop classifier bundle. This artifact is not a diagnostic device "
        "and must not be used for clinical decision-making without independent validation.\n"
    )
