from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bloodfilm.classification.dinobloom import PREPROCESSING
from bloodfilm.classification.uncertainty import UncertaintyPolicy
from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError, InputNotFoundError, ModelLoadError
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
    dataset_metadata: dict[str, object] | None = None,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    head_source = Path(head_checkpoint)
    head_target = bundle / "head.pt"
    if head_source.resolve() != head_target.resolve():
        shutil.copyfile(head_source, head_target)
    head_sha256 = sha256_file(head_target)

    bundle_doc = {
        "schema_version": 1,
        "name": bundle.name,
        "version": "0.1",
        "created_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "backbone": "DinoBloom-B",
        "backbone_expected_path": "models/backbones/dinobloom-b.pth",
        "backbone_sha256": backbone_sha256,
        "head": head_name,
        "head_architecture": head_name,
        "num_classes": len(class_names),
        "embedding_dim": 768,
        "head_file": "head.pt",
        "head_sha256": head_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "dataset": dataset_metadata or {"name": "MLL23"},
        "software_versions": {"python": sys.version.split()[0]},
        "known_limitations": [
            "Research-only artifact; not a diagnostic device.",
            "MLL23 grouping uses image-level surrogate groups until official grouping metadata is available.",
            "OOD and novelty behavior require external validation.",
        ],
    }
    write_json(bundle / "bundle.json", bundle_doc)
    write_json(
        bundle / "taxonomy.json",
        {"schema_version": 1, "class_names": class_names, "class_count": len(class_names)},
    )
    write_json(
        bundle / "preprocessing.json",
        {
            "recipe": PREPROCESSING,
            "input_image_size": PREPROCESSING["input_size"],
            "colour_mode": "RGB",
            "resize_behavior": PREPROCESSING["resize_method"],
            "crop_behavior": PREPROCESSING["crop_method"],
            "normalization": {"mean": PREPROCESSING["mean"], "std": PREPROCESSING["std"]},
        },
    )
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
    checksum_errors = _bundle_checksum_errors(bundle) if not missing else []

    return {
        "status": "incomplete" if missing else "invalid" if checksum_errors else "ok",
        "bundle_dir": str(bundle),
        "bundle": bundle_doc,
        "taxonomy": taxonomy,
        "files": files,
        "missing_files": missing,
        "checksum_errors": checksum_errors,
    }


def verify_bundle_head(bundle_doc: dict[str, Any], head_path: Path | str) -> None:
    """Fail clearly when head.pt does not match bundle.json head_sha256."""
    expected_head = bundle_doc.get("head_sha256")
    if isinstance(expected_head, str) and sha256_file(head_path) != expected_head:
        raise ConfigError("Classifier bundle head.pt checksum does not match bundle metadata")


def verify_backbone_checksum(bundle_doc: dict[str, Any], backbone_weights: Path | str) -> None:
    """Fail clearly when backbone weights do not match bundle metadata.

    Missing weight files are left to the backbone loader so callers keep the
    actionable missing-asset error.
    """
    expected_backbone_sha = str(bundle_doc.get("backbone_sha256", ""))
    weights_path = Path(backbone_weights)
    if (
        expected_backbone_sha
        and weights_path.exists()
        and sha256_file(weights_path) != expected_backbone_sha
    ):
        raise ModelLoadError("DinoBloom-B backbone checksum does not match bundle metadata")


def load_bundle_documents(bundle_dir: Path | str) -> dict[str, Any]:
    report = inspect_bundle(bundle_dir)
    if report["missing_files"]:
        raise ConfigError(f"Classifier bundle is incomplete: {report['missing_files']}")
    bundle = Path(bundle_dir)
    _verify_sha256sums(bundle)
    bundle_doc = load_config_document(bundle / "bundle.json")
    verify_bundle_head(bundle_doc, bundle / "head.pt")
    return {
        "bundle": bundle_doc,
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


def _verify_sha256sums(bundle: Path) -> None:
    errors = _bundle_checksum_errors(bundle)
    if errors:
        raise ConfigError(f"Classifier bundle checksum validation failed: {errors}")


def _bundle_checksum_errors(bundle: Path) -> list[str]:
    errors: list[str] = []
    checksums = bundle / "sha256sums.txt"
    if not checksums.exists():
        return ["sha256sums.txt missing"]
    seen: set[str] = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        clean_name = name.strip()
        seen.add(clean_name)
        path = bundle / clean_name
        if not path.exists():
            errors.append(f"checksum target missing: {clean_name}")
        elif sha256_file(path) != expected:
            errors.append(f"checksum mismatch: {clean_name}")
    for required in REQUIRED_BUNDLE_FILES:
        if required != "sha256sums.txt" and required not in seen:
            errors.append(f"checksum entry missing: {required}")
    bundle_json = bundle / "bundle.json"
    head = bundle / "head.pt"
    if bundle_json.exists() and head.exists():
        try:
            verify_bundle_head(load_config_document(bundle_json), head)
        except ConfigError:
            errors.append("head.pt mismatch with bundle.json head_sha256")
    return errors


def _model_card(bundle_doc: dict[str, Any]) -> str:
    return (
        f"# {bundle_doc['name']}\n\n"
        "Research-only WBC crop classifier bundle. This artifact is not a diagnostic device "
        "and must not be used for clinical decision-making without independent validation.\n"
    )
