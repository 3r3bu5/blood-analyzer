from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import InputNotFoundError
from bloodfilm.util import sha256_file

REQUIRED_DETECTOR_BUNDLE_FILES = (
    "weights.pt",
    "bundle.json",
    "config.json",
    "thresholds.json",
    "metrics.json",
    "dataset_audit.json",
    "model_card.md",
    "sha256sums.txt",
)


def write_detector_bundle(
    bundle_dir: Path | str,
    *,
    weights: Path | str,
    config: Path | str,
    metrics: Path | str,
    thresholds: Path | str,
    audit_report: Path | str,
    model_name: str,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    weights_path = _require_file(weights)
    shutil.copyfile(weights_path, bundle / "weights.pt")
    shutil.copyfile(_require_file(config), bundle / "config.json")
    shutil.copyfile(_require_file(metrics), bundle / "metrics.json")
    shutil.copyfile(_require_file(thresholds), bundle / "thresholds.json")
    shutil.copyfile(_require_file(audit_report), bundle / "dataset_audit.json")
    metrics_doc = load_config_document(bundle / "metrics.json")
    thresholds_doc = load_config_document(bundle / "thresholds.json")
    bundle_doc = {
        "schema_version": 1,
        "name": model_name,
        "version": "0.1",
        "created_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "implementation": "ultralytics_yolo",
        "architecture": "YOLO26n",
        "weights_file": "weights.pt",
        "weights_sha256": sha256_file(bundle / "weights.pt"),
        "config_file": "config.json",
        "thresholds_file": "thresholds.json",
        "metrics_file": "metrics.json",
        "dataset_audit_file": "dataset_audit.json",
        "software_versions": {"python": sys.version.split()[0]},
        "known_limitations": [
            "Research-only detector; not a diagnostic device.",
            "TXL-PBC detects broad WBC candidates and does not establish 18-class morphology.",
            "Local microscope-field validation is still required before operational use.",
        ],
    }
    write_json(bundle / "bundle.json", bundle_doc)
    (bundle / "model_card.md").write_text(
        _model_card(bundle_doc, metrics_doc, thresholds_doc), encoding="utf-8"
    )
    _write_sha256sums(bundle)
    return bundle


def inspect_detector_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    if not bundle.exists():
        raise InputNotFoundError(f"Detector bundle not found: {bundle}")
    missing = [name for name in REQUIRED_DETECTOR_BUNDLE_FILES if not (bundle / name).exists()]
    checksum_errors = [] if missing else _checksum_errors(bundle)
    return {
        "status": "incomplete" if missing else "invalid" if checksum_errors else "ok",
        "bundle_dir": str(bundle),
        "missing_files": missing,
        "checksum_errors": checksum_errors,
        "bundle": load_config_document(bundle / "bundle.json")
        if (bundle / "bundle.json").exists()
        else {},
    }


def _require_file(path: Path | str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise InputNotFoundError(f"Detector bundle input not found: {file_path}")
    return file_path


def _write_sha256sums(bundle: Path) -> None:
    rows = []
    for name in REQUIRED_DETECTOR_BUNDLE_FILES:
        if name == "sha256sums.txt":
            continue
        path = bundle / name
        if path.exists():
            rows.append(f"{sha256_file(path)}  {name}")
    (bundle / "sha256sums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _checksum_errors(bundle: Path) -> list[str]:
    checksums = bundle / "sha256sums.txt"
    if not checksums.exists():
        return ["sha256sums.txt missing"]
    errors = []
    seen: set[str] = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected_checksum, name = line.split(maxsplit=1)
        clean_name = name.strip()
        seen.add(clean_name)
        path = bundle / clean_name
        if not path.exists():
            errors.append(f"checksum target missing: {clean_name}")
        elif sha256_file(path) != expected_checksum:
            errors.append(f"checksum mismatch: {clean_name}")
    for required in REQUIRED_DETECTOR_BUNDLE_FILES:
        if required != "sha256sums.txt" and required not in seen:
            errors.append(f"checksum entry missing: {required}")
    bundle_doc = bundle / "bundle.json"
    weights = bundle / "weights.pt"
    if bundle_doc.exists() and weights.exists():
        expected_weights = load_config_document(bundle_doc).get("weights_sha256")
        if isinstance(expected_weights, str) and sha256_file(weights) != expected_weights:
            errors.append("weights.pt mismatch with bundle.json weights_sha256")
    return errors


def _model_card(
    bundle_doc: dict[str, Any], metrics_doc: dict[str, Any], thresholds_doc: dict[str, Any]
) -> str:
    metric_lines = "\n".join(
        f"- `{key}`: `{value}`" for key, value in metrics_doc.items() if not isinstance(value, list)
    )
    threshold_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in thresholds_doc.items())
    limitations = "\n".join(f"- {item}" for item in bundle_doc["known_limitations"])
    return (
        f"# {bundle_doc['name']}\n\n"
        "Research-only WBC candidate detector bundle. This artifact is not a diagnostic device.\n\n"
        "## Metrics\n\n"
        f"{metric_lines}\n\n"
        "## Thresholds\n\n"
        f"{threshold_lines}\n\n"
        "## Known limitations\n\n"
        f"{limitations}\n"
    )
