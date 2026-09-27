from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from bloodfilm.data import create_leakage_report, read_manifest, read_split_manifest
from bloodfilm.documents import write_json
from bloodfilm.schemas import MLL23_CANONICAL_CLASSES
from bloodfilm.util import sha256_file


def write_post_training_reports(
    *,
    manifest: Path,
    splits: Path,
    invalid_manifest: Path,
    embeddings_report: Path,
    comparison_report: Path,
    bundle_inspection: Path,
    thresholds: Path,
    test_evaluation: Path,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _read_json(comparison_report)
    embeddings = _read_json(embeddings_report)
    inspection = _read_json(bundle_inspection)
    threshold_doc = _read_json(thresholds)
    test_report = _read_json(test_evaluation) if test_evaluation.exists() else None
    outputs = [
        _write(
            output_dir / "mll23_artifact_integrity.json",
            artifact_integrity_report(
                [
                    manifest,
                    splits,
                    invalid_manifest,
                    embeddings_report,
                    comparison_report,
                    bundle_inspection,
                    thresholds,
                ]
            ),
        ),
        _write(output_dir / "mll23_split_integrity.json", split_integrity_report(manifest, splits)),
        _write(
            output_dir / "mll23_evaluation_protocol.json",
            evaluation_protocol_report(comparison, embeddings, test_report),
        ),
        _write(
            output_dir / "mll23_cosine_investigation.json", cosine_investigation_report(comparison)
        ),
        _write(
            output_dir / "mll23_uncertainty_policy.json",
            uncertainty_policy_report(threshold_doc, comparison, inspection),
        ),
    ]
    return outputs


def artifact_integrity_report(paths: list[Path]) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for path in paths:
        artifacts[str(path)] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    missing = [path for path in artifacts if not artifacts[path]["exists"]]  # type: ignore[index]
    return {"status": "blocked" if missing else "ok", "missing": missing, "artifacts": artifacts}


def split_integrity_report(manifest_path: Path, split_path: Path) -> dict[str, object]:
    manifest_rows = read_manifest(manifest_path)
    split_rows = read_split_manifest(split_path)
    manifest_ids = [row.image_id for row in manifest_rows]
    split_ids = [row.image_id for row in split_rows]
    split_counts = Counter(row.split for row in split_rows)
    class_by_split: dict[str, dict[str, int]] = {}
    for row in split_rows:
        counts = class_by_split.setdefault(row.split, {})
        counts[row.canonical_label] = counts.get(row.canonical_label, 0) + 1
    duplicate_manifest_ids = _duplicates(manifest_ids)
    duplicate_split_ids = _duplicates(split_ids)
    missing_from_splits = sorted(set(manifest_ids) - set(split_ids))
    orphan_split_ids = sorted(set(split_ids) - set(manifest_ids))
    return {
        "status": "ok"
        if not duplicate_manifest_ids
        and not duplicate_split_ids
        and not missing_from_splits
        and not orphan_split_ids
        else "blocked",
        "manifest_count": len(manifest_rows),
        "split_count": len(split_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_by_split": {
            split: {name: counts.get(name, 0) for name in MLL23_CANONICAL_CLASSES}
            for split, counts in sorted(class_by_split.items())
        },
        "duplicate_manifest_ids": duplicate_manifest_ids,
        "duplicate_split_ids": duplicate_split_ids,
        "missing_from_splits": missing_from_splits[:50],
        "orphan_split_ids": orphan_split_ids[:50],
        "leakage": create_leakage_report(split_rows),
    }


def evaluation_protocol_report(
    comparison: dict[str, Any], embeddings: dict[str, Any], test_report: dict[str, Any] | None
) -> dict[str, object]:
    selected = str(comparison["selected"])
    selected_metrics = comparison["heads"][selected]
    final_test: dict[str, object]
    status = "ok"
    if test_report is None:
        status = "blocked_final_test_not_run"
        final_test = {
            "status": "blocked",
            "blocker": "Torch-capable environment required to load mll23_embeddings.pt and outputs/checkpoints/mlp.pt for a one-time test split evaluation.",
            "runnable_command": "bloodfilm classifier evaluate-cache --checkpoint outputs/checkpoints/mlp.pt --embeddings data/embeddings/mll23_embeddings.pt --split test --output outputs/reports/mll23_test_evaluation.json",
        }
    else:
        final_test = {
            "status": "ok",
            "split": test_report.get("split"),
            "size": test_report.get("size"),
            "metrics": test_report.get("metrics"),
            "top2_accuracy": test_report.get("top2_accuracy"),
            "ece_before": test_report.get("ece_before"),
            "ece_after": test_report.get("ece_after"),
        }
    return {
        "status": status,
        "selected_head": selected,
        "selection_metric": comparison.get("selection_metric"),
        "selection_split": selected_metrics.get("eval_split"),
        "validation_used_for_selection": selected_metrics.get("eval_split") == "validation",
        "temperature_fit_split": selected_metrics.get("eval_split"),
        "embedding_cache": {
            "rows": embeddings.get("row_count"),
            "ok_count": embeddings.get("ok_count"),
            "skipped_count": embeddings.get("skipped_count"),
            "device": embeddings.get("device"),
        },
        "final_test_evaluation": final_test,
        "note": "Head selection and temperature fitting used validation. Final test metrics are reported only when mll23_test_evaluation.json is present.",
    }


def cosine_investigation_report(comparison: dict[str, Any]) -> dict[str, object]:
    cosine = comparison["heads"].get("cosine", {})
    linear = comparison["heads"].get("linear", {})
    mlp = comparison["heads"].get("mlp", {})
    return {
        "status": "investigated_no_code_defect_confirmed",
        "cosine_validation": cosine,
        "linear_validation_macro_f1": linear.get("macro_f1"),
        "mlp_validation_macro_f1": mlp.get("macro_f1"),
        "observations": [
            "Cosine head underperformed linear and MLP on the same validation split.",
            "No retraining or implementation change was made because no defect was isolated by a failing test.",
            "Cosine remains an experimental peer rather than the packaged head.",
        ],
    }


def uncertainty_policy_report(
    thresholds: dict[str, Any], comparison: dict[str, Any], inspection: dict[str, Any]
) -> dict[str, object]:
    selected = str(comparison["selected"])
    return {
        "status": "configured",
        "policy": thresholds,
        "selected_head": selected,
        "validation_calibration": comparison["heads"][selected],
        "bundle_status": inspection.get("status"),
        "actions": {
            "accepted": "High-confidence prediction; still research-only.",
            "review_required": "Prediction requires human review before any downstream research use.",
            "unknown": "Low-confidence crop is treated as outside supported classifier confidence.",
        },
    }


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: dict[str, object]) -> Path:
    write_json(path, value)
    return path
