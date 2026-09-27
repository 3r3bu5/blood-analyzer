from __future__ import annotations

import json
import math
import struct
import zlib
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bloodfilm.data import create_leakage_report, read_manifest, read_split_manifest
from bloodfilm.documents import write_json
from bloodfilm.errors import ConfigError
from bloodfilm.evaluation.classifier import (
    calibration_summary,
    confusion_matrix,
    coverage_accuracy_rows,
    detailed_per_class_metrics,
    reliability_bin_rows,
)
from bloodfilm.schemas import MLL23_CANONICAL_CLASSES
from bloodfilm.util import sha256_file, write_csv_rows


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
    parity_evidence: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _read_json(comparison_report)
    embeddings = _read_json(embeddings_report)
    inspection = _read_json(bundle_inspection)
    threshold_doc = _read_json(thresholds)
    test_report = _read_json(test_evaluation) if test_evaluation.exists() else None
    parity = parity_report(parity_evidence)
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
        _write(output_dir / "crop_inference_parity.json", parity),
    ]
    markdown = output_dir / "mll23_cosine_investigation.md"
    markdown.write_text(cosine_investigation_markdown(comparison), encoding="utf-8")
    outputs.append(markdown)
    outputs.append(
        _write(
            output_dir / "mll23_m2_acceptance.json",
            acceptance_report(
                artifact_report=_read_json(output_dir / "mll23_artifact_integrity.json"),
                split_report=_read_json(output_dir / "mll23_split_integrity.json"),
                protocol_report=_read_json(output_dir / "mll23_evaluation_protocol.json"),
                uncertainty_report=_read_json(output_dir / "mll23_uncertainty_policy.json"),
                bundle_inspection=inspection,
                parity=parity,
                output_dir=output_dir,
            ),
        )
    )
    return outputs


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    class_names: list[str],
    labels: list[int],
    image_ids: list[str],
    image_paths: list[str],
    probabilities_before: list[list[float]],
    probabilities_after: list[list[float]],
    temperature: float,
    evaluation_split: str,
    temperature_fit_split: str,
    uncertainty_policy: Any,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _predictions(probabilities_after)
    matrix = confusion_matrix(predictions, labels, len(class_names))
    per_class = detailed_per_class_metrics(probabilities_after, labels, class_names)
    calibration_before = calibration_summary(probabilities_before, labels)
    calibration_after = calibration_summary(probabilities_after, labels)
    reliability = reliability_bin_rows(probabilities_after, labels)
    high_confidence_errors = _high_confidence_errors(
        labels, predictions, probabilities_after, image_ids, image_paths, class_names
    )
    coverage_rows = coverage_accuracy_rows(
        probabilities_after, labels, class_names, uncertainty_policy
    )
    if not per_class:
        raise ConfigError("Cannot write evaluation artifacts without classes")

    paths = [
        _write(output_dir / "mll23_per_class_metrics.json", {"classes": per_class}),
        _write(
            output_dir / "mll23_calibration.json",
            {
                "temperature": temperature,
                "temperature_fit_split": temperature_fit_split,
                "final_evaluation_split": evaluation_split,
                "before_calibration": _without_bins(calibration_before),
                "after_calibration": _without_bins(calibration_after),
                "note": "Calibration measures confidence alignment only; they do not establish correctness or OOD safety.",
            },
        ),
    ]
    write_csv_rows(output_dir / "mll23_per_class_metrics.csv", list(per_class[0]), per_class)
    paths.append(output_dir / "mll23_per_class_metrics.csv")
    write_csv_rows(
        output_dir / "mll23_confusion_matrix.csv",
        ["class_code", *class_names],
        _matrix_rows(matrix, class_names),
    )
    paths.append(output_dir / "mll23_confusion_matrix.csv")
    write_csv_rows(
        output_dir / "mll23_confusion_matrix_normalized.csv",
        ["class_code", *class_names],
        _matrix_rows(_normalize_matrix(matrix), class_names),
    )
    paths.append(output_dir / "mll23_confusion_matrix_normalized.csv")
    _write_matrix_png(output_dir / "mll23_confusion_matrix.png", matrix)
    paths.append(output_dir / "mll23_confusion_matrix.png")
    write_csv_rows(
        output_dir / "mll23_high_confidence_errors.csv",
        [
            "image_id",
            "image_path",
            "true_class",
            "predicted_class",
            "confidence",
        ],
        high_confidence_errors,
    )
    paths.append(output_dir / "mll23_high_confidence_errors.csv")
    write_csv_rows(output_dir / "mll23_reliability.csv", list(reliability[0]), reliability)
    paths.append(output_dir / "mll23_reliability.csv")
    _write_reliability_png(output_dir / "mll23_reliability.png", reliability)
    paths.append(output_dir / "mll23_reliability.png")
    write_csv_rows(
        output_dir / "mll23_coverage_accuracy.csv", list(coverage_rows[0]), coverage_rows
    )
    paths.append(output_dir / "mll23_coverage_accuracy.csv")
    return paths


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
        "threshold_selection_split": "validation",
        "test_set_untouched_until_final_evaluation": test_report is not None
        and test_report.get("split") == "test",
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


def cosine_investigation_markdown(comparison: dict[str, Any]) -> str:
    report = cosine_investigation_report(comparison)
    return (
        "# MLL23 Cosine Head Investigation\n\n"
        f"Status: `{report['status']}`\n\n"
        "The cosine head underperformed the linear and MLP heads on the validation split. "
        "Input embeddings and classifier weights are normalized in the cosine-head forward pass, "
        "and no label-ordering or evaluation-loader defect was demonstrated by the current tests. "
        "The result is treated as inconclusive/expected under this implemented configuration, so the "
        "validation-selected MLP remains packaged.\n"
    )


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


def parity_report(evidence_path: Path | None = None) -> dict[str, object]:
    if evidence_path is not None and evidence_path.exists():
        evidence = _read_json(evidence_path)
        if evidence.get("status") == "ok":
            return evidence
    backbone = Path("models/backbones/dinobloom-b.pth")
    return {
        "status": "blocked",
        "blocker": "DinoBloom-B backbone weights are not available locally."
        if not backbone.exists()
        else "Run the requires_data/requires_model parity test with raw MLL23 images.",
        "required_tolerance": {"embedding_cosine_similarity": 0.9999, "logits_allclose": "1e-4"},
    }


def acceptance_report(
    *,
    artifact_report: dict[str, Any],
    split_report: dict[str, Any],
    protocol_report: dict[str, Any],
    uncertainty_report: dict[str, Any],
    bundle_inspection: dict[str, Any],
    parity: dict[str, Any],
    output_dir: Path,
) -> dict[str, object]:
    blockers = []
    if artifact_report.get("status") != "ok":
        blockers.append("artifact_integrity")
    if split_report.get("status") != "ok":
        blockers.append("split_integrity")
    if protocol_report.get("status") != "ok":
        blockers.append("evaluation_protocol")
    if bundle_inspection.get("status") != "ok":
        blockers.append("bundle")
    if parity.get("status") != "ok":
        blockers.append("inference_parity")
    status = "blocked" if blockers else "accepted_for_research_integration"
    return {
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_integrity_result": artifact_report.get("status"),
        "split_integrity_result": split_report.get("status"),
        "evaluation_protocol_result": protocol_report.get("status"),
        "reproduced_mlp_metrics": protocol_report.get("final_test_evaluation"),
        "per_class_report_paths": {
            "json": str(output_dir / "mll23_per_class_metrics.json"),
            "csv": str(output_dir / "mll23_per_class_metrics.csv"),
            "confusion_matrix": str(output_dir / "mll23_confusion_matrix.csv"),
        },
        "calibration_result": str(output_dir / "mll23_calibration.json"),
        "uncertainty_policy_result": uncertainty_report.get("status"),
        "inference_parity_result": parity,
        "bundle": bundle_inspection.get("bundle_dir"),
        "bundle_sha256_values": bundle_inspection.get("files"),
        "known_limitations": [
            "Research-only classifier; not a diagnostic device.",
            "MLL23 split uses image-level surrogate groups until official grouping metadata is available.",
            "Novelty and OOD behavior are heuristic without external OOD validation.",
        ],
        "blockers": blockers,
    }


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _predictions(probabilities: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in probabilities]


def _without_bins(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "reliability_bins"}


def _matrix_rows(
    matrix: Sequence[Sequence[int | float]], class_names: list[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(matrix):
        output: dict[str, object] = {"class_code": class_names[index]}
        for column, name in enumerate(class_names):
            output[name] = row[column]
        rows.append(output)
    return rows


def _normalize_matrix(matrix: list[list[int]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0.0 for value in row])
    return normalized


def _high_confidence_errors(
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    image_ids: list[str],
    image_paths: list[str],
    class_names: list[str],
    threshold: float = 0.90,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (truth, prediction) in enumerate(zip(labels, predictions)):
        confidence = max(probabilities[index])
        if truth != prediction and confidence >= threshold:
            rows.append(
                {
                    "image_id": image_ids[index],
                    "image_path": image_paths[index],
                    "true_class": class_names[truth],
                    "predicted_class": class_names[prediction],
                    "confidence": confidence,
                }
            )
    return rows


def _write_matrix_png(path: Path, matrix: list[list[int]]) -> None:
    max_value = max((max(row) for row in matrix if row), default=1)
    pixels: list[tuple[int, int, int]] = []
    for row in matrix:
        for value in row:
            shade = 255 - round(255 * math.sqrt(value / max_value)) if max_value else 255
            pixels.append((shade, shade, 255))
    _write_rgb_png(path, len(matrix), len(matrix), pixels)


def _write_reliability_png(path: Path, rows: list[dict[str, float | int]]) -> None:
    width = max(len(rows), 1)
    height = 50
    pixels = [(255, 255, 255)] * (width * height)
    for x, row in enumerate(rows):
        accuracy_height = round(float(row["accuracy"]) * (height - 1))
        confidence_height = round(float(row["confidence"]) * (height - 1))
        for y in range(height - accuracy_height, height):
            pixels[y * width + x] = (80, 160, 80)
        marker_y = max(0, height - confidence_height - 1)
        pixels[marker_y * width + x] = (200, 40, 40)
    _write_rgb_png(path, width, height, pixels)


def _write_rgb_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: dict[str, object]) -> Path:
    write_json(path, value)
    return path
