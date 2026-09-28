from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError, InputNotFoundError

METRIC_ALIASES = {
    "precision": ["metrics/precision(B)", "precision", "P"],
    "recall": ["metrics/recall(B)", "recall", "R"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"],
}


def compare_kaggle_detector_runs(
    *,
    plan_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    plan = load_config_document(plan_path)
    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ConfigError("Kaggle training plan must contain commands")

    experiments = [_experiment_row(command) for command in commands]
    deltas = _pairwise_deltas(experiments)
    report = {
        "schema_version": 1,
        "research_only": True,
        "status": "comparison_only",
        "selection": None,
        "packageable": False,
        "requires_human_review": True,
        "plan": str(plan_path),
        "data_yaml": plan.get("data_yaml"),
        "experiments": experiments,
        "pairwise_deltas": deltas,
        "blocked_until": [
            "target smoke evaluation is implemented and passed",
            "field-sized false-box checks are implemented and passed",
            "human review accepts detector provenance and metrics",
        ],
    }
    write_json(output_path, report)
    return report


def _experiment_row(command: Any) -> dict[str, Any]:
    if not isinstance(command, dict):
        raise ConfigError("Training plan command entries must be objects")
    experiment = str(command.get("experiment", ""))
    expected_weights = Path(str(command.get("expected_best_weights", "")))
    expected_test_dir = Path(str(command.get("expected_test_dir", "")))
    train_dir = expected_weights.parents[1] if len(expected_weights.parents) >= 2 else Path("")
    train_metrics = _read_metrics(train_dir / "results.csv")
    test_metrics = _read_metrics(expected_test_dir / "results.csv")
    return {
        "experiment": experiment,
        "purpose": command.get("purpose", ""),
        "initial_weights": command.get("initial_weights", ""),
        "expected_best_weights": str(expected_weights),
        "best_weights_exists": expected_weights.exists(),
        "train_dir": str(train_dir),
        "test_dir": str(expected_test_dir),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "artifact_status": _artifact_status(expected_weights, train_metrics, test_metrics),
    }


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path), "metrics": {}}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"status": "empty", "path": str(path), "metrics": {}}
    row = rows[-1]
    metrics: dict[str, float] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        value = _first_float(row, aliases)
        if value is not None:
            metrics[canonical] = value
    return {"status": "ok", "path": str(path), "metrics": metrics}


def _first_float(row: dict[str, str], aliases: list[str]) -> float | None:
    normalized = {key.strip(): value for key, value in row.items()}
    for alias in aliases:
        raw = normalized.get(alias)
        if raw is None or raw == "":
            continue
        return float(raw)
    return None


def _artifact_status(
    weights: Path, train_metrics: dict[str, Any], test_metrics: dict[str, Any]
) -> str:
    if not weights.exists():
        return "missing_best_weights"
    if train_metrics["status"] != "ok":
        return "missing_train_metrics"
    if test_metrics["status"] != "ok":
        return "missing_test_metrics"
    return "complete"


def _pairwise_deltas(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(experiments) != 2:
        return []
    left, right = experiments
    if left["test_metrics"]["status"] != "ok" or right["test_metrics"]["status"] != "ok":
        return []
    deltas: dict[str, float] = {}
    left_metrics = left["test_metrics"]["metrics"]
    right_metrics = right["test_metrics"]["metrics"]
    for metric in sorted(set(left_metrics) & set(right_metrics)):
        deltas[metric] = round(right_metrics[metric] - left_metrics[metric], 6)
    return [
        {
            "left_experiment": left["experiment"],
            "right_experiment": right["experiment"],
            "metric_source": "test_results_csv",
            "delta_is_right_minus_left": deltas,
            "selection_made": False,
        }
    ]


def require_plan_exists(plan_path: Path | str) -> None:
    if not Path(plan_path).exists():
        raise InputNotFoundError(f"Kaggle training plan not found: {plan_path}")
