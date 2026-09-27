import pytest

from bloodfilm.classification.uncertainty import UncertaintyPolicy
from bloodfilm.evaluation.classifier import (
    calibration_summary,
    coverage_accuracy_rows,
    detailed_per_class_metrics,
)


def test_detailed_per_class_metrics_include_confusions_and_confidence() -> None:
    rows = detailed_per_class_metrics(
        probabilities=[
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.45, 0.55],
        ],
        labels=[0, 0, 1, 1],
        class_names=["basophil", "eosinophil"],
        high_confidence_threshold=0.60,
    )

    assert rows[0]["true_positives"] == 2
    assert rows[0]["false_positives"] == 1
    assert rows[0]["false_negatives"] == 0
    assert rows[1]["most_frequent_confusion_class"] == "basophil"
    assert rows[1]["high_confidence_error_count"] == 1
    assert rows[0]["mean_calibrated_confidence_correct"] == pytest.approx(0.85)


def test_calibration_summary_reports_reliability_bins() -> None:
    summary = calibration_summary(
        probabilities=[[0.8, 0.2], [0.4, 0.6]],
        labels=[0, 0],
        bins=2,
    )

    assert summary["negative_log_likelihood"] > 0
    assert summary["brier_score"] > 0
    assert summary["maximum_calibration_error"] >= summary["expected_calibration_error"]
    assert len(summary["reliability_bins"]) == 2


def test_coverage_accuracy_rows_apply_uncertainty_policy() -> None:
    rows = coverage_accuracy_rows(
        probabilities=[[0.9, 0.1], [0.51, 0.49], [0.3, 0.7]],
        labels=[0, 1, 1],
        class_names=["basophil", "eosinophil"],
        policy=UncertaintyPolicy(
            min_accept_confidence=0.8,
            min_accept_margin=0.2,
            unknown_below_confidence=0.35,
            require_review_for_classes=(),
        ),
    )

    by_status = {row["status"]: row for row in rows}
    assert by_status["accepted"]["count"] == 1
    assert by_status["accepted"]["accuracy"] == pytest.approx(1.0)
    assert by_status["review_required"]["count"] == 2
    assert by_status["review_required"]["error_count"] == 1
    assert by_status["accepted"]["error_coverage"] == pytest.approx(0.0)


def test_calibration_summary_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        calibration_summary([[0.5, 0.5]], [2], bins=2)


def test_parity_report_accepts_ok_evidence(tmp_path) -> None:
    import json

    from bloodfilm.classification.reports import parity_report

    evidence = tmp_path / "parity.json"
    evidence.write_text(json.dumps({"status": "ok", "detail": "ran"}), encoding="utf-8")

    assert parity_report(evidence)["status"] == "ok"
    assert parity_report(None)["status"] == "blocked"
