from pathlib import Path

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


def test_parity_tolerance_strings_match_canonical_declaration() -> None:
    from bloodfilm.classification import parity as parity_module
    from bloodfilm.classification.reports import parity_report

    blocked = parity_report(None)

    assert blocked["required_tolerance"] == {
        "embedding_cosine_similarity": parity_module.EMBEDDING_COSINE_MIN,
        "logits_allclose": parity_module.LOGITS_TOLERANCE_LABEL,
    }


def _acceptance_inputs(**overrides):
    from bloodfilm.schemas import MLL23_CANONICAL_CLASSES

    base = {
        "artifact_report": {"status": "ok"},
        "split_report": {
            "status": "ok",
            "leakage": {
                "leaking_group_count": 0,
                "leakage_free": None,
                "grouping_status": "unverified_image_level_surrogate",
            },
        },
        "protocol_report": {
            "status": "ok",
            "temperature_fit_split": "validation",
            "final_test_evaluation": {"status": "ok"},
        },
        "uncertainty_report": {
            "status": "configured",
            "policy": {"min_accept_confidence": 0.8},
        },
        "bundle_inspection": {
            "status": "ok",
            "bundle_dir": "models/mll23-dinobloom-b-mlp-v0.1",
            "bundle": {
                "num_classes": 18,
                "preprocessing_sha256": "abc",
            },
            "taxonomy": {"class_names": list(MLL23_CANONICAL_CLASSES)},
            "files": {},
        },
        "parity": {"status": "ok"},
        "output_dir": Path("outputs/reports"),
        "verification": {"pytest": "passed", "ruff": "passed", "mypy": "passed"},
    }
    base.update(overrides)
    return base


def test_acceptance_reports_limitations_without_hard_blockers() -> None:
    from bloodfilm.classification.reports import acceptance_report

    report = acceptance_report(**_acceptance_inputs())

    assert report["status"] == "accepted_with_limitations"
    assert "surrogate_grouping_unverified" in report["limitations"]


def test_acceptance_blocks_on_leakage_temperature_and_verification() -> None:
    from bloodfilm.classification.reports import acceptance_report

    leaked = _acceptance_inputs()
    leaked["split_report"] = {
        "status": "ok",
        "leakage": {"leaking_group_count": 2, "leakage_free": False},
    }
    assert "split_leakage" in acceptance_report(**leaked)["blockers"]

    hot_temp = _acceptance_inputs()
    hot_temp["protocol_report"] = {
        "status": "ok",
        "temperature_fit_split": "test",
        "final_test_evaluation": {"status": "ok"},
    }
    assert "temperature_fit_on_test" in acceptance_report(**hot_temp)["blockers"]

    no_verification = _acceptance_inputs()
    no_verification["verification"] = None
    assert "verification_missing" in acceptance_report(**no_verification)["blockers"]


def test_evaluation_protocol_reports_threshold_provenance() -> None:
    from bloodfilm.classification.reports import evaluation_protocol_report

    report = evaluation_protocol_report(
        {"selected": "mlp", "heads": {"mlp": {"eval_split": "validation"}}},
        {"row_count": 1, "ok_count": 1, "skipped_count": 0, "device": "cpu"},
        {"split": "test", "size": 1, "metrics": {}, "top2_accuracy": 1.0},
    )

    assert report["threshold_selection_split"] == "validation"
    assert "assumed" in str(report["threshold_selection_provenance"])
