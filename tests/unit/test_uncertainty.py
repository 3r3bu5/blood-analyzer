import pytest

from bloodfilm.classification.uncertainty import UncertaintyPolicy, decide_prediction


def test_uncertainty_accepts_confident_separated_prediction() -> None:
    decision = decide_prediction(
        [0.90, 0.05, 0.05],
        ["basophil", "eosinophil", "monocyte"],
        UncertaintyPolicy(min_accept_confidence=0.80, min_accept_margin=0.20),
    )

    assert decision.status == "accepted"
    assert decision.label == "basophil"
    assert decision.confidence == pytest.approx(0.90)
    assert decision.reasons == []


def test_uncertainty_requires_review_for_ambiguous_prediction() -> None:
    decision = decide_prediction(
        [0.52, 0.44, 0.04],
        ["basophil", "eosinophil", "monocyte"],
        UncertaintyPolicy(min_accept_confidence=0.50, min_accept_margin=0.20),
    )

    assert decision.status == "review_required"
    assert decision.label == "basophil"
    assert "margin_below_minimum" in decision.reasons


def test_uncertainty_marks_low_confidence_as_unknown() -> None:
    decision = decide_prediction(
        [0.34, 0.33, 0.33],
        ["basophil", "eosinophil", "monocyte"],
        UncertaintyPolicy(unknown_below_confidence=0.35),
    )

    assert decision.status == "unknown"
    assert decision.label is None
    assert "confidence_below_unknown_threshold" in decision.reasons


def test_uncertainty_requires_review_for_configured_classes() -> None:
    decision = decide_prediction(
        [0.95, 0.03, 0.02],
        ["myeloblast", "eosinophil", "monocyte"],
        UncertaintyPolicy(min_accept_confidence=0.80, require_review_for_classes=("myeloblast",)),
    )

    assert decision.status == "review_required"
    assert "class_requires_review" in decision.reasons


def test_uncertainty_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_accept_confidence"):
        UncertaintyPolicy(min_accept_confidence=1.2)


def test_uncertainty_rejects_mismatched_classes() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        decide_prediction([1.0], ["basophil", "eosinophil"], UncertaintyPolicy())
