from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

PredictionStatus = Literal["accepted", "review_required", "unknown"]


@dataclass(frozen=True)
class UncertaintyPolicy:
    min_accept_confidence: float = 0.80
    min_accept_margin: float = 0.15
    max_accept_entropy: float | None = None
    unknown_below_confidence: float = 0.35
    require_review_for_classes: tuple[str, ...] = (
        "myeloblast",
        "promyelocyte_atypical",
        "lymphocyte_neoplastic_other",
    )

    def __post_init__(self) -> None:
        _validate_probability_threshold(self.min_accept_confidence, "min_accept_confidence")
        _validate_probability_threshold(self.min_accept_margin, "min_accept_margin")
        _validate_probability_threshold(self.unknown_below_confidence, "unknown_below_confidence")
        if self.max_accept_entropy is not None and self.max_accept_entropy < 0.0:
            raise ValueError("max_accept_entropy must be non-negative")
        if self.unknown_below_confidence > self.min_accept_confidence:
            raise ValueError("unknown_below_confidence must not exceed min_accept_confidence")
        if not isinstance(self.require_review_for_classes, tuple):
            object.__setattr__(
                self, "require_review_for_classes", tuple(self.require_review_for_classes)
            )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionDecision:
    status: PredictionStatus
    label: str | None
    class_index: int | None
    confidence: float
    margin: float
    entropy: float
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide_prediction(
    probabilities: list[float],
    class_names: list[str],
    policy: UncertaintyPolicy,
) -> PredictionDecision:
    if len(probabilities) != len(class_names):
        raise ValueError("probabilities and class_names must have the same length")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(value < 0.0 for value in probabilities):
        raise ValueError("probabilities must be non-negative")
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("probabilities must be finite")

    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    class_index, confidence = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = confidence - runner_up
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)

    if confidence < policy.unknown_below_confidence:
        return PredictionDecision(
            status="unknown",
            label=None,
            class_index=None,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            reasons=["confidence_below_unknown_threshold"],
        )

    reasons: list[str] = []
    if confidence < policy.min_accept_confidence:
        reasons.append("confidence_below_accept_threshold")
    if margin < policy.min_accept_margin:
        reasons.append("margin_below_minimum")
    if policy.max_accept_entropy is not None and entropy > policy.max_accept_entropy:
        reasons.append("entropy_above_maximum")
    if class_names[class_index] in policy.require_review_for_classes:
        reasons.append("class_requires_review")

    return PredictionDecision(
        status="review_required" if reasons else "accepted",
        label=class_names[class_index],
        class_index=class_index,
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        reasons=reasons,
    )


def _validate_probability_threshold(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
