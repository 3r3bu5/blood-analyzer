from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bloodfilm.errors import ConfigError


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "wbc_candidate"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "score": self.score,
            "label": self.label,
        }


class CandidateCellDetector(Protocol):
    version: str

    def detect(self, image_rgb: object) -> list[Detection]: ...


def validate_detections(
    detections: list[Detection], *, image_width: int, image_height: int
) -> list[Detection]:
    for detection in detections:
        if detection.x1 < 0 or detection.y1 < 0:
            raise ConfigError("Detection box coordinates must be non-negative")
        if detection.x2 > image_width or detection.y2 > image_height:
            raise ConfigError("Detection box exceeds image bounds")
        if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
            raise ConfigError("Detection box must have positive area")
        if not 0.0 <= detection.score <= 1.0:
            raise ConfigError("Detection score must be between 0 and 1")
    return detections
