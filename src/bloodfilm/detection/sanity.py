from __future__ import annotations

from dataclasses import dataclass

from bloodfilm.detection.detector import Detection


@dataclass(frozen=True)
class BoxSanityResult:
    status: str
    reasons: list[str]


def evaluate_box_sanity(
    detection: Detection,
    *,
    image_width: int,
    image_height: int,
    maximum_area_ratio: float = 0.35,
    minimum_area_ratio: float = 0.0001,
    maximum_aspect_ratio: float = 4.0,
) -> BoxSanityResult:
    reasons: list[str] = []
    width = detection.x2 - detection.x1
    height = detection.y2 - detection.y1
    if width <= 0 or height <= 0:
        reasons.append("zero_area_box")
    if (
        detection.x1 < 0
        or detection.y1 < 0
        or detection.x2 > image_width
        or detection.y2 > image_height
    ):
        reasons.append("box_outside_image")
    image_area = image_width * image_height
    area_ratio = (width * height / image_area) if image_area and width > 0 and height > 0 else 0.0
    if area_ratio > maximum_area_ratio:
        reasons.append("field_sized_box")
    if 0 < area_ratio < minimum_area_ratio:
        reasons.append("tiny_box")
    if width > 0 and height > 0:
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > maximum_aspect_ratio:
            reasons.append("extreme_aspect_ratio")
    return BoxSanityResult(status="review" if reasons else "accepted", reasons=reasons)
