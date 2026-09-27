from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bloodfilm.imaging.io import load_image


@dataclass(frozen=True)
class FieldROI:
    x1: int
    y1: int
    x2: int
    y2: int
    method: str
    confidence: float | None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "method": self.method,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }


def detect_field_roi(
    image_path: Path | str,
    *,
    dark_threshold: int = 20,
    minimum_area_ratio: float = 0.30,
    padding_ratio: float = 0.02,
    fallback_to_full_image: bool = True,
) -> FieldROI:
    """Detect a conservative illuminated-field ROI or safely return the full image."""
    image = load_image(image_path)
    lit_pixels: list[tuple[int, int]] = []
    for y in range(image.height):
        for x in range(image.width):
            offset = (y * image.width + x) * image.channels
            channels = image.pixels[offset : offset + image.channels]
            if max(channels) > dark_threshold:
                lit_pixels.append((x, y))
    image_area = image.width * image.height
    lit_area = len(lit_pixels)
    if not lit_pixels or lit_area / image_area < minimum_area_ratio:
        return _full_image_roi(
            image.width, image.height, "lit_area_below_minimum", fallback_to_full_image
        )
    x_values = [x for x, _ in lit_pixels]
    y_values = [y for _, y in lit_pixels]
    x1, x2 = min(x_values), max(x_values) + 1
    y1, y2 = min(y_values), max(y_values) + 1
    if x1 == 0 and y1 == 0 and x2 == image.width and y2 == image.height:
        return FieldROI(0, 0, image.width, image.height, "full_image", 1.0)
    pad = round(max(x2 - x1, y2 - y1) * padding_ratio)
    return FieldROI(
        x1=max(0, x1 - pad),
        y1=max(0, y1 - pad),
        x2=min(image.width, x2 + pad),
        y2=min(image.height, y2 + pad),
        method="auto_illuminated_region",
        confidence=round(lit_area / image_area, 6),
    )


def _full_image_roi(width: int, height: int, reason: str, fallback_to_full_image: bool) -> FieldROI:
    if not fallback_to_full_image:
        return FieldROI(0, 0, width, height, "unresolved", None, reason)
    return FieldROI(0, 0, width, height, "full_image", 1.0, reason)
