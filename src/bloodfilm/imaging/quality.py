from __future__ import annotations

from dataclasses import dataclass

from bloodfilm.imaging.io import ImageData
from bloodfilm.schemas import QualityStatus


@dataclass(frozen=True)
class QualityConfig:
    min_width: int = 512
    min_height: int = 512
    min_focus_score: float | None = None
    min_mean_brightness: float | None = None
    max_mean_brightness: float | None = None
    max_dark_fraction: float | None = None
    max_bright_fraction: float | None = None
    action_on_unusable: str = "stop"
    low_pixel_value: int = 10
    high_pixel_value: int = 245


@dataclass(frozen=True)
class ImageQualityResult:
    width: int
    height: int
    focus_score: float
    mean_brightness: float
    dark_fraction: float
    bright_fraction: float
    status: QualityStatus
    reasons: list[str]
    thresholds: dict[str, float | int | None]


def assess_quality(image: ImageData, config: QualityConfig) -> ImageQualityResult:
    width = image.width
    height = image.height
    pixels = image.pixels
    gray = _rgb_to_gray(pixels)
    focus_score = _laplacian_variance(gray, width, height)
    mean_brightness = sum(gray) / len(gray) if gray else 0.0
    dark_fraction = sum(1 for value in gray if value <= config.low_pixel_value) / len(gray)
    bright_fraction = sum(1 for value in gray if value >= config.high_pixel_value) / len(gray)

    unusable: list[str] = []
    review: list[str] = []
    if width < config.min_width:
        unusable.append("width_below_minimum")
    if height < config.min_height:
        unusable.append("height_below_minimum")
    if config.min_focus_score is not None and focus_score < config.min_focus_score:
        unusable.append("focus_below_minimum")
    if config.min_mean_brightness is not None and mean_brightness < config.min_mean_brightness:
        review.append("mean_brightness_below_minimum")
    if config.max_mean_brightness is not None and mean_brightness > config.max_mean_brightness:
        review.append("mean_brightness_above_maximum")
    if config.max_dark_fraction is not None and dark_fraction >= config.max_dark_fraction:
        review.append("dark_fraction_at_limit")
    if config.max_bright_fraction is not None and bright_fraction >= config.max_bright_fraction:
        review.append("bright_fraction_at_limit")

    status: QualityStatus
    if unusable:
        status = "unusable"
        reasons = unusable + review
    elif review:
        status = "review_quality"
        reasons = review
    else:
        status = "acceptable"
        reasons = []

    return ImageQualityResult(
        width=width,
        height=height,
        focus_score=focus_score,
        mean_brightness=mean_brightness,
        dark_fraction=dark_fraction,
        bright_fraction=bright_fraction,
        status=status,
        reasons=reasons,
        thresholds={
            "min_width": config.min_width,
            "min_height": config.min_height,
            "min_focus_score": config.min_focus_score,
            "min_mean_brightness": config.min_mean_brightness,
            "max_mean_brightness": config.max_mean_brightness,
            "max_dark_fraction": config.max_dark_fraction,
            "max_bright_fraction": config.max_bright_fraction,
        },
    )


def _rgb_to_gray(pixels: bytes) -> list[int]:
    if len(pixels) % 3 != 0:
        raise ValueError("RGB pixel buffer length must be divisible by 3")
    gray: list[int] = []
    for index in range(0, len(pixels), 3):
        red, green, blue = pixels[index], pixels[index + 1], pixels[index + 2]
        gray.append(round(0.299 * red + 0.587 * green + 0.114 * blue))
    return gray


def _laplacian_variance(gray: list[int], width: int, height: int) -> float:
    if width < 3 or height < 3:
        return 0.0
    values: list[float] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = gray[y * width + x]
            laplacian = (
                gray[(y - 1) * width + x]
                + gray[(y + 1) * width + x]
                + gray[y * width + x - 1]
                + gray[y * width + x + 1]
                - 4 * center
            )
            values.append(float(laplacian))
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
