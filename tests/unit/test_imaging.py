from pathlib import Path

from bloodfilm.imaging.io import ImageData, load_image
from bloodfilm.imaging.quality import QualityConfig, assess_quality

from tests.helpers.png import write_rgb_png


def test_load_image_reads_png_and_quality_uses_configured_thresholds(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "field.png"
    write_rgb_png(
        image_path,
        width=4,
        height=2,
        pixels=[
            (0, 0, 0),
            (255, 255, 255),
            (100, 100, 100),
            (150, 150, 150),
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
            (40, 40, 40),
        ],
    )

    image = load_image(image_path)
    quality = assess_quality(
        image,
        QualityConfig(min_width=4, min_height=2, max_dark_fraction=0.25),
    )

    assert image.width == 4
    assert image.height == 2
    assert image.mode == "RGB"
    assert quality.width == 4
    assert quality.height == 2
    assert quality.status == "review_quality"
    assert quality.dark_fraction == 0.25
    assert "dark_fraction_at_limit" in quality.reasons


def test_quality_marks_too_small_images_unusable() -> None:
    image = ImageData(
        path=Path("tiny.png"),
        width=1,
        height=1,
        mode="RGB",
        pixels=bytes([128, 128, 128]),
    )

    quality = assess_quality(image, QualityConfig(min_width=2, min_height=2))

    assert quality.status == "unusable"
    assert quality.reasons == ["width_below_minimum", "height_below_minimum"]
