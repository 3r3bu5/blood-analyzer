from pathlib import Path

import pytest

from bloodfilm.errors import ImageDecodeError, InputNotFoundError, UnsupportedImageFormatError
from bloodfilm.imaging.io import ImageData, load_image, probe_image
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


def test_probe_image_reads_png_dimensions_without_decoding_pixels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "field.png"
    write_rgb_png(image_path, width=4, height=2, pixels=[(9, 9, 9)] * 8)

    metadata = probe_image(image_path)

    assert (metadata.width, metadata.height, metadata.mode) == (4, 2, "RGB")


def test_probe_image_rejects_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        probe_image(tmp_path / "absent.png")
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    with pytest.raises(UnsupportedImageFormatError):
        probe_image(notes)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\nshort")
    with pytest.raises(ImageDecodeError):
        probe_image(broken)


def test_probe_image_reads_tiff_header(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image", reason="TIFF probe needs Pillow")
    image_path = tmp_path / "cell.tif"
    pil.new("L", (6, 3), color=200).save(image_path)

    metadata = probe_image(image_path)

    assert (metadata.width, metadata.height, metadata.mode) == (6, 3, "L")


def test_png_extension_with_jpeg_payload_falls_back_to_pillow(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image", reason="mislabelled image fallback needs Pillow")
    image_path = tmp_path / "field.png"
    pil.new("RGB", (5, 7), color=(10, 20, 30)).save(image_path, format="JPEG")

    metadata = probe_image(image_path)
    image = load_image(image_path)

    assert (metadata.width, metadata.height, metadata.mode) == (5, 7, "RGB")
    assert (image.width, image.height, image.mode) == (5, 7, "RGB")


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
