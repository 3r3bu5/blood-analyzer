from pathlib import Path

from tests.helpers.png import write_rgb_png


def test_mask_to_box_combines_masks_with_padding_and_clipping(tmp_path: Path) -> None:
    from bloodfilm.detection.lisc import mask_to_padded_box

    nucleus = tmp_path / "nucleus.png"
    cytoplasm = tmp_path / "cytoplasm.png"
    pixels_nucleus = []
    pixels_cytoplasm = []
    for y in range(20):
        for x in range(20):
            pixels_nucleus.append((255, 255, 255) if 8 <= x <= 10 and 8 <= y <= 10 else (0, 0, 0))
            pixels_cytoplasm.append((255, 255, 255) if 5 <= x <= 15 and 5 <= y <= 15 else (0, 0, 0))
    write_rgb_png(nucleus, 20, 20, pixels_nucleus)
    write_rgb_png(cytoplasm, 20, 20, pixels_cytoplasm)

    box = mask_to_padded_box(
        [nucleus, cytoplasm], image_width=20, image_height=20, padding_ratio=0.10
    )

    assert box.to_dict() == {"x1": 4.0, "y1": 4.0, "x2": 17.0, "y2": 17.0}


def test_mask_to_box_returns_none_for_empty_masks(tmp_path: Path) -> None:
    from bloodfilm.detection.lisc import mask_to_padded_box

    mask = tmp_path / "empty.png"
    write_rgb_png(mask, 5, 5, [(0, 0, 0)] * 25)

    assert mask_to_padded_box([mask], image_width=5, image_height=5, padding_ratio=0.1) is None
