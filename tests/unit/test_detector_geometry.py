from pathlib import Path

from bloodfilm.detection.detector import Detection
from tests.helpers.png import write_rgb_png


def test_circular_roi_detects_illuminated_region(tmp_path: Path) -> None:
    from bloodfilm.detection.roi import detect_field_roi

    image = tmp_path / "circle.png"
    pixels = []
    for y in range(100):
        for x in range(100):
            inside = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
            pixels.append((230, 230, 230) if inside else (0, 0, 0))
    write_rgb_png(image, 100, 100, pixels)

    roi = detect_field_roi(image, dark_threshold=20, minimum_area_ratio=0.20, padding_ratio=0.0)

    assert roi.method == "auto_illuminated_region"
    assert 18 <= roi.x1 <= 22
    assert 78 <= roi.x2 <= 82
    assert 18 <= roi.y1 <= 22
    assert 78 <= roi.y2 <= 82


def test_rectangular_roi_falls_back_to_full_image(tmp_path: Path) -> None:
    from bloodfilm.detection.roi import detect_field_roi

    image = tmp_path / "rect.png"
    write_rgb_png(image, 40, 30, [(220, 220, 220)] * (40 * 30))

    roi = detect_field_roi(image)

    assert roi.to_dict() == {
        "x1": 0,
        "y1": 0,
        "x2": 40,
        "y2": 30,
        "method": "full_image",
        "confidence": 1.0,
        "fallback_reason": None,
    }


def test_generate_tiles_covers_small_and_large_images() -> None:
    from bloodfilm.detection.tiling import generate_tiles

    small = generate_tiles(200, 100, tile_size=512, overlap_ratio=0.2)
    assert [(tile.x1, tile.y1, tile.x2, tile.y2) for tile in small] == [(0, 0, 200, 100)]

    large = generate_tiles(1000, 800, tile_size=512, overlap_ratio=0.25)
    assert large[0].to_dict()["x1"] == 0
    assert max(tile.x2 for tile in large) == 1000
    assert max(tile.y2 for tile in large) == 800
    assert all(tile.x2 > tile.x1 and tile.y2 > tile.y1 for tile in large)


def test_tile_detection_mapping_clips_to_image_bounds() -> None:
    from bloodfilm.detection.tiling import Tile, map_tile_detection

    mapped = map_tile_detection(
        Detection(-5, 10, 50, 70, 0.9),
        Tile(100, 200, 300, 400, "tile-1"),
        image_width=140,
        image_height=260,
    )

    assert (mapped.x1, mapped.y1, mapped.x2, mapped.y2) == (95, 210, 140, 260)


def test_global_nms_merges_duplicate_boxes_deterministically() -> None:
    from bloodfilm.detection.tiling import global_nms

    detections = [
        Detection(0, 0, 10, 10, 0.8),
        Detection(1, 1, 11, 11, 0.9),
        Detection(30, 30, 40, 40, 0.7),
    ]

    kept = global_nms(detections, iou_threshold=0.5)

    assert [round(row.score, 2) for row in kept] == [0.9, 0.7]
    assert [(row.x1, row.y1) for row in kept] == [(1, 1), (30, 30)]


def test_box_sanity_flags_field_sized_box() -> None:
    from bloodfilm.detection.sanity import evaluate_box_sanity

    result = evaluate_box_sanity(
        Detection(0, 0, 90, 90, 0.8),
        image_width=100,
        image_height=100,
        maximum_area_ratio=0.35,
    )

    assert result.status == "review"
    assert "field_sized_box" in result.reasons


def test_target_hash_leakage_prevention(tmp_path: Path) -> None:
    from bloodfilm.detection.manifest import assert_no_locked_target_leakage
    from bloodfilm.util import sha256_file

    image = tmp_path / "locked.png"
    write_rgb_png(image, 2, 2, [(1, 2, 3)] * 4)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,image_sha256,unified_split,locked_for_validation\n"
        f"locked,{image},{sha256_file(image)},train,false\n",
        encoding="utf-8",
    )

    violations = assert_no_locked_target_leakage(manifest, locked_hashes={sha256_file(image)})

    assert violations == ["locked"]
