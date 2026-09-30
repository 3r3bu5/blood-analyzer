from pathlib import Path

from bloodfilm.config import load_config


def test_load_config_reads_project_and_quality_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        '{"project":{"seed":123,"run_root":"outputs/runs"},'
        '"quality":{"min_width":16,"min_height":8,"max_dark_fraction":0.25},'
        '"imaging":{"crop_padding_ratio":0.2}}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.seed == 123
    assert config.project.run_root == Path("outputs/runs")
    assert config.quality.min_width == 16
    assert config.quality.min_height == 8
    assert config.quality.max_dark_fraction == 0.25
    assert config.imaging.crop_padding_ratio == 0.2


def test_load_config_parses_false_boolean_strings_and_detector_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        '{"imaging":{"save_crops":"false"},'
        '"detector":{"implementation":"ultralytics_yolo","weights":"models/detector/wbc.pt"}}',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.imaging.save_crops is False
    assert config.detector.implementation == "ultralytics_yolo"
    assert config.detector.weights == Path("models/detector/wbc.pt")


def test_load_config_reads_dense_field_tiling_profile() -> None:
    config = load_config(Path("configs/inference_dense_field_recall_v0_1.yaml"))

    assert config.detector_profile == "dense_field_recall_v0.1"
    assert config.detector.confidence_threshold == 0.15
    assert config.detector.iou_threshold == 0.60
    assert config.tiling.enabled is True
    assert config.tiling.tile_size == 384
    assert config.tiling.overlap_ratio == 0.20
    assert config.tiling.tile_core_ownership is True
    assert config.tiling.containment_deduplication is True


def test_load_config_reads_conservative_tiling_profile() -> None:
    config = load_config(Path("configs/inference_conservative_tiled.yaml"))

    assert config.detector_profile == "dense_field_conservative_v0.1"
    assert config.tiling.tile_size == 512
