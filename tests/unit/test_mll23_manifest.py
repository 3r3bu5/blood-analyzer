from pathlib import Path

import pytest

from bloodfilm.data.manifest import build_mll23_manifest
from bloodfilm.data.splits import create_split_manifest
from bloodfilm.schemas import MLL23_CANONICAL_CLASSES
from tests.helpers.png import write_rgb_png


def test_build_mll23_manifest_maps_valid_images_and_reports_invalid_items(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "mll23"
    (dataset_root / "Basophil").mkdir(parents=True)
    (dataset_root / "Mystery").mkdir(parents=True)
    write_rgb_png(dataset_root / "Basophil" / "cell.png", 2, 2, [(1, 2, 3)] * 4)
    write_rgb_png(dataset_root / "Mystery" / "cell.png", 2, 2, [(4, 5, 6)] * 4)

    mapping_path = tmp_path / "mll23.yaml"
    _write_mll23_mapping(mapping_path)

    result = build_mll23_manifest(dataset_root, mapping_path)

    assert [row.canonical_label for row in result.valid_rows] == ["basophil"]
    assert result.valid_rows[0].width == 2
    assert result.valid_rows[0].image_id == result.valid_rows[0].sha256
    assert len(result.valid_rows[0].image_id) == 64
    assert result.class_distribution["basophil"] == 1
    assert result.class_distribution["normoblast"] == 0
    assert len(result.invalid_rows) == 1
    assert result.invalid_rows[0].reason == "unmapped_label"


def test_build_mll23_manifest_reports_unsupported_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mll23"
    (dataset_root / "Basophil").mkdir(parents=True)
    (dataset_root / "Basophil" / "notes.txt").write_text("not an image", encoding="utf-8")
    mapping_path = tmp_path / "mll23.yaml"
    _write_mll23_mapping(mapping_path)

    result = build_mll23_manifest(dataset_root, mapping_path)

    assert result.valid_rows == []
    assert len(result.invalid_rows) == 1
    assert result.invalid_rows[0].reason == "UNSUPPORTED_IMAGE_FORMAT"


def test_build_mll23_manifest_reports_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_root = tmp_path / "mll23"
    (dataset_root / "Basophil").mkdir(parents=True)
    write_rgb_png(dataset_root / "Basophil" / "cell.png", 2, 2, [(1, 2, 3)] * 4)
    mapping_path = tmp_path / "mll23.yaml"
    _write_mll23_mapping(mapping_path)

    build_mll23_manifest(dataset_root, mapping_path, progress_every=1)

    assert "manifest: scanned 1/1 files" in capsys.readouterr().err


def test_split_manifest_assigns_each_valid_row_once(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mll23"
    (dataset_root / "Basophil").mkdir(parents=True)
    for index in range(5):
        write_rgb_png(dataset_root / "Basophil" / f"cell_{index}.png", 2, 2, [(index, 2, 3)] * 4)

    mapping_path = tmp_path / "mll23.yaml"
    _write_mll23_mapping(mapping_path)

    result = build_mll23_manifest(dataset_root, mapping_path)
    split_rows = create_split_manifest(result.valid_rows, seed=7)

    assert sorted(row.image_path for row in split_rows) == sorted(
        row.image_path for row in result.valid_rows
    )
    assert {row.split for row in split_rows} <= {"train", "validation", "test"}
    assert all(row.patient_or_source_group.startswith("ungrouped:") for row in result.valid_rows)


def _write_mll23_mapping(path: Path) -> None:
    path.write_text(
        "{"
        '"schema_version":1,'
        '"source":"mll23",'
        f'"canonical_classes":{MLL23_CANONICAL_CLASSES!r},'
        '"mappings":{"Basophil":"basophil"}'
        "}".replace("'", '"'),
        encoding="utf-8",
    )
