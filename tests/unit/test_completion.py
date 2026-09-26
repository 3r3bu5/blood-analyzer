from pathlib import Path

import pytest

from bloodfilm.data.completion import (
    dataset_completion_status,
    require_complete_dataset,
    write_completion_record,
)
from bloodfilm.errors import DatasetNotAvailableError


def test_completion_round_trip_marks_dataset_verified(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mll23"
    dataset_dir.mkdir()

    record_path = write_completion_record(
        dataset_dir, dataset="mll23", verified=True, expected_classes=18, file_count=3
    )

    assert record_path.name == ".complete.json"
    status = dataset_completion_status(dataset_dir)
    assert status["complete"] is True
    assert status["verified"] is True
    assert status["expected_classes"] == 18
    require_complete_dataset(dataset_dir, dataset="mll23")


def test_require_missing_completion_points_to_download(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mll23"
    dataset_dir.mkdir()

    assert dataset_completion_status(dataset_dir)["complete"] is False
    with pytest.raises(DatasetNotAvailableError) as exc_info:
        require_complete_dataset(dataset_dir, dataset="mll23")

    assert "assets download" in str(exc_info.value)


def test_require_rejects_unverified_completion_record(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mll23"
    dataset_dir.mkdir()
    record = write_completion_record(dataset_dir, dataset="mll23")
    record.write_text(
        record.read_text(encoding="utf-8").replace('"verified": true', '"verified": false'),
        encoding="utf-8",
    )

    with pytest.raises(DatasetNotAvailableError):
        require_complete_dataset(dataset_dir, dataset="mll23")
