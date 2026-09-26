from __future__ import annotations

from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError, DatasetNotAvailableError

COMPLETION_FILENAME = ".complete.json"


def write_completion_record(
    dataset_dir: Path | str,
    *,
    dataset: str,
    expected_classes: int | None = None,
    expected_images: int | None = None,
    file_count: int | None = None,
) -> Path:
    directory = Path(dataset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": dataset,
        "verified": True,
        "expected_classes": expected_classes,
        "expected_images": expected_images,
        "file_count": file_count,
    }
    record_path = directory / COMPLETION_FILENAME
    write_json(record_path, record)
    return record_path


def dataset_completion_status(dataset_dir: Path | str) -> dict[str, Any]:
    record_path = Path(dataset_dir) / COMPLETION_FILENAME
    if not record_path.exists():
        return {"complete": False, "verified": False, "record_path": str(record_path)}
    try:
        record = load_config_document(record_path)
        verified = bool(record.get("verified", False))
    except ConfigError:
        return {"complete": False, "verified": False, "record_path": str(record_path)}
    return {
        "complete": verified,
        "verified": verified,
        "record_path": str(record_path),
        "dataset": record.get("dataset"),
        "expected_classes": record.get("expected_classes"),
        "expected_images": record.get("expected_images"),
        "file_count": record.get("file_count"),
    }


def require_complete_dataset(dataset_dir: Path | str, *, dataset: str) -> None:
    status = dataset_completion_status(dataset_dir)
    if status["complete"]:
        return
    raise DatasetNotAvailableError(
        f"{dataset} dataset is not installed or not verified.\n"
        f"Expected location: {dataset_dir}\n"
        f"Install it with: bloodfilm assets download {dataset}"
    )
