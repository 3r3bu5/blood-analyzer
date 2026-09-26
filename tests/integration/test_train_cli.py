import json
from pathlib import Path

import pytest

from bloodfilm.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(path: Path, dataset_root: Path, weights: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": {"mll23_root": str(dataset_root)},
                "classifier": {"weights": str(weights)},
            }
        ),
        encoding="utf-8",
    )


def test_config_validate_works_without_data(tmp_path: Path, capsys: object) -> None:
    exit_code = main(["config", "validate", "--config", str(REPO_ROOT / "configs/base.yaml")])

    assert exit_code == 0


def test_train_classifier_requires_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "absent-mll23", tmp_path / "dinobloom-b.pth")

    exit_code = main(["train", "classifier", "--config", str(config_path)])

    assert exit_code == 2
    assert "assets download" in capsys.readouterr().err


def test_train_classifier_requires_completion_record(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mll23"
    dataset_root.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, dataset_root, tmp_path / "dinobloom-b.pth")

    assert main(["train", "classifier", "--config", str(config_path)]) == 2


def test_train_classifier_blocked_without_backbone(tmp_path: Path) -> None:
    from bloodfilm.data.completion import write_completion_record

    dataset_root = tmp_path / "mll23"
    dataset_root.mkdir()
    write_completion_record(dataset_root, dataset="mll23")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, dataset_root, tmp_path / "absent-weights.pth")

    assert main(["train", "classifier", "--config", str(config_path)]) == 2
