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


def test_config_validate_works_without_data(tmp_path: Path) -> None:
    exit_code = main(["config", "validate", "--config", str(REPO_ROOT / "configs/base.yaml")])

    assert exit_code == 0


def test_config_validate_rejects_unknown_head(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({"classifier": {"head": "transformer", "num_classes": 18}}),
        encoding="utf-8",
    )

    assert main(["config", "validate", "--config", str(config_path)]) == 2


def test_config_validate_rejects_non_positive_classes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"classifier": {"num_classes": 0}}), encoding="utf-8")

    assert main(["config", "validate", "--config", str(config_path)]) == 2


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


def _write_verified_dataset(dataset_root: Path) -> None:
    from bloodfilm.data.completion import write_completion_record

    dataset_root.mkdir(parents=True, exist_ok=True)
    write_completion_record(dataset_root, dataset="mll23", verified=True)


def _write_tiny_weights(path: Path) -> None:
    torch = pytest.importorskip("torch")

    torch.save({"probe": torch.tensor([1.0])}, path)


def _write_synthetic_cache(path: Path, weight_sha256: str) -> None:
    pytest.importorskip("torch")

    from bloodfilm.classification.embeddings import (
        CacheEntry,
        EmbeddingCache,
        save_embedding_cache,
    )
    from tests.helpers.fake_backbone import fake_embedding_bank

    embeddings, labels = fake_embedding_bank(num_classes=2, per_class=12, seed=3)
    entries = [
        CacheEntry(
            image_id=f"cell-{index}",
            image_path=f"class/cell-{index}.png",
            label_index=label,
            split="train" if index % 2 == 0 else "validation",
            status="ok",
        )
        for index, label in enumerate(labels)
    ]
    cache = EmbeddingCache(
        entries=entries,
        embeddings=embeddings,
        metadata={"weight_sha256": weight_sha256, "preprocessing_sha256": "0" * 64},
    )
    save_embedding_cache(cache, path)


def _write_train_config(path: Path, dataset_root: Path, weights: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": {"mll23_root": str(dataset_root)},
                "classifier": {
                    "weights": str(weights),
                    "head": "linear",
                    "num_classes": 2,
                },
            }
        ),
        encoding="utf-8",
    )


def test_train_classifier_trains_and_calibrates_from_cache(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from bloodfilm.util import sha256_file

    dataset_root = tmp_path / "mll23"
    _write_verified_dataset(dataset_root)
    weights = tmp_path / "dinobloom-b.pth"
    _write_tiny_weights(weights)
    cache_path = tmp_path / "embeddings.pt"
    _write_synthetic_cache(cache_path, sha256_file(weights))
    config_path = tmp_path / "config.yaml"
    _write_train_config(config_path, dataset_root, weights)
    checkpoint_dir = tmp_path / "checkpoints"
    report_path = tmp_path / "comparison.json"

    exit_code = main(
        [
            "train",
            "classifier",
            "--config",
            str(config_path),
            "--embeddings",
            str(cache_path),
            "--epochs",
            "50",
            "--learning-rate",
            "0.05",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--report-output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selected"] == "linear"
    assert report["heads"]["linear"]["macro_f1"] > 0.9
    assert report["heads"]["linear"]["temperature"] > 0
    assert (checkpoint_dir / "linear.pt").exists()


def test_train_classifier_rejects_stale_cache(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_root = tmp_path / "mll23"
    _write_verified_dataset(dataset_root)
    weights = tmp_path / "dinobloom-b.pth"
    _write_tiny_weights(weights)
    cache_path = tmp_path / "embeddings.pt"
    _write_synthetic_cache(cache_path, "0" * 64)
    config_path = tmp_path / "config.yaml"
    _write_train_config(config_path, dataset_root, weights)

    exit_code = main(
        ["train", "classifier", "--config", str(config_path), "--embeddings", str(cache_path)]
    )

    assert exit_code == 2
