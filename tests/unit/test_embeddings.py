"""Unit tests for Stage A embedding-cache extraction (TDD 9.4)."""

from pathlib import Path
from typing import Any

import pytest

from bloodfilm.classification.embeddings import (
    EmbeddingCache,
    build_embedding_cache,
    load_embedding_cache,
    preprocessing_checksum,
    save_embedding_cache,
)
from bloodfilm.schemas import MLL23_CANONICAL_CLASSES, ManifestRow, SplitRow

torch = pytest.importorskip("torch")


def _split_row(image_id: str, label: str, split: str = "train") -> SplitRow:
    return SplitRow(
        image_id=image_id,
        image_path=f"{label}/{image_id}.png",
        canonical_label=label,
        patient_or_source_group=f"ungrouped:{image_id}",
        split=split,  # type: ignore[arg-type]
    )


def _backbone(batch: Any) -> Any:
    return torch.full((1, 768), float(batch[0, 0, 0, 0]))


def _preprocess(path: Any) -> Any:
    return torch.zeros(1, 3, 224, 224)


def test_build_cache_encodes_labels_and_preserves_splits(tmp_path: Path) -> None:
    rows = [
        _split_row("a", "basophil", "train"),
        _split_row("b", "monocyte", "validation"),
        _split_row("c", "myeloblast", "test"),
    ]

    cache = build_embedding_cache(
        rows,
        dataset_root=tmp_path,
        backbone=_backbone,
        preprocess=_preprocess,
        weight_sha256="abc123",
    )

    assert isinstance(cache, EmbeddingCache)
    assert [entry.status for entry in cache.entries] == ["ok", "ok", "ok"]
    assert [entry.label_index for entry in cache.entries] == [
        MLL23_CANONICAL_CLASSES.index("basophil"),
        MLL23_CANONICAL_CLASSES.index("monocyte"),
        MLL23_CANONICAL_CLASSES.index("myeloblast"),
    ]
    assert [entry.split for entry in cache.entries] == ["train", "validation", "test"]
    assert cache.embeddings.shape == (3, 768)
    assert cache.metadata["weight_sha256"] == "abc123"
    assert len(cache.metadata["preprocessing_sha256"]) == 64


def test_build_cache_respects_limit(tmp_path: Path) -> None:
    rows = [_split_row(str(index), "basophil") for index in range(5)]

    cache = build_embedding_cache(
        rows,
        dataset_root=tmp_path,
        backbone=_backbone,
        preprocess=_preprocess,
        weight_sha256="abc123",
        limit=2,
    )

    assert len(cache.entries) == 2
    assert cache.embeddings.shape == (2, 768)


def test_build_cache_records_unreadable_images_and_continues(tmp_path: Path) -> None:
    rows = [_split_row("good", "basophil"), _split_row("bad", "monocyte")]

    def flaky_preprocess(path: Any) -> Any:
        if "bad" in str(path):
            raise RuntimeError("unreadable test image")
        return torch.zeros(1, 3, 224, 224)

    cache = build_embedding_cache(
        rows,
        dataset_root=tmp_path,
        backbone=_backbone,
        preprocess=flaky_preprocess,
        weight_sha256="abc123",
    )

    assert [entry.status for entry in cache.entries] == ["ok", "skipped"]
    assert "unreadable" in cache.entries[1].reason.lower()
    assert cache.embeddings.shape == (1, 768)


def test_preprocessing_checksum_is_stable() -> None:
    assert preprocessing_checksum() == preprocessing_checksum()
    assert len(preprocessing_checksum()) == 64


def test_save_and_load_cache_roundtrip(tmp_path: Path) -> None:
    rows = [_split_row("a", "basophil"), _split_row("b", "eosinophil")]
    cache = build_embedding_cache(
        rows,
        dataset_root=tmp_path,
        backbone=_backbone,
        preprocess=_preprocess,
        weight_sha256="abc123",
    )
    path = tmp_path / "embeddings.pt"

    save_embedding_cache(cache, path)
    reloaded = load_embedding_cache(path)

    assert torch.equal(reloaded.embeddings, cache.embeddings)
    assert [(e.image_id, e.label_index, e.split) for e in reloaded.entries] == [
        (e.image_id, e.label_index, e.split) for e in cache.entries
    ]
    assert reloaded.metadata == cache.metadata


def test_manifest_row_converts_to_cache_input() -> None:
    row = ManifestRow(
        image_id="abc",
        image_path="basophil/basophil/basophil_0001.TIF",
        source_folder="basophil",
        canonical_label="basophil",
        sha256="abc",
        width=288,
        height=288,
        mode="RGB",
        patient_or_source_group="ungrouped:abc",
    )
    assert MLL23_CANONICAL_CLASSES.index(row.canonical_label) == 0


def test_build_cache_rejects_unsupported_device(tmp_path: Path) -> None:
    from bloodfilm.errors import ConfigError

    with pytest.raises(ConfigError, match="device"):
        build_embedding_cache(
            [],
            dataset_root=tmp_path,
            backbone=_backbone,
            preprocess=_preprocess,
            weight_sha256="abc123",
            device="tpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_build_cache_extracts_on_cuda(tmp_path: Path) -> None:
    def cuda_backbone(batch: object) -> object:
        assert isinstance(batch, torch.Tensor) and batch.is_cuda
        return torch.full((1, 768), 1.0, device="cuda")

    rows = [_split_row("a", "basophil")]
    cache = build_embedding_cache(
        rows,
        dataset_root=tmp_path,
        backbone=cuda_backbone,
        preprocess=_preprocess,
        weight_sha256="abc123",
        device="cuda",
    )

    assert cache.entries[0].status == "ok"
    assert not cache.embeddings.is_cuda
    assert cache.embeddings.shape == (1, 768)
