"""Unit tests for the real DinoBloom-B backbone adapter (TDD 9.1/9.2)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bloodfilm.classification.dinobloom import (
    EXPECTED_EMBEDDING_SHAPE,
    PREPROCESSING,
    TIMM_MODEL_ID,
    extract_backbone_embedding,
    load_dinobloom_b_backbone,
    preprocess_crop,
    smoke_dinobloom_b,
)
from bloodfilm.errors import ModelLoadError
from tests.helpers.png import write_rgb_png


def test_preprocessing_record_matches_official_weight_recipe() -> None:
    assert TIMM_MODEL_ID == "vit_base_patch14_dinov2"
    assert PREPROCESSING == {
        "resize": [224, 224],
        "resize_method": "direct",
        "crop_method": "none",
        "input_size": 224,
        "interpolation": "bilinear",
        "rgb_conversion": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "colour_normalization": "none",
    }


def test_load_backbone_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        load_dinobloom_b_backbone(tmp_path / "absent.pth")


def test_load_backbone_rejects_unsupported_device(tmp_path: Path) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")

    with pytest.raises(ModelLoadError, match="device"):
        load_dinobloom_b_backbone(weights, device="tpu")


def test_load_backbone_requires_timm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")
    monkeypatch.setitem(sys.modules, "timm", None)

    with pytest.raises(ModelLoadError, match="timm"):
        load_dinobloom_b_backbone(weights)


def test_extract_embedding_asserts_expected_shape() -> None:
    pytest.importorskip("torch")
    embedding = extract_backbone_embedding(_StubModel((1, 768)), object())

    assert list(embedding.shape) == list(EXPECTED_EMBEDDING_SHAPE)


def test_extract_embedding_rejects_wrong_shape() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ModelLoadError, match="1, 768"):
        extract_backbone_embedding(_StubModel((1, 10)), object())


def test_preprocess_crop_requires_pil_and_torchvision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "cell.png"
    write_rgb_png(image_path, 2, 2, [(1, 2, 3)] * 4)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)

    with pytest.raises(ModelLoadError):
        preprocess_crop(image_path)


def test_smoke_with_sample_image_needs_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")
    image_path = tmp_path / "cell.png"
    write_rgb_png(image_path, 2, 2, [(1, 2, 3)] * 4)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "timm", None)

    with pytest.raises(ModelLoadError):
        smoke_dinobloom_b(weights, sample_image=image_path)


class _StubModel:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = shape

    def __call__(self, batch: Any) -> Any:
        return SimpleNamespace(shape=self._shape)
