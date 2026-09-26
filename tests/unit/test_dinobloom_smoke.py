from pathlib import Path

import pytest

from bloodfilm.classification.dinobloom import require_dinobloom_b_weights, smoke_dinobloom_b
from bloodfilm.errors import MissingAssetError, ModelLoadError


def test_smoke_reports_blocked_when_weights_missing(tmp_path: Path) -> None:
    result = smoke_dinobloom_b(tmp_path / "dinobloom-b.pth")

    assert result.status == "blocked"
    assert result.embedding_shape is None


def test_smoke_verifies_embedding_shape_through_loader(tmp_path: Path) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")

    result = smoke_dinobloom_b(weights, backbone_loader=lambda path: (1, 768))

    assert result.status == "ok"
    assert result.embedding_shape == [1, 768]
    assert result.blocker is None


def test_smoke_raises_when_loader_fails(tmp_path: Path) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")

    def broken_loader(path: Path) -> tuple[int, int]:
        raise RuntimeError("boom")

    with pytest.raises(ModelLoadError):
        smoke_dinobloom_b(weights, backbone_loader=broken_loader)


def test_require_weights_rejects_missing_and_unverifiable_weights(tmp_path: Path) -> None:
    with pytest.raises(MissingAssetError):
        require_dinobloom_b_weights(tmp_path / "dinobloom-b.pth")

    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"fake-checkpoint")
    with pytest.raises(ModelLoadError):
        require_dinobloom_b_weights(weights, backbone_loader=lambda path: (1, 10))
