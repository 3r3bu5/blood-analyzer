"""Real-backbone integration tests: skip (never fail) without weights, data or backends."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "models" / "backbones" / "dinobloom-b.pth"
MLL23_ROOT = REPO_ROOT / "data" / "raw" / "MLL23"

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("torchvision")

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="DinoBloom-B weights missing")


def _sample_crop() -> Path:
    crops = sorted((MLL23_ROOT / "basophil" / "basophil").glob("*.TIF"))
    if not crops:
        pytest.skip("no MLL23 basophil crops available")
    return crops[0]


@pytest.mark.requires_model
@pytest.mark.requires_data
@needs_weights
def test_real_backbone_produces_expected_embedding() -> None:
    from bloodfilm.classification.dinobloom import (
        extract_backbone_embedding,
        load_dinobloom_b_backbone,
        preprocess_crop,
    )

    model = load_dinobloom_b_backbone(WEIGHTS)
    batch = preprocess_crop(_sample_crop())

    embedding = extract_backbone_embedding(model, batch)

    assert list(embedding.shape) == [1, 768]
    assert bool(torch.isfinite(embedding).all())


@pytest.mark.requires_model
@pytest.mark.requires_data
@needs_weights
def test_real_backbone_is_deterministic() -> None:
    from bloodfilm.classification.dinobloom import (
        extract_backbone_embedding,
        load_dinobloom_b_backbone,
        preprocess_crop,
    )

    model = load_dinobloom_b_backbone(WEIGHTS)
    batch = preprocess_crop(_sample_crop())

    first = extract_backbone_embedding(model, batch)
    second = extract_backbone_embedding(model, batch)

    assert torch.equal(first, second)
