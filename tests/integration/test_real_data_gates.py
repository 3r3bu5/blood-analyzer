"""Dataset-gated integration tests: skip (never fail) without real assets."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MLL23_ROOT = REPO_ROOT / "data" / "raw" / "MLL23"

needs_mll23 = pytest.mark.skipif(not MLL23_ROOT.exists(), reason="MLL23 is not installed")


@pytest.mark.requires_data
@needs_mll23
def test_real_mll23_manifest_builds() -> None:
    from bloodfilm.data.manifest import build_mll23_manifest

    result = build_mll23_manifest(MLL23_ROOT, REPO_ROOT / "configs" / "mappings" / "mll23.yaml")

    assert result.class_distribution["basophil"] > 0
    assert len(result.valid_rows) == sum(result.class_distribution.values())


@pytest.mark.requires_model
@pytest.mark.skipif(
    not (REPO_ROOT / "models" / "backbones" / "dinobloom-b.pth").exists(),
    reason="DinoBloom-B weights are not installed",
)
def test_real_dinobloom_smoke_produces_768_vector() -> None:
    pytest.skip("Backbone wiring belongs to the M2 classifier baseline")
