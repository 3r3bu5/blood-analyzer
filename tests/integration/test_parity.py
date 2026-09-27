"""Live-vs-cached parity: skip (never fail) without weights, data or backends."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "models" / "backbones" / "dinobloom-b.pth"
MLL23_ROOT = REPO_ROOT / "data" / "raw" / "MLL23"
CACHE = REPO_ROOT / "data" / "embeddings" / "mll23_embeddings.pt"
CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "mlp.pt"

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("torchvision")

needs_assets = pytest.mark.skipif(
    not (WEIGHTS.exists() and CACHE.exists() and CHECKPOINT.exists() and MLL23_ROOT.exists()),
    reason="parity needs DinoBloom-B weights, MLL23 crops, embeddings cache and mlp checkpoint",
)


@pytest.mark.requires_data
@pytest.mark.requires_model
@pytest.mark.slow
@needs_assets
def test_live_parity_matches_cached_embeddings_and_logits() -> None:
    from bloodfilm.classification.parity import (
        EMBEDDING_COSINE_MIN,
        run_live_parity,
    )

    evidence = run_live_parity(
        weights=WEIGHTS,
        dataset_root=MLL23_ROOT,
        cache_path=CACHE,
        checkpoint_path=CHECKPOINT,
        max_crops=5,
    )

    assert evidence["status"] == "ok"
    assert evidence["crops_checked"] >= 1
    assert evidence["min_cosine_similarity"] >= EMBEDDING_COSINE_MIN
    assert all(row["logits_match"] for row in evidence["crops"])
    assert len({row["label_index"] for row in evidence["crops"]}) == evidence["crops_checked"]
