"""Live-vs-cached parity: skip (never fail) without weights, data or backends."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "models" / "backbones" / "dinobloom-b.pth"
MLL23_ROOT = REPO_ROOT / "data" / "raw" / "MLL23"
CACHE = REPO_ROOT / "data" / "embeddings" / "mll23_embeddings.pt"
CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "mlp.pt"

torch = pytest.importorskip("torch")
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
def test_live_embeddings_match_cached_embeddings() -> None:
    from bloodfilm.classification.dinobloom import (
        extract_backbone_embedding,
        load_dinobloom_b_backbone,
        preprocess_crop,
    )
    from bloodfilm.classification.embeddings import load_embedding_cache

    cache = load_embedding_cache(CACHE)
    ok_entries = [entry for entry in cache.entries if entry.status == "ok"]
    assert tuple(cache.embeddings.shape) == (len(ok_entries), 768)

    seen_classes: set[int] = set()
    selected: list[tuple[int, object]] = []
    for position, entry in enumerate(ok_entries):
        if entry.label_index not in seen_classes:
            seen_classes.add(entry.label_index)
            selected.append((position, entry))
        if len(selected) >= 5:
            break
    assert selected, "no cached ok entries available for parity"

    model = load_dinobloom_b_backbone(WEIGHTS)
    for position, entry in selected:
        image = MLL23_ROOT / entry.image_path
        if not image.exists():
            pytest.skip(f"parity crop missing: {image}")
        live = extract_backbone_embedding(model, preprocess_crop(image))[0]
        cached = cache.embeddings[position]
        cosine = torch.nn.functional.cosine_similarity(
            live.flatten(), cached.flatten(), dim=0
        ).item()

        assert cosine >= 0.9999


@pytest.mark.requires_data
@pytest.mark.requires_model
@pytest.mark.slow
@needs_assets
def test_live_mlp_logits_match_cached_embedding_logits() -> None:
    from bloodfilm.classification.dinobloom import (
        extract_backbone_embedding,
        load_dinobloom_b_backbone,
        preprocess_crop,
    )
    from bloodfilm.classification.embeddings import load_embedding_cache
    from bloodfilm.classification.heads import build_head

    cache = load_embedding_cache(CACHE)
    ok_entries = [entry for entry in cache.entries if entry.status == "ok"]
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    head = build_head(str(checkpoint["head"]), len(checkpoint["class_names"]))
    head.load_state_dict(checkpoint["model_state"])
    head.eval()

    model = load_dinobloom_b_backbone(WEIGHTS)
    checked = 0
    for position, entry in enumerate(ok_entries[:50]):
        image = MLL23_ROOT / entry.image_path
        if not image.exists():
            continue
        live_embedding = extract_backbone_embedding(model, preprocess_crop(image))
        with torch.no_grad():
            live_logits = head(live_embedding)[0]
            cached_logits = head(cache.embeddings[position].unsqueeze(0))[0]

        assert torch.allclose(live_logits, cached_logits, atol=1e-4, rtol=1e-4)
        checked += 1
        if checked >= 5:
            break
    assert checked > 0, "no parity crops with images on disk"
