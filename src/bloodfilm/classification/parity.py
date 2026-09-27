from __future__ import annotations

from pathlib import Path
from typing import Any

from bloodfilm.errors import ConfigError, InputNotFoundError

EMBEDDING_COSINE_MIN = 0.9999
LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4
LOGITS_TOLERANCE_LABEL = "1e-4"
DEFAULT_PARITY_CROPS = 18


def run_live_parity(
    *,
    weights: Path | str,
    dataset_root: Path | str,
    cache_path: Path | str,
    checkpoint_path: Path | str,
    max_crops: int = DEFAULT_PARITY_CROPS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare live backbone/head outputs against cached embeddings and logits.

    Checks up to ``max_crops`` crops from distinct classes: live embedding
    cosine similarity against the cached row must reach ``EMBEDDING_COSINE_MIN``
    and MLP logits must match within ``LOGITS_ATOL``/``LOGITS_RTOL``.
    """
    from bloodfilm.classification.dinobloom import (
        extract_backbone_embedding,
        load_dinobloom_b_backbone,
        preprocess_crop,
    )
    from bloodfilm.classification.embeddings import load_embedding_cache
    from bloodfilm.classification.heads import build_head
    from bloodfilm.ml import require_torch

    weights_path = Path(weights)
    dataset_path = Path(dataset_root)
    cache_file = Path(cache_path)
    checkpoint_file = Path(checkpoint_path)
    if not weights_path.exists():
        raise InputNotFoundError(f"DinoBloom-B weights not found: {weights_path}")
    if not dataset_path.exists():
        raise InputNotFoundError(f"Dataset root not found: {dataset_path}")
    if not cache_file.exists():
        raise InputNotFoundError(f"Embedding cache not found: {cache_file}")
    if not checkpoint_file.exists():
        raise InputNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if max_crops < 1:
        raise ConfigError(f"max_crops must be positive, got {max_crops}")

    torch = require_torch("Parity check")
    cache = load_embedding_cache(cache_file)
    ok_entries = [entry for entry in cache.entries if entry.status == "ok"]
    if tuple(cache.embeddings.shape) != (len(ok_entries), 768):
        raise ConfigError("Embedding cache shape does not match ok entries")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    head = build_head(str(checkpoint["head"]), len(checkpoint["class_names"]))
    head.load_state_dict(checkpoint["model_state"])
    head.eval()
    head = head.to(device)

    seen_classes: set[int] = set()
    selected: list[tuple[int, Any]] = []
    for position, entry in enumerate(ok_entries):
        if entry.label_index not in seen_classes:
            seen_classes.add(entry.label_index)
            selected.append((position, entry))
        if len(selected) >= max_crops:
            break
    if not selected:
        raise InputNotFoundError("Embedding cache has no usable rows for parity")

    model = load_dinobloom_b_backbone(weights_path, device=device)
    crops: list[dict[str, Any]] = []
    for position, entry in selected:
        image = dataset_path / entry.image_path
        if not image.exists():
            raise InputNotFoundError(f"Parity crop missing on disk: {image}")
        batch = preprocess_crop(image)
        if device != "cpu":
            batch = batch.to(device)
        live_embedding = extract_backbone_embedding(model, batch)
        cached = cache.embeddings[position].to(device)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                live_embedding[0].flatten(), cached.flatten(), dim=0
            )
        )
        with torch.no_grad():
            live_logits = head(live_embedding)[0].detach().cpu()
            cached_logits = head(cached.unsqueeze(0))[0].detach().cpu()
        logits_match = bool(
            torch.allclose(live_logits, cached_logits, atol=LOGITS_ATOL, rtol=LOGITS_RTOL)
        )
        if cosine < EMBEDDING_COSINE_MIN or not logits_match:
            raise ConfigError(
                f"Parity failed for {entry.image_path}: "
                f"cosine={cosine:.6f} logits_match={logits_match}"
            )
        crops.append(
            {
                "image_path": entry.image_path,
                "label_index": entry.label_index,
                "cosine_similarity": cosine,
                "logits_match": logits_match,
            }
        )
    return {
        "status": "ok",
        "weights": str(weights_path),
        "dataset_root": str(dataset_path),
        "embeddings": str(cache_file),
        "checkpoint": str(checkpoint_file),
        "crops_checked": len(crops),
        "min_cosine_similarity": min(row["cosine_similarity"] for row in crops),
        "required_tolerance": {
            "embedding_cosine_similarity": EMBEDDING_COSINE_MIN,
            "logits_allclose": LOGITS_TOLERANCE_LABEL,
        },
        "crops": crops,
    }
