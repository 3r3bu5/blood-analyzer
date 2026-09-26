from __future__ import annotations

from typing import Any


def fake_embedding_bank(
    num_classes: int = 3,
    per_class: int = 8,
    embedding_dim: int = 768,
    seed: int = 0,
) -> tuple[Any, list[int]]:
    """Well-separated fake DinoBloom-style embeddings for offline training tests.

    Requires torch; callers must ``pytest.importorskip("torch")`` first.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    chunks = []
    labels: list[int] = []
    for class_id in range(num_classes):
        center = torch.zeros(embedding_dim)
        center[class_id] = 5.0
        chunks.append(center + 0.1 * torch.randn(per_class, embedding_dim, generator=generator))
        labels.extend([class_id] * per_class)
    return torch.cat(chunks), labels
