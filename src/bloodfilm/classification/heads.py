from __future__ import annotations

from typing import Any

from bloodfilm.errors import ConfigError
from bloodfilm.ml import require_torch

LINEAR = "linear"
MLP = "mlp"
COSINE = "cosine"
HEAD_NAMES = (LINEAR, MLP, COSINE)


def build_linear_head(num_classes: int, embedding_dim: int = 768) -> Any:
    torch = require_torch("Classifier heads")
    return torch.nn.Linear(embedding_dim, num_classes)


def build_mlp_head(
    num_classes: int,
    embedding_dim: int = 768,
    hidden_dim: int = 512,
    dropout: float = 0.20,
) -> Any:
    torch = require_torch("Classifier heads")
    return torch.nn.Sequential(
        torch.nn.LayerNorm(embedding_dim),
        torch.nn.Linear(embedding_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, num_classes),
    )


def build_cosine_head(
    num_classes: int, embedding_dim: int = 768, initial_scale: float = 10.0
) -> Any:
    """Cosine head per TDD 9.3: experimental peer; scale starts positive, learned."""
    torch = require_torch("Classifier heads")
    BaseModule: Any = torch.nn.Module

    class _CosineHead(BaseModule):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.class_weights = torch.nn.Parameter(torch.randn(num_classes, embedding_dim))
            self.log_scale = torch.nn.Parameter(torch.tensor(float(initial_scale)).log())

        def forward(self, embeddings: Any) -> Any:
            normalized_inputs = torch.nn.functional.normalize(embeddings, dim=1)
            normalized_weights = torch.nn.functional.normalize(self.class_weights, dim=1)
            scale = self.log_scale.exp()
            return normalized_inputs @ normalized_weights.t() * scale

    return _CosineHead()


def build_head(name: str, num_classes: int, **kwargs: Any) -> Any:
    normalized = name.strip().lower()
    if normalized == LINEAR:
        return build_linear_head(num_classes, **kwargs)
    if normalized == MLP:
        return build_mlp_head(num_classes, **kwargs)
    if normalized == COSINE:
        return build_cosine_head(num_classes, **kwargs)
    raise ConfigError(f"Unknown classifier head {name!r}; expected one of {HEAD_NAMES}")
