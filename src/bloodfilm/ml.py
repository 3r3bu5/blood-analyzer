from __future__ import annotations

from typing import Any

from bloodfilm.errors import ConfigError


def require_torch(feature: str) -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ConfigError(f"{feature} requires torch; install the ml extras first") from exc
    return torch
