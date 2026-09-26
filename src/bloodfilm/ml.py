from __future__ import annotations

from typing import Any

from bloodfilm.errors import BloodFilmError, ConfigError


def require_torch(feature: str, error: type[BloodFilmError] = ConfigError) -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise error(f"{feature} requires torch; install the ml extras first") from exc
    return torch
