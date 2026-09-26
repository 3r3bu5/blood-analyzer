from __future__ import annotations

from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document
from bloodfilm.errors import ConfigError


def load_asset_registry(path: Path | str) -> dict[str, Any]:
    registry = load_config_document(path)
    datasets = registry.get("datasets")
    models = registry.get("models")
    if not isinstance(datasets, list) or not isinstance(models, list):
        raise ConfigError(f"{path} must define 'datasets' and 'models' lists")
    return registry
