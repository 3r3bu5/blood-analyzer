from pathlib import Path

import pytest

from bloodfilm.data.registry import load_asset_registry
from bloodfilm.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_asset_registry_accepts_repo_registry() -> None:
    registry = load_asset_registry(REPO_ROOT / "configs/registry/assets.yaml")

    assert registry["schema_version"] == 1
    assert registry["datasets"][0]["name"] == "MLL23"
    assert registry["models"][0]["name"] == "DinoBloom-B"


def test_load_asset_registry_rejects_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "assets.yaml"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ConfigError):
        load_asset_registry(path)
