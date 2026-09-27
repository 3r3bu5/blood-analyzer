from pathlib import Path

import pytest

from bloodfilm.data.registry import load_asset_registry
from bloodfilm.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_asset_registry_accepts_repo_registry() -> None:
    registry = load_asset_registry(REPO_ROOT / "configs/registry/assets.yaml")

    assert registry["schema_version"] == 1
    assert registry["datasets"][0]["name"] == "MLL23"
    assert {dataset["name"] for dataset in registry["datasets"]} == {
        "LeukemiaAttri",
        "MLL23",
        "TXL-PBC",
    }
    txl_pbc = next(dataset for dataset in registry["datasets"] if dataset["name"] == "TXL-PBC")
    assert txl_pbc["source_url"] == "https://github.com/lugan113/TXL-PBC_Dataset"
    assert txl_pbc["local_path"] == "data/raw/TXL-PBC"
    leukemia_attri = next(
        dataset for dataset in registry["datasets"] if dataset["name"] == "LeukemiaAttri"
    )
    assert leukemia_attri["verification_state"] == "manual_download_required"
    assert leukemia_attri["expected_archive_sha256"] is None
    assert registry["models"][0]["name"] == "DinoBloom-B"


def test_load_asset_registry_rejects_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "assets.yaml"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ConfigError):
        load_asset_registry(path)
