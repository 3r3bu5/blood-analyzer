from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document
from bloodfilm.errors import ConfigError


@dataclass(frozen=True)
class RegistryFile:
    filename: str
    url: str
    md5: str | None = None
    sha256: str | None = None
    extract: bool = False

    def has_checksums(self) -> bool:
        return self.md5 is not None or self.sha256 is not None


@dataclass(frozen=True)
class RegistryDataset:
    name: str
    local_path: Path
    label_mapping: Path | None = None
    grouping_field: str = "patient_or_source_group"
    files: list[RegistryFile] = field(default_factory=list)
    acquisition: dict[str, Any] = field(default_factory=dict)

    def dest_name(self) -> str:
        return Path(self.local_path).name


def load_asset_registry(path: Path | str) -> dict[str, Any]:
    registry = load_config_document(path)
    datasets = registry.get("datasets")
    models = registry.get("models")
    if not isinstance(datasets, list) or not isinstance(models, list):
        raise ConfigError(f"{path} must define 'datasets' and 'models' lists")
    return registry


def list_assets(registry_path: Path | str) -> dict[str, Any]:
    registry = load_asset_registry(registry_path)
    return {
        "schema_version": registry.get("schema_version"),
        "datasets": {
            str(entry.get("name")): _dataset_presence(entry) for entry in registry["datasets"]
        },
        "models": {str(entry.get("name")): _model_presence(entry) for entry in registry["models"]},
    }


def get_dataset(registry_path: Path | str, name: str) -> RegistryDataset:
    registry = load_asset_registry(registry_path)
    for entry in registry["datasets"]:
        if str(entry.get("name")).lower() == name.lower():
            return _parse_dataset(entry)
    known = sorted(str(entry.get("name")) for entry in registry["datasets"])
    raise ConfigError(f"Unknown dataset {name!r}; known datasets: {known}")


def resolve_dataset_paths(
    registry_path: Path | str,
    name: str,
    *,
    dataset_root: Path | str | None = None,
    label_mapping: Path | str | None = None,
) -> tuple[Path, Path | None]:
    dataset = get_dataset(registry_path, name)
    root = Path(dataset_root) if dataset_root is not None else dataset.local_path
    mapping = Path(label_mapping) if label_mapping is not None else dataset.label_mapping
    return root, mapping


def _parse_dataset(entry: dict[str, Any]) -> RegistryDataset:
    files_raw = entry.get("files", [])
    if not isinstance(files_raw, list):
        raise ConfigError("Dataset 'files' must be a list")
    files = []
    for item in files_raw:
        if not isinstance(item, dict) or "filename" not in item or "url" not in item:
            raise ConfigError("Each dataset file needs 'filename' and 'url'")
        files.append(
            RegistryFile(
                filename=str(item["filename"]),
                url=str(item["url"]),
                md5=_optional_str(item.get("md5")),
                sha256=_optional_str(item.get("sha256")),
                extract=bool(item.get("extract", False)),
            )
        )
    acquisition = entry.get("acquisition", {})
    if not isinstance(acquisition, dict):
        raise ConfigError("Dataset 'acquisition' must be an object")
    label_mapping = entry.get("label_mapping")
    return RegistryDataset(
        name=str(entry.get("name")),
        local_path=Path(str(entry.get("local_path", ""))),
        label_mapping=Path(str(label_mapping)) if label_mapping is not None else None,
        grouping_field=str(entry.get("grouping_field", "patient_or_source_group")),
        files=files,
        acquisition=dict(acquisition),
    )


def _dataset_presence(entry: dict[str, Any]) -> dict[str, Any]:
    local_path = entry.get("local_path")
    exists = Path(str(local_path)).exists() if local_path else False
    return {
        "local_path": local_path,
        "present": exists,
        "file_count": len(entry.get("files", [])),
        "source_url": entry.get("source_url"),
    }


def _model_presence(entry: dict[str, Any]) -> dict[str, Any]:
    local_path = entry.get("local_path")
    exists = Path(str(local_path)).exists() if local_path else False
    return {"local_path": local_path, "present": exists}


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
