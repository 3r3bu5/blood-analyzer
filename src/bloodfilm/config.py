from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document
from bloodfilm.errors import ConfigError
from bloodfilm.imaging.quality import QualityConfig


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    device: str = "auto"
    run_root: Path = Path("outputs/runs")


@dataclass(frozen=True)
class ImagingConfig:
    crop_padding_ratio: float = 0.10
    color_normalization: str = "none"
    save_crops: bool = True
    save_annotated: bool = True


@dataclass(frozen=True)
class ClassifierConfig:
    backbone: str = "dinobloom_b"
    weights: Path = Path("models/backbones/dinobloom-b.pth")
    class_names_file: Path = Path("configs/mappings/mll23.yaml")
    num_classes: int = 18
    head: str | None = None
    batch_size: int | None = None
    mixed_precision: bool | None = None


@dataclass(frozen=True)
class DatasetConfig:
    mll23_root: Path = Path("data/raw/MLL23")
    manifest_path: Path = Path("data/manifests/mll23_manifest.csv")
    split_manifest_path: Path = Path("data/manifests/mll23_splits.csv")


@dataclass(frozen=True)
class RegistryConfig:
    assets: Path = Path("configs/registry/assets.yaml")


@dataclass(frozen=True)
class DetectorConfig:
    implementation: str | None = None
    weights: Path | None = None
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    image_size: int | None = None


@dataclass(frozen=True)
class TilingConfig:
    enabled: bool = False
    tile_size: int = 384
    overlap_ratio: float = 0.20
    tile_core_ownership: bool = True
    containment_deduplication: bool = True
    profile: str | None = None


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    detector_profile: str | None = None


def load_config(path: Path | str) -> AppConfig:
    raw = load_config_document(path)
    _reject_unknown_sections(raw)
    return AppConfig(
        project=_project_config(raw.get("project", {})),
        quality=_quality_config(raw.get("quality", {})),
        imaging=_imaging_config(raw.get("imaging", {})),
        classifier=_classifier_config(raw.get("classifier", {})),
        dataset=_dataset_config(raw.get("dataset", {})),
        registry=_registry_config(raw.get("registry", {})),
        detector=_detector_config(raw.get("detector", {})),
        tiling=_tiling_config(raw.get("tiling", {})),
        detector_profile=_optional_str(raw.get("detector_profile")),
    )


def _coerce_object(value: object, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be an object")
    return value


def _project_config(raw: object) -> ProjectConfig:
    data = _coerce_object(raw, "project")
    return ProjectConfig(
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        run_root=Path(str(data.get("run_root", "outputs/runs"))),
    )


def _quality_config(raw: object) -> QualityConfig:
    data = _coerce_object(raw, "quality")
    return QualityConfig(
        min_width=int(data.get("min_width", 512)),
        min_height=int(data.get("min_height", 512)),
        min_focus_score=_optional_float(data.get("min_focus_score")),
        min_mean_brightness=_optional_float(data.get("min_mean_brightness")),
        max_mean_brightness=_optional_float(data.get("max_mean_brightness")),
        max_dark_fraction=_optional_float(data.get("max_dark_fraction")),
        max_bright_fraction=_optional_float(data.get("max_bright_fraction")),
        action_on_unusable=str(data.get("action_on_unusable", "stop")),
    )


def _imaging_config(raw: object) -> ImagingConfig:
    data = _coerce_object(raw, "imaging")
    padding = float(data.get("crop_padding_ratio", 0.10))
    if padding < 0:
        raise ConfigError("imaging.crop_padding_ratio must be non-negative")
    return ImagingConfig(
        crop_padding_ratio=padding,
        color_normalization=str(data.get("color_normalization", "none")),
        save_crops=_coerce_bool(data.get("save_crops", True), "imaging.save_crops"),
        save_annotated=_coerce_bool(data.get("save_annotated", True), "imaging.save_annotated"),
    )


def _classifier_config(raw: object) -> ClassifierConfig:
    data = _coerce_object(raw, "classifier")
    return ClassifierConfig(
        backbone=str(data.get("backbone", "dinobloom_b")),
        weights=Path(str(data.get("weights", "models/backbones/dinobloom-b.pth"))),
        class_names_file=Path(str(data.get("class_names_file", "configs/mappings/mll23.yaml"))),
        num_classes=int(data.get("num_classes", 18)),
        head=_optional_str(data.get("head")),
        batch_size=_optional_int(data.get("batch_size")),
        mixed_precision=_optional_bool(data.get("mixed_precision"), "classifier.mixed_precision"),
    )


def _dataset_config(raw: object) -> DatasetConfig:
    data = _coerce_object(raw, "dataset")
    return DatasetConfig(
        mll23_root=Path(str(data.get("mll23_root", "data/raw/MLL23"))),
        manifest_path=Path(str(data.get("manifest_path", "data/manifests/mll23_manifest.csv"))),
        split_manifest_path=Path(
            str(data.get("split_manifest_path", "data/manifests/mll23_splits.csv"))
        ),
    )


def _registry_config(raw: object) -> RegistryConfig:
    data = _coerce_object(raw, "registry")
    return RegistryConfig(assets=Path(str(data.get("assets", "configs/registry/assets.yaml"))))


def _detector_config(raw: object) -> DetectorConfig:
    data = _coerce_object(raw, "detector")
    return DetectorConfig(
        implementation=_optional_str(data.get("implementation")),
        weights=_optional_path(data.get("weights")),
        confidence_threshold=_optional_float(data.get("confidence_threshold")),
        iou_threshold=_optional_float(data.get("iou_threshold")),
        image_size=_optional_int(data.get("image_size")),
    )


def _tiling_config(raw: object) -> TilingConfig:
    data = _coerce_object(raw, "tiling")
    tile_size = int(data.get("tile_size", 384))
    overlap_ratio = float(data.get("overlap_ratio", 0.20))
    if tile_size <= 0:
        raise ConfigError("tiling.tile_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ConfigError("tiling.overlap_ratio must be in [0, 1)")
    return TilingConfig(
        enabled=_coerce_bool(data.get("enabled", False), "tiling.enabled"),
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        tile_core_ownership=_coerce_bool(
            data.get("tile_core_ownership", True), "tiling.tile_core_ownership"
        ),
        containment_deduplication=_coerce_bool(
            data.get("containment_deduplication", True), "tiling.containment_deduplication"
        ),
        profile=_optional_str(data.get("profile")),
    )


def _reject_unknown_sections(raw: dict[str, Any]) -> None:
    known = {
        "project",
        "quality",
        "imaging",
        "classifier",
        "dataset",
        "registry",
        "detector",
        "tiling",
        "detector_profile",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Unknown config sections: {unknown}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any, field: str) -> bool | None:
    return None if value is None else _coerce_bool(value, field)


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


def _coerce_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ConfigError(f"{field} must be a boolean")
