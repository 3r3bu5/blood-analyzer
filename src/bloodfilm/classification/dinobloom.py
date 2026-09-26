from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bloodfilm.errors import MissingAssetError, ModelLoadError
from bloodfilm.util import sha256_file

EXPECTED_VARIANT = "DinoBloom-B"
EXPECTED_EMBEDDING_SHAPE = (1, 768)

SmokeStatus = Literal["blocked", "ok", "weights_present_model_definition_missing"]

BackboneLoader = Callable[[Path], Sequence[int]]


@dataclass(frozen=True)
class DinoBloomSmokeResult:
    variant: str
    expected_embedding_dim: int
    weight_path: str
    weight_sha256: str | None
    status: SmokeStatus
    blocker: str | None
    embedding_shape: list[int] | None = None


def smoke_dinobloom_b(
    weight_path: Path | str, *, backbone_loader: BackboneLoader | None = None
) -> DinoBloomSmokeResult:
    path = Path(weight_path)
    if not path.exists():
        return DinoBloomSmokeResult(
            variant=EXPECTED_VARIANT,
            expected_embedding_dim=EXPECTED_EMBEDDING_SHAPE[1],
            weight_path=str(path),
            weight_sha256=None,
            status="blocked",
            blocker=(
                "DinoBloom-B weights are not present. Download the selected artifact explicitly, "
                "record its license and SHA-256 in configs/registry/assets.yaml, then rerun smoke."
            ),
        )
    checksum = sha256_file(path)
    if backbone_loader is not None:
        return _smoke_through_loader(path, checksum, backbone_loader)
    _require_torch()
    _validate_checkpoint_readable(path)
    return DinoBloomSmokeResult(
        variant=EXPECTED_VARIANT,
        expected_embedding_dim=EXPECTED_EMBEDDING_SHAPE[1],
        weight_path=str(path),
        weight_sha256=checksum,
        status="weights_present_model_definition_missing",
        blocker=(
            "Weights are present and readable, but the M0 scaffold does not vendor the "
            "DinoBloom-B architecture. Wiring the real backbone and asserting [1, 768] "
            "belongs to classifier baseline work."
        ),
    )


def _smoke_through_loader(
    path: Path, checksum: str, backbone_loader: BackboneLoader
) -> DinoBloomSmokeResult:
    try:
        shape = [int(dim) for dim in backbone_loader(path)]
    except Exception as exc:
        raise ModelLoadError(f"DinoBloom-B backbone failed on weights {path}: {exc}") from exc
    if shape != list(EXPECTED_EMBEDDING_SHAPE):
        raise ModelLoadError(
            f"DinoBloom-B produced shape {shape}, expected {list(EXPECTED_EMBEDDING_SHAPE)}"
        )
    return DinoBloomSmokeResult(
        variant=EXPECTED_VARIANT,
        expected_embedding_dim=EXPECTED_EMBEDDING_SHAPE[1],
        weight_path=str(path),
        weight_sha256=checksum,
        status="ok",
        blocker=None,
        embedding_shape=shape,
    )


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModelLoadError("PyTorch is required to load DinoBloom-B weights") from exc
    return torch


def _validate_checkpoint_readable(path: Path) -> None:
    if path.stat().st_size == 0:
        raise ModelLoadError(f"DinoBloom-B weights are empty: {path}")
    torch = _require_torch()
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelLoadError(f"DinoBloom-B weights are not a readable checkpoint: {path}") from exc


def require_dinobloom_b_weights(
    weight_path: Path | str, *, backbone_loader: BackboneLoader | None = None
) -> None:
    result = smoke_dinobloom_b(weight_path, backbone_loader=backbone_loader)
    if result.status == "blocked":
        raise MissingAssetError(result.blocker or "DinoBloom-B weights are missing")
    if result.status != "ok":
        raise ModelLoadError(
            result.blocker or f"DinoBloom-B weights are not verified: {weight_path}"
        )
