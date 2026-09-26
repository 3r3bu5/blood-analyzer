from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bloodfilm.errors import MissingAssetError, ModelLoadError
from bloodfilm.util import sha256_file


@dataclass(frozen=True)
class DinoBloomSmokeResult:
    variant: str
    expected_embedding_dim: int
    weight_path: str
    weight_sha256: str | None
    status: str
    blocker: str | None


def smoke_dinobloom_b(weight_path: Path | str) -> DinoBloomSmokeResult:
    path = Path(weight_path)
    if not path.exists():
        return DinoBloomSmokeResult(
            variant="DinoBloom-B",
            expected_embedding_dim=768,
            weight_path=str(path),
            weight_sha256=None,
            status="blocked",
            blocker=(
                "DinoBloom-B weights are not present. Download the selected artifact explicitly, "
                "record its license and SHA-256 in configs/registry/assets.yaml, then rerun smoke."
            ),
        )
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModelLoadError("PyTorch is required to load DinoBloom-B weights") from exc
    return DinoBloomSmokeResult(
        variant="DinoBloom-B",
        expected_embedding_dim=768,
        weight_path=str(path),
        weight_sha256=sha256_file(path),
        status="weights_present_loader_not_implemented_in_m0",
        blocker="M0 verifies explicit assets only; full model loading belongs to classifier baseline work.",
    )


def require_dinobloom_b_weights(weight_path: Path | str) -> None:
    result = smoke_dinobloom_b(weight_path)
    if result.status == "blocked":
        raise MissingAssetError(result.blocker or "DinoBloom-B weights are missing")
