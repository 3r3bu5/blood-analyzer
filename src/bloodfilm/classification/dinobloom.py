from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bloodfilm.errors import MissingAssetError, ModelLoadError
from bloodfilm.ml import require_torch
from bloodfilm.util import sha256_file

EXPECTED_VARIANT = "DinoBloom-B"
EXPECTED_EMBEDDING_SHAPE = (1, 768)
TIMM_MODEL_ID = "vit_base_patch14_dinov2"
TORCH_HUB_MODEL_ID = "dinov2_vitb14"
BACKBONE_IMAGE_SIZE = 224
ALLOWED_UNEXPECTED_KEYS = frozenset({"mask_token"})

PREPROCESSING = {
    "resize": [224, 224],
    "resize_method": "direct",
    "crop_method": "none",
    "input_size": BACKBONE_IMAGE_SIZE,
    "interpolation": "bilinear",
    "rgb_conversion": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "colour_normalization": "none",
}

SmokeStatus = Literal["blocked", "ok", "weights_present_no_sample_image"]

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
    preprocessing: dict[str, object] | None = None


def smoke_dinobloom_b(
    weight_path: Path | str,
    *,
    backbone_loader: BackboneLoader | None = None,
    sample_image: Path | str | None = None,
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
    if sample_image is None:
        _require_torch()
        _validate_checkpoint_readable(path)
        return DinoBloomSmokeResult(
            variant=EXPECTED_VARIANT,
            expected_embedding_dim=EXPECTED_EMBEDDING_SHAPE[1],
            weight_path=str(path),
            weight_sha256=checksum,
            status="weights_present_no_sample_image",
            blocker=(
                "Weights are present and readable. Provide --image to run the "
                "DinoBloom-B forward pass and assert [1, 768]."
            ),
        )
    return _smoke_through_backbone(path, checksum, sample_image)


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
    return require_torch("DinoBloom-B loading", ModelLoadError)


def _smoke_through_backbone(
    path: Path, checksum: str, sample_image: Path | str
) -> DinoBloomSmokeResult:
    try:
        model = load_dinobloom_b_backbone(path)
        batch = preprocess_crop(sample_image)
        embedding = extract_backbone_embedding(model, batch)
    except Exception as exc:
        raise ModelLoadError(f"DinoBloom-B backbone failed on weights {path}: {exc}") from exc
    shape = [int(dim) for dim in embedding.shape]
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
        preprocessing=dict(PREPROCESSING),
    )


def load_dinobloom_b_backbone(weight_path: Path | str, device: str = "cpu") -> Any:
    """Load the DinoBloom-B ViT-B/14 backbone in evaluation mode."""
    path = Path(weight_path)
    if not path.exists():
        raise ModelLoadError(f"DinoBloom-B weights not found: {path}")
    if device not in ("cpu", "cuda"):
        raise ModelLoadError(f"Unsupported device {device!r}; expected 'cpu' or 'cuda'")
    try:
        import timm
    except ImportError as exc:
        raise ModelLoadError(
            "Loading DinoBloom-B requires timm; install the ml extras first"
        ) from exc
    torch = _require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("CUDA device requested but torch reports no CUDA device")
    try:
        model = timm.create_model(
            TIMM_MODEL_ID, pretrained=False, num_classes=0, img_size=BACKBONE_IMAGE_SIZE
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as exc:
        raise ModelLoadError(f"DinoBloom-B weights are not loadable: {path}") from exc
    if missing:
        raise ModelLoadError(f"DinoBloom-B weights are missing keys: {sorted(missing)}")
    surplus = set(unexpected) - ALLOWED_UNEXPECTED_KEYS
    if surplus:
        raise ModelLoadError(f"DinoBloom-B weights have unexpected keys: {sorted(surplus)}")
    model.eval()
    return model.to(device)


def preprocess_crop(image_path: Path | str) -> Any:
    """Preprocess one crop with the recipe associated with the DinoBloom-B weights."""
    path = Path(image_path)
    if not path.exists():
        raise ModelLoadError(f"Sample image not found: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ModelLoadError(
            "Preprocessing crops requires Pillow; install the ml extras first"
        ) from exc
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ModelLoadError(
            "Preprocessing crops requires torchvision; install the ml extras first"
        ) from exc
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
    except Exception as exc:
        raise ModelLoadError(f"Could not decode sample image: {path}") from exc
    size = PREPROCESSING["input_size"]
    assert isinstance(size, int)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=PREPROCESSING["mean"], std=PREPROCESSING["std"]),
        ]
    )
    batch = transform(rgb).unsqueeze(0)
    if list(batch.shape) != [1, 3, size, size]:
        raise ModelLoadError(
            f"Preprocessed crop has shape {list(batch.shape)}, expected [1, 3, {size}, {size}]"
        )
    return batch


def extract_backbone_embedding(model: Any, batch: Any) -> Any:
    """Run one forward pass and assert the [1, 768] embedding contract."""
    torch = _require_torch()
    try:
        with torch.no_grad():
            embedding = model(batch)
    except Exception as exc:
        raise ModelLoadError(f"DinoBloom-B forward pass failed: {exc}") from exc
    shape = [int(dim) for dim in embedding.shape]
    if shape != list(EXPECTED_EMBEDDING_SHAPE):
        raise ModelLoadError(
            f"DinoBloom-B produced shape {shape}, expected {list(EXPECTED_EMBEDDING_SHAPE)}"
        )
    return embedding


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
    if result.status not in ("ok", "weights_present_no_sample_image"):
        raise ModelLoadError(
            result.blocker or f"DinoBloom-B weights are not verified: {weight_path}"
        )
