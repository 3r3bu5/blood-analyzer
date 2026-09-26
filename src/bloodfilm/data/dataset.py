from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bloodfilm.errors import ModelLoadError
from bloodfilm.ml import require_torch
from bloodfilm.schemas import ManifestRow

RowLoader = Callable[[ManifestRow], Any]


class ManifestImageDataset:
    def __init__(
        self,
        rows: list[ManifestRow],
        class_names: list[str],
        loader: RowLoader | None = None,
    ) -> None:
        unknown = sorted({row.canonical_label for row in rows} - set(class_names))
        if unknown:
            raise ModelLoadError(f"Rows reference classes outside class_names: {unknown}")
        self._rows = list(rows)
        self._class_names = list(class_names)
        self._loader = loader or default_image_loader

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Any, str, int]:
        row = self._rows[index]
        return self._loader(row), row.image_id, self._class_names.index(row.canonical_label)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


def default_image_loader(row: ManifestRow) -> Any:
    torch = require_torch("The default image loader")
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModelLoadError(
            "The default image loader requires Pillow; "
            "install the ml extras or pass an explicit loader"
        ) from exc
    with Image.open(Path(row.image_path)) as image:
        rgb = image.convert("RGB")
        pixels = list(rgb.tobytes())
    tensor = torch.ByteTensor(pixels)
    return tensor.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1)
