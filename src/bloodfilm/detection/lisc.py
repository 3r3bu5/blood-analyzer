from __future__ import annotations

from pathlib import Path

from bloodfilm.detection.annotations import Box
from bloodfilm.imaging.io import load_image


def mask_to_padded_box(
    mask_paths: list[Path],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Box | None:
    """Convert one or more binary masks into a clipped padded bounding box."""
    foreground: list[tuple[int, int]] = []
    for mask_path in mask_paths:
        mask = load_image(mask_path)
        if mask.width != image_width or mask.height != image_height:
            raise ValueError(f"Mask dimensions do not match image dimensions: {mask_path}")
        for y in range(mask.height):
            for x in range(mask.width):
                offset = (y * mask.width + x) * mask.channels
                if max(mask.pixels[offset : offset + mask.channels]) > 0:
                    foreground.append((x, y))
    if not foreground:
        return None
    x_values = [x for x, _ in foreground]
    y_values = [y for _, y in foreground]
    x1 = min(x_values)
    y1 = min(y_values)
    x2 = max(x_values) + 1
    y2 = max(y_values) + 1
    pad = round(max(x2 - x1, y2 - y1) * padding_ratio)
    return Box(
        x1=float(max(0, x1 - pad)),
        y1=float(max(0, y1 - pad)),
        x2=float(min(image_width, x2 + pad)),
        y2=float(min(image_height, y2 + pad)),
    )
