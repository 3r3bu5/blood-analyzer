from __future__ import annotations

from dataclasses import dataclass

from bloodfilm.detection.detector import Detection
from bloodfilm.detection.metrics import iou


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int
    tile_id: str

    def to_dict(self) -> dict[str, int | str]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "tile_id": self.tile_id}


def generate_tiles(
    image_width: int,
    image_height: int,
    *,
    tile_size: int,
    overlap_ratio: float,
) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if image_width <= tile_size and image_height <= tile_size:
        return [Tile(0, 0, image_width, image_height, "tile-0000")]
    stride = max(1, round(tile_size * (1.0 - overlap_ratio)))
    x_starts = _axis_starts(image_width, tile_size, stride)
    y_starts = _axis_starts(image_height, tile_size, stride)
    tiles: list[Tile] = []
    index = 0
    for y in y_starts:
        for x in x_starts:
            tiles.append(
                Tile(
                    x,
                    y,
                    min(image_width, x + tile_size),
                    min(image_height, y + tile_size),
                    f"tile-{index:04d}",
                )
            )
            index += 1
    return tiles


def map_tile_detection(
    detection: Detection, tile: Tile, *, image_width: int, image_height: int
) -> Detection:
    return Detection(
        x1=max(0, min(image_width, detection.x1 + tile.x1)),
        y1=max(0, min(image_height, detection.y1 + tile.y1)),
        x2=max(0, min(image_width, detection.x2 + tile.x1)),
        y2=max(0, min(image_height, detection.y2 + tile.y1)),
        score=detection.score,
        label=detection.label,
    )


def global_nms(detections: list[Detection], *, iou_threshold: float) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda row: (-row.score, row.label, row.x1, row.y1, row.x2, row.y2),
    )
    kept: list[Detection] = []
    for candidate in ordered:
        if all(
            candidate.label != existing.label or iou(candidate, existing) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))
