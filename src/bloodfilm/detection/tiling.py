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
        box_status=detection.box_status,
        box_review_reasons=detection.box_review_reasons,
    )


def detection_center_in_tile_core(
    detection: Detection,
    tile: Tile,
    tiles: list[Tile],
    *,
    image_width: int,
    image_height: int,
) -> bool:
    core = tile_core(tile, tiles, image_width=image_width, image_height=image_height)
    center_x = (detection.x1 + detection.x2) / 2.0
    center_y = (detection.y1 + detection.y2) / 2.0
    return core.x1 <= center_x < core.x2 and core.y1 <= center_y < core.y2


def tile_core(tile: Tile, tiles: list[Tile], *, image_width: int, image_height: int) -> Tile:
    x_intervals = sorted({(row.x1, row.x2) for row in tiles})
    y_intervals = sorted({(row.y1, row.y2) for row in tiles})
    core_x1, core_x2 = _core_axis_bounds(tile.x1, tile.x2, x_intervals, image_width)
    core_y1, core_y2 = _core_axis_bounds(tile.y1, tile.y2, y_intervals, image_height)
    return Tile(core_x1, core_y1, core_x2, core_y2, f"{tile.tile_id}-core")


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


def global_deduplicate(detections: list[Detection], *, iou_threshold: float) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda row: (-row.score, row.label, row.x1, row.y1, row.x2, row.y2),
    )
    kept: list[Detection] = []
    for candidate in ordered:
        if all(
            not _is_duplicate(candidate, existing, iou_threshold=iou_threshold) for existing in kept
        ):
            kept.append(candidate)
    return kept


def _is_duplicate(candidate: Detection, existing: Detection, *, iou_threshold: float) -> bool:
    if candidate.label != existing.label:
        return False
    if iou(candidate, existing) >= iou_threshold:
        return True
    intersection = _intersection_area(candidate, existing)
    if intersection <= 0:
        return False
    candidate_area = _area(candidate)
    existing_area = _area(existing)
    smaller_area = min(candidate_area, existing_area)
    larger_area = max(candidate_area, existing_area)
    if smaller_area <= 0 or larger_area <= 0:
        return False
    overlap_smaller = intersection / smaller_area
    area_similarity = smaller_area / larger_area
    center_distance = _center_distance(candidate, existing)
    average_short_side = (
        min(candidate.x2 - candidate.x1, candidate.y2 - candidate.y1)
        + min(existing.x2 - existing.x1, existing.y2 - existing.y1)
    ) / 2.0
    return (
        overlap_smaller >= 0.80
        and area_similarity >= 0.30
        and center_distance <= average_short_side * 0.45
    )


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))


def _core_axis_bounds(
    start: int, end: int, intervals: list[tuple[int, int]], image_length: int
) -> tuple[int, int]:
    starts = [item[0] for item in intervals]
    index = starts.index(start)
    core_start = 0 if index == 0 else round((start + intervals[index - 1][1]) / 2)
    core_end = (
        image_length if index == len(intervals) - 1 else round((end + intervals[index + 1][0]) / 2)
    )
    return core_start, core_end


def _intersection_area(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def _area(detection: Detection) -> float:
    return max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)


def _center_distance(left: Detection, right: Detection) -> float:
    left_x = (left.x1 + left.x2) / 2.0
    left_y = (left.y1 + left.y2) / 2.0
    right_x = (right.x1 + right.x2) / 2.0
    right_y = (right.y1 + right.y2) / 2.0
    return ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5
