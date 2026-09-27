from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


def write_matrix_png(path: Path, matrix: list[list[int]]) -> None:
    max_value = max((max(row) for row in matrix if row), default=1)
    pixels: list[tuple[int, int, int]] = []
    for row in matrix:
        for value in row:
            shade = 255 - round(255 * math.sqrt(value / max_value)) if max_value else 255
            pixels.append((shade, shade, 255))
    write_rgb_png(path, len(matrix), len(matrix), pixels)


def write_reliability_png(path: Path, rows: list[dict[str, float | int]]) -> None:
    width = max(len(rows), 1)
    height = 50
    pixels = [(255, 255, 255)] * (width * height)
    for x, row in enumerate(rows):
        accuracy_height = round(float(row["accuracy"]) * (height - 1))
        confidence_height = round(float(row["confidence"]) * (height - 1))
        for y in range(height - accuracy_height, height):
            pixels[y * width + x] = (80, 160, 80)
        marker_y = max(0, height - confidence_height - 1)
        pixels[marker_y * width + x] = (200, 40, 40)
    write_rgb_png(path, width, height, pixels)


def write_rgb_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )
