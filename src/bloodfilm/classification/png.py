from __future__ import annotations

import importlib
import math
import struct
import zlib
from pathlib import Path
from typing import Any


def write_matrix_png(
    path: Path, matrix: list[list[int]], class_names: list[str] | None = None
) -> None:
    if class_names and _write_matplotlib_matrix(path, matrix, class_names):
        return
    max_value = max((max(row) for row in matrix if row), default=1)
    pixels: list[tuple[int, int, int]] = []
    for row in matrix:
        for value in row:
            shade = 255 - round(255 * math.sqrt(value / max_value)) if max_value else 255
            pixels.append((shade, shade, 255))
    write_rgb_png(path, len(matrix), len(matrix), pixels)


def write_reliability_png(path: Path, rows: list[dict[str, float | int]]) -> None:
    if _write_matplotlib_reliability(path, rows):
        return
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


def _write_matplotlib_matrix(path: Path, matrix: list[list[int]], class_names: list[str]) -> bool:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except (ImportError, RuntimeError):
        return False

    size = max(8.0, min(14.0, len(class_names) * 0.65))
    figure, axis = plt.subplots(figsize=(size, size), dpi=150)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("MLL23 confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=6)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True


def _write_matplotlib_reliability(path: Path, rows: list[dict[str, float | int]]) -> bool:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except (ImportError, RuntimeError):
        return False

    bins = [int(row["bin_index"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    confidence = [float(row["confidence"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.bar(bins, accuracy, width=0.8, label="Accuracy", color="#4f9d69")
    axis.plot(bins, confidence, marker="o", label="Mean confidence", color="#c43b3b")
    axis.set_ylim(0, 1.0)
    axis.set_title("MLL23 reliability by confidence bin")
    axis.set_xlabel("Confidence bin")
    axis.set_ylabel("Fraction")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True


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
