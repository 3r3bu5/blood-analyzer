from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from bloodfilm.errors import ImageDecodeError, InputNotFoundError, UnsupportedImageFormatError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageData:
    path: Path
    width: int
    height: int
    mode: str
    pixels: bytes

    @property
    def channels(self) -> int:
        return 3 if self.mode == "RGB" else 1


def load_image(path: Path | str) -> ImageData:
    image_path = Path(path)
    if not image_path.exists():
        raise InputNotFoundError(f"Image not found: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise UnsupportedImageFormatError(f"Unsupported image extension: {suffix}")
    if suffix == ".png":
        return _load_png(image_path)
    return _load_with_pillow(image_path)


@dataclass(frozen=True)
class ImageMetadata:
    path: Path
    width: int
    height: int
    mode: str


def probe_image(path: Path | str) -> ImageMetadata:
    """Read image dimensions and mode from the file header without decoding pixels."""
    image_path = Path(path)
    if not image_path.exists():
        raise InputNotFoundError(f"Image not found: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise UnsupportedImageFormatError(f"Unsupported image extension: {suffix}")
    if suffix == ".png":
        width, height, color_type = _read_png_header(image_path.read_bytes(), image_path)
        return ImageMetadata(
            path=image_path, width=width, height=height, mode=_PNG_MODES[color_type]
        )
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ImageDecodeError(
            f"Decoding {image_path.suffix} requires Pillow or OpenCV; install the ml extras"
        ) from exc
    try:
        with Image.open(image_path) as image:
            return ImageMetadata(
                path=image_path, width=image.width, height=image.height, mode=image.mode
            )
    except Exception as exc:
        raise ImageDecodeError(f"Could not read image header: {image_path}") from exc


def _load_with_pillow(path: Path) -> ImageData:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ImageDecodeError(
            f"Decoding {path.suffix} requires Pillow or OpenCV; install the ml extras"
        ) from exc

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return ImageData(
                path=path, width=rgb.width, height=rgb.height, mode="RGB", pixels=rgb.tobytes()
            )
    except Exception as exc:  # pragma: no cover - dependent on optional Pillow codecs
        raise ImageDecodeError(f"Could not decode image: {path}") from exc


_PNG_MODES = {0: "L", 2: "RGB", 6: "RGBA"}


def _load_png(path: Path) -> ImageData:
    data = path.read_bytes()
    width, height, color_type = _read_png_header(data, path)

    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = width * channels
    compressed = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        if kind == b"IDAT":
            compressed.extend(data[payload_start:payload_end])
        elif kind == b"IEND":
            break
        offset = payload_end + 4
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ImageDecodeError(f"Could not decompress PNG data: {path}") from exc

    rows = _unfilter_png_rows(raw, width=width, height=height, channels=channels, stride=stride)
    rgb = bytearray()
    for row in rows:
        for x in range(width):
            start = x * channels
            if color_type == 0:
                value = row[start]
                rgb.extend((value, value, value))
            else:
                rgb.extend(row[start : start + 3])
    return ImageData(path=path, width=width, height=height, mode="RGB", pixels=bytes(rgb))


def _read_png_header(data: bytes, path: Path) -> tuple[int, int, int]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageDecodeError(f"Invalid PNG signature: {path}")
    offset = 8
    while offset < len(data):
        if offset + 8 > len(data):
            raise ImageDecodeError(f"Truncated PNG chunk header: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset = offset + 8 + length + 4
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ImageDecodeError(f"Unsupported PNG encoding settings: {path}")
            if bit_depth != 8 or color_type not in _PNG_MODES:
                raise ImageDecodeError(
                    "Only 8-bit grayscale/RGB/RGBA PNG images are supported without Pillow"
                )
            return width, height, color_type
        if kind == b"IEND":
            break
    raise ImageDecodeError(f"PNG missing IHDR: {path}")


def _unfilter_png_rows(
    raw: bytes, *, width: int, height: int, channels: int, stride: int
) -> list[bytearray]:
    rows: list[bytearray] = []
    offset = 0
    previous = bytearray(stride)
    for _ in range(height):
        if offset >= len(raw):
            raise ImageDecodeError("PNG pixel data ended before all rows were decoded")
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        if len(row) != stride:
            raise ImageDecodeError("PNG row is truncated")
        _apply_png_filter(row, previous, filter_type, channels)
        rows.append(row)
        previous = row
    return rows


def _apply_png_filter(row: bytearray, previous: bytearray, filter_type: int, bpp: int) -> None:
    for index, value in enumerate(row):
        left = row[index - bpp] if index >= bpp else 0
        up = previous[index]
        up_left = previous[index - bpp] if index >= bpp else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth(left, up, up_left)
        else:
            raise ImageDecodeError(f"Unsupported PNG filter type: {filter_type}")
        row[index] = (value + predictor) & 0xFF


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left
