class BloodFilmError(Exception):
    """Base exception with a stable machine-readable code."""

    code = "BLOODFILM_ERROR"

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class ConfigError(BloodFilmError):
    code = "CONFIG_ERROR"


class InputNotFoundError(BloodFilmError):
    code = "INPUT_NOT_FOUND"


class ImageDecodeError(BloodFilmError):
    code = "IMAGE_DECODE_FAILED"


class UnsupportedImageFormatError(BloodFilmError):
    code = "UNSUPPORTED_IMAGE_FORMAT"


class MissingAssetError(BloodFilmError):
    code = "MISSING_ASSET"


class ModelLoadError(BloodFilmError):
    code = "MODEL_LOAD_ERROR"


class MappingError(BloodFilmError):
    code = "LABEL_MAPPING_ERROR"


class ChecksumMismatchError(BloodFilmError):
    code = "CHECKSUM_MISMATCH"


class DownloadError(BloodFilmError):
    code = "DOWNLOAD_ERROR"
