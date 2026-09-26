from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from bloodfilm.errors import ChecksumMismatchError, InputNotFoundError

SUPPORTED_HASH_ALGORITHMS = ("md5", "sha256")


def hash_file(path: Path | str, algorithm: str = "sha256") -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise InputNotFoundError(f"File not found: {file_path}")
    if algorithm not in SUPPORTED_HASH_ALGORITHMS:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    digest = hashlib.new(algorithm)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path | str, *, md5: str | None = None, sha256: str | None = None) -> None:
    if md5 is None and sha256 is None:
        raise ValueError("verify_file requires an expected md5 or sha256 checksum")
    if md5 is not None and hash_file(path, "md5") != md5.lower():
        raise ChecksumMismatchError(f"MD5 mismatch: {path}")
    if sha256 is not None and hash_file(path, "sha256") != sha256.lower():
        raise ChecksumMismatchError(f"SHA-256 mismatch: {path}")


def find_duplicates(paths: Sequence[Path | str]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_hash[hash_file(path)].append(str(path))
    return sorted(group for group in by_hash.values() if len(group) > 1)
