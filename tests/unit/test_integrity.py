from pathlib import Path

import pytest

from bloodfilm.data.integrity import find_duplicates, hash_file, verify_file
from bloodfilm.errors import ChecksumMismatchError, InputNotFoundError


def test_hash_and_verify_accepts_sha256_and_md5(tmp_path: Path) -> None:
    target = tmp_path / "cell.bin"
    target.write_bytes(b"wbc-bytes")

    verify_file(target, sha256=hash_file(target, "sha256"))
    verify_file(target, md5=hash_file(target, "md5"))


def test_verify_rejects_checksum_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "cell.bin"
    target.write_bytes(b"wbc-bytes")

    with pytest.raises(ChecksumMismatchError):
        verify_file(target, sha256="0" * 64)


def test_verify_missing_file_raises_input_not_found(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        verify_file(tmp_path / "absent.bin", sha256="0" * 64)


def test_find_duplicates_groups_by_content_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    other = tmp_path / "other.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    other.write_bytes(b"different")

    duplicates = find_duplicates([first, second, other])

    assert len(duplicates) == 1
    assert sorted(duplicates[0]) == [str(first), str(second)]
