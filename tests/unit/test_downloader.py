from __future__ import annotations

import functools
import http.server
import json
import threading
import zipfile
from pathlib import Path
from urllib.request import pathname2url

import pytest

from bloodfilm.data.completion import require_complete_dataset
from bloodfilm.data.downloader import (
    DownloadFile,
    download_dataset,
    download_files,
    verify_downloads,
)
from bloodfilm.data.integrity import hash_file
from bloodfilm.errors import ChecksumMismatchError, DatasetNotAvailableError, DownloadError


def _file_url(path: Path) -> str:
    return "file://" + pathname2url(str(path))


def test_download_fetches_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"class-archive-bytes")
    dest_dir = tmp_path / "dest"

    report = download_files(
        [DownloadFile(filename="archive.bin", url=_file_url(source))],
        dest_dir,
    )

    assert (dest_dir / "archive.bin").read_bytes() == b"class-archive-bytes"
    assert report["files"][0]["status"] == "downloaded"
    assert report["files"][0]["verified"] is False
    assert report["status"] == "complete"


def test_download_skips_verified_files_without_rewriting(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-bytes")
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    cached = dest_dir / "archive.bin"
    cached.write_bytes(b"stable-bytes")
    checksum = hash_file(cached, "sha256")
    before = cached.stat().st_mtime_ns

    report = download_files(
        [DownloadFile(filename="archive.bin", url=_file_url(source), sha256=checksum)],
        dest_dir,
    )

    assert report["files"][0]["status"] == "skipped_verified"
    assert report["files"][0]["verified"] is True
    assert cached.stat().st_mtime_ns == before


def test_download_refuses_extraction_on_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("cell.png", b"pixels")
    dest_dir = tmp_path / "dest"

    with pytest.raises(ChecksumMismatchError):
        download_files(
            [
                DownloadFile(
                    filename="source.zip",
                    url=_file_url(source),
                    sha256="0" * 64,
                    extract=True,
                )
            ],
            dest_dir,
        )

    assert not (dest_dir / "source").exists()


def test_download_rejects_zip_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "evil.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../escape.txt", b"escape")
    dest_dir = tmp_path / "dest"

    with pytest.raises(DownloadError):
        download_files(
            [DownloadFile(filename="evil.zip", url=_file_url(source), extract=True)],
            dest_dir,
        )

    assert not (tmp_path / "escape.txt").exists()
    assert not (dest_dir / "evil").exists()


def test_download_extracts_atomically_to_final_directory(tmp_path: Path) -> None:
    source = tmp_path / "data.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Basophil/cell.png", b"pixels")
    dest_dir = tmp_path / "dest"

    report = download_files(
        [DownloadFile(filename="data.zip", url=_file_url(source), extract=True)],
        dest_dir,
    )

    assert (dest_dir / "data" / "Basophil" / "cell.png").read_bytes() == b"pixels"
    assert report["files"][0]["status"] == "downloaded"
    assert report["files"][0]["extracted"] == ["Basophil/cell.png"]
    leftovers = [path for path in dest_dir.iterdir() if path.suffix == ".tmp"]
    assert leftovers == []


def test_download_recovers_interrupted_http_transfer(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 64
    server_root = tmp_path / "server"
    server_root.mkdir()
    (server_root / "archive.bin").write_bytes(payload)
    server = _serve_directory(server_root)
    try:
        port = server.server_address[1]
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        partial = dest_dir / "archive.bin"
        partial.write_bytes(payload[: len(payload) // 2])

        report = download_files(
            [
                DownloadFile(
                    filename="archive.bin",
                    url=f"http://127.0.0.1:{port}/archive.bin",
                )
            ],
            dest_dir,
        )

        assert partial.read_bytes() == payload
        assert report["files"][0]["status"] in {"resumed", "downloaded"}
    finally:
        server.shutdown()


def _serve_directory(root: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _registry_with_files(tmp_path: Path, source_zip: Path, checksum: str) -> Path:
    registry = tmp_path / "assets.yaml"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "name": "MLL23",
                        "local_path": "data/raw/MLL23",
                        "label_mapping": "configs/mappings/mll23.yaml",
                        "grouping_field": "patient_or_source_group",
                        "files": [
                            {
                                "filename": "mll23.zip",
                                "url": _file_url(source_zip),
                                "md5": checksum,
                                "extract": True,
                            }
                        ],
                    }
                ],
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_download_dataset_end_to_end_through_registry(tmp_path: Path) -> None:
    source_zip = tmp_path / "mll23.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("Basophil/cell.png", b"pixels")
    registry = _registry_with_files(tmp_path, source_zip, hash_file(source_zip, "md5"))
    dest_root = tmp_path / "raw"

    first = download_dataset("MLL23", registry, dest_root, tmp_path / "first.json")

    assert first["status"] == "complete"
    assert first["files"][0]["status"] == "downloaded"
    assert first["files"][0]["verified"] is True
    assert (dest_root / "MLL23" / "mll23" / "Basophil" / "cell.png").read_bytes() == b"pixels"

    completion = json.loads((dest_root / "MLL23" / ".complete.json").read_text(encoding="utf-8"))
    assert completion["verified"] is True

    verified = verify_downloads("MLL23", registry, dest_root, tmp_path / "verify.json")

    assert verified["status"] == "complete"
    assert verified["files"][0]["status"] == "verified"

    second = download_dataset("MLL23", registry, dest_root, tmp_path / "second.json")

    assert second["status"] == "complete"
    assert second["files"][0]["status"] == "skipped_verified"


def test_unverified_download_blocks_training_gate(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"no-checksum-bytes")
    registry = tmp_path / "assets.yaml"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "name": "MLL23",
                        "local_path": "data/raw/MLL23",
                        "files": [
                            {"filename": "source.bin", "url": _file_url(source)},
                        ],
                    }
                ],
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    dest_root = tmp_path / "raw"

    report = download_dataset("MLL23", registry, dest_root, tmp_path / "dl.json")

    assert report["status"] == "complete"
    assert report["files"][0]["verified"] is False
    with pytest.raises(DatasetNotAvailableError):
        require_complete_dataset(dest_root / "MLL23", dataset="MLL23")

    verify = verify_downloads("MLL23", registry, dest_root, tmp_path / "v.json")

    assert verify["files"][0]["status"] == "present_unverified"
    with pytest.raises(DatasetNotAvailableError):
        require_complete_dataset(dest_root / "MLL23", dataset="MLL23")


def test_verify_invalidates_stale_completion_record(tmp_path: Path) -> None:
    from bloodfilm.data.completion import dataset_completion_status

    source = tmp_path / "source.bin"
    source.write_bytes(b"v1-bytes")
    registry = tmp_path / "assets.yaml"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "name": "MLL23",
                        "local_path": "data/raw/MLL23",
                        "files": [
                            {
                                "filename": "source.bin",
                                "url": _file_url(source),
                                "md5": hash_file(source, "md5"),
                            },
                        ],
                    }
                ],
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    dest_root = tmp_path / "raw"

    assert download_dataset("MLL23", registry, dest_root)["status"] == "complete"
    assert dataset_completion_status(dest_root / "MLL23")["verified"] is True

    (dest_root / "MLL23" / "source.bin").write_bytes(b"tampered-bytes")
    verify = verify_downloads("MLL23", registry, dest_root, tmp_path / "v.json")

    assert verify["status"] == "partial"
    assert verify["files"][0]["status"] == "checksum_mismatch"
    assert dataset_completion_status(dest_root / "MLL23")["verified"] is False
    with pytest.raises(DatasetNotAvailableError):
        require_complete_dataset(dest_root / "MLL23", dataset="MLL23")
