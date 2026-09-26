from __future__ import annotations

import functools
import http.server
import threading
import zipfile
from pathlib import Path
from urllib.request import pathname2url

import pytest

from bloodfilm.data.downloader import DownloadFile, download_files
from bloodfilm.data.integrity import hash_file
from bloodfilm.errors import ChecksumMismatchError, DownloadError


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
