from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bloodfilm.data.completion import write_completion_record
from bloodfilm.data.integrity import verify_file
from bloodfilm.data.registry import RegistryDataset, RegistryFile, get_dataset
from bloodfilm.documents import write_json
from bloodfilm.errors import ChecksumMismatchError, DownloadError

CHUNK_SIZE = 1024 * 256

FileFetchStatus = Literal["downloaded", "resumed"]
FileEntryStatus = Literal["downloaded", "resumed", "skipped_verified"]
VerifyEntryStatus = Literal["verified", "missing", "present_unverified", "checksum_mismatch"]
DatasetReportStatus = Literal["complete", "partial", "blocked"]


@dataclass(frozen=True)
class DownloadFile:
    filename: str
    url: str
    md5: str | None = None
    sha256: str | None = None
    extract: bool = False

    @classmethod
    def from_registry_file(cls, item: RegistryFile) -> DownloadFile:
        return cls(
            filename=item.filename,
            url=item.url,
            md5=item.md5,
            sha256=item.sha256,
            extract=item.extract,
        )

    def has_checksums(self) -> bool:
        return self.md5 is not None or self.sha256 is not None


def download_files(
    files: list[DownloadFile], dest_dir: Path | str, *, timeout: int = 60
) -> dict[str, Any]:
    target = Path(dest_dir)
    target.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for item in files:
        entries.append(_download_one(item, target, timeout=timeout))
    done: DatasetReportStatus = "complete"
    return {"dest_dir": str(target), "files": entries, "status": done}


def download_dataset(
    name: str,
    registry_path: Path | str,
    dest_root: Path | str,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = get_dataset(registry_path, name)
    dest_dir = Path(dest_root) / dataset.dest_name()
    report: dict[str, Any] = _blocked_report(dataset, dest_dir)
    if dataset.files:
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_files(
            [DownloadFile.from_registry_file(item) for item in dataset.files],
            dest_dir,
        )
        report = {"dataset": dataset.name, **downloaded}
        write_completion_record(dest_dir, dataset=dataset.name, file_count=len(dataset.files))
    if report_path is not None:
        write_json(report_path, report)
    return report


def verify_downloads(
    name: str,
    registry_path: Path | str,
    dest_root: Path | str,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = get_dataset(registry_path, name)
    dest_dir = Path(dest_root) / dataset.dest_name()
    report: dict[str, Any] = _blocked_report(dataset, dest_dir)
    if dataset.files:
        entries = [_verify_one(item, dest_dir) for item in dataset.files]
        failed = [entry for entry in entries if entry["status"] != "verified"]
        finished: DatasetReportStatus = "complete" if not failed else "partial"
        report = {
            "dataset": dataset.name,
            "dest_dir": str(dest_dir),
            "files": entries,
            "status": finished,
        }
    if report_path is not None:
        write_json(report_path, report)
    return report


def _blocked_report(dataset: RegistryDataset, dest_dir: Path) -> dict[str, Any]:
    blocked: DatasetReportStatus = "blocked"
    return {
        "dataset": dataset.name,
        "dest_dir": str(dest_dir),
        "files": [],
        "status": blocked,
        "blocker": _no_file_entries_blocker(dataset),
    }


def _no_file_entries_blocker(dataset: RegistryDataset) -> str:
    hint = dataset.acquisition.get("note") or dataset.acquisition.get("zenodo_api")
    suffix = f" Registry hint: {hint}" if hint else ""
    return (
        f"No file entries recorded for dataset {dataset.name!r}; refusing to guess URLs. "
        f"Record official {{filename, url, md5}} entries in the asset registry first.{suffix}"
    )


def _download_one(item: DownloadFile, dest_dir: Path, *, timeout: int) -> dict[str, Any]:
    dest = dest_dir / item.filename
    if dest.exists() and item.has_checksums() and _checksums_ok(item, dest):
        skipped: FileEntryStatus = "skipped_verified"
        entry: dict[str, Any] = {
            "filename": item.filename,
            "status": skipped,
            "verified": True,
            "bytes": dest.stat().st_size,
        }
    else:
        scheme = urlparse(item.url).scheme
        if scheme == "file":
            _copy_local_file(item.url, dest)
            status: FileFetchStatus = "downloaded"
        elif scheme in {"http", "https"}:
            status = _fetch_http(item.url, dest, timeout=timeout)
        else:
            raise DownloadError(f"Unsupported URL scheme for {item.filename}: {scheme!r}")
        _verify_or_raise(item, dest)
        entry = {
            "filename": item.filename,
            "status": status,
            "verified": item.has_checksums(),
            "bytes": dest.stat().st_size,
        }
    if item.extract:
        entry["extracted"] = _extract_zip_safely(dest, dest_dir)
    return entry


def _checksums_ok(item: DownloadFile, dest: Path) -> bool:
    try:
        verify_file(dest, md5=item.md5, sha256=item.sha256)
    except ChecksumMismatchError:
        return False
    return True


def _verify_or_raise(item: DownloadFile, dest: Path) -> None:
    if not item.has_checksums():
        return
    try:
        verify_file(dest, md5=item.md5, sha256=item.sha256)
    except ChecksumMismatchError as exc:
        dest.unlink(missing_ok=True)
        raise ChecksumMismatchError(f"{item.filename} failed verification and was removed") from exc


def _copy_local_file(url: str, dest: Path) -> None:
    source = Path(urlparse(url).path)
    if not source.exists():
        raise DownloadError(f"Local source not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _fetch_http(url: str, dest: Path, *, timeout: int) -> FileFetchStatus:
    offset = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(url, headers=headers)
    status: FileFetchStatus = "downloaded"
    try:
        with urlopen(request, timeout=timeout) as response:
            assert response.status is not None
            if response.status == 206 and offset:
                mode, status = "ab", "resumed"
            else:
                if offset:
                    dest.unlink()
                    offset = 0
                mode, status = "wb", "downloaded"
            with dest.open(mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
    except ChecksumMismatchError:
        raise
    except Exception as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc
    return status


def _verify_one(item: DownloadFile | RegistryFile, dest_dir: Path) -> dict[str, Any]:
    dest = dest_dir / item.filename
    file_status: VerifyEntryStatus
    if not dest.exists():
        file_status = "missing"
    elif not item.has_checksums():
        file_status = "present_unverified"
    else:
        try:
            verify_file(dest, md5=item.md5, sha256=item.sha256)
        except ChecksumMismatchError:
            file_status = "checksum_mismatch"
        else:
            file_status = "verified"
    return {"filename": item.filename, "status": file_status}


def _extract_zip_safely(archive_path: Path, dest_dir: Path) -> list[str]:
    final_dir = dest_dir / archive_path.stem
    staging = Path(tempfile.mkdtemp(prefix=f"{archive_path.stem}.", dir=dest_dir))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            _reject_traversal(members, staging)
            archive.extractall(staging)
        extracted = sorted(
            path.relative_to(staging).as_posix()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging, final_dir)
        return extracted
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reject_traversal(members: list[str], staging: Path) -> None:
    staging_resolved = staging.resolve()
    for member in members:
        candidate = (staging_resolved / PurePosixPath(member)).resolve()
        if candidate != staging_resolved and staging_resolved not in candidate.parents:
            raise DownloadError(f"Zip entry escapes the destination: {member!r}")
