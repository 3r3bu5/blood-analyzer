from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_TXL_URL = "https://github.com/lugan113/TXL-PBC_Dataset.git"
DEFAULT_LEUKEMIA_URL = (
    "https://drive.google.com/drive/folders/1J5ld-tK6cewj9wXWUi3rs6UdlHnDBe8U?usp=sharing"
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "dry_run": args.dry_run,
        "txl_pbc": {},
        "leukemia_attri": {},
    }
    if not args.skip_txl:
        report["txl_pbc"] = _prepare_txl_pbc(
            url=args.txl_url,
            output=args.txl_output,
            dry_run=args.dry_run,
        )
    if not args.skip_leukemia:
        zip_urls = list(args.leukemia_zip_url) + list(args.leukemia_zip_file)
        if not zip_urls:
            zip_urls = _env_zip_urls()
        report["leukemia_attri"] = _prepare_leukemia_attri(
            url=args.leukemia_url,
            output=args.leukemia_output,
            dry_run=args.dry_run,
            attempts=args.leukemia_attempts,
            retry_sleep_seconds=args.leukemia_retry_sleep,
            zip_urls=zip_urls,
        )
    _write_json(args.report_output, report)
    print(args.report_output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download detector recovery datasets into a Kaggle working directory."
    )
    parser.add_argument("--txl-url", default=DEFAULT_TXL_URL)
    parser.add_argument("--txl-output", type=Path, default=Path("data/raw/TXL-PBC"))
    parser.add_argument("--leukemia-url", default=DEFAULT_LEUKEMIA_URL)
    parser.add_argument(
        "--leukemia-zip-url",
        action="append",
        default=[],
        help=(
            "Google Takeout zip URL for the LeukemiaAttri folder. Repeat for each "
            "Takeout part, or set LEUKEMIA_ZIP_URLS (one URL per line). "
            "When given, the zip path is used instead of per-file gdown."
        ),
    )
    parser.add_argument(
        "--leukemia-zip-file",
        action="append",
        default=[],
        help=(
            "Local Takeout zip already available to the runner, e.g. an uploaded "
            "Kaggle dataset at /kaggle/input/<dataset>/<file>.zip. Repeat per part."
        ),
    )
    parser.add_argument("--leukemia-output", type=Path, default=Path("data/raw/LeukemiaAttri"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/reports/kaggle_detector_data_download.json"),
    )
    parser.add_argument("--skip-txl", action="store_true")
    parser.add_argument("--skip-leukemia", action="store_true")
    parser.add_argument("--leukemia-attempts", type=int, default=8)
    parser.add_argument("--leukemia-retry-sleep", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _prepare_txl_pbc(*, url: str, output: Path, dry_run: bool) -> dict[str, Any]:
    command = ["git", "clone", url, str(output)]
    if dry_run:
        return {
            "status": "dry_run",
            "output": str(output),
            "command": command,
            "post_clone_commands": [["git", "lfs", "install"], ["git", "lfs", "pull"]],
        }
    if not output.exists():
        _ensure_git_lfs()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    if (output / ".git").exists():
        _ensure_git_lfs()
        subprocess.run(["git", "lfs", "install"], cwd=output, check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=output, check=True)
    commit = (
        _run_text(["git", "rev-parse", "HEAD"], cwd=output) if (output / ".git").exists() else ""
    )
    image_count = len(list((output / "TXL-PBC" / "images").glob("*/*")))
    pointer_count = _count_lfs_pointer_files(output / "TXL-PBC" / "images")
    if image_count == 0:
        raise RuntimeError(f"TXL-PBC images were not found under {output / 'TXL-PBC' / 'images'}")
    if pointer_count:
        raise RuntimeError(
            f"TXL-PBC still contains {pointer_count} Git LFS pointer files after git lfs pull"
        )
    return {
        "status": "ok",
        "output": str(output),
        "dataset_root": str(output / "TXL-PBC"),
        "commit": commit,
        "data_yaml_exists": (output / "TXL-PBC" / "data.yaml").exists(),
        "image_count": image_count,
        "git_lfs_pointer_count": pointer_count,
    }


def _prepare_leukemia_attri(
    *,
    url: str,
    output: Path,
    dry_run: bool,
    attempts: int = 8,
    retry_sleep_seconds: float = 60.0,
    zip_urls: list[str] | None = None,
) -> dict[str, Any]:
    urls = [item for item in (zip_urls or []) if item.strip()]
    if dry_run:
        if urls:
            return {
                "status": "dry_run",
                "output": str(output),
                "method": "takeout_zip",
                "zip_count": len(urls),
            }
        return {
            "status": "dry_run",
            "output": str(output),
            "command": _gdown_folder_command(url, output, remaining_ok=True),
        }
    if urls:
        return _prepare_leukemia_from_zips(urls=urls, output=output)
    _ensure_gdown()
    command = _gdown_folder_command(url, output, remaining_ok=_gdown_supports_remaining_ok())
    output.mkdir(parents=True, exist_ok=True)
    completed_attempts = 0
    for attempt in range(1, max(attempts, 1) + 1):
        completed_attempts = attempt
        try:
            subprocess.run(command, check=True)
            break
        except subprocess.CalledProcessError as exc:
            if attempt >= max(attempts, 1):
                raise RuntimeError(
                    "LeukemiaAttri download stopped after "
                    f"{completed_attempts} attempt(s) with Google Drive quota errors. "
                    "Partial files are kept under data/raw/LeukemiaAttri because gdown "
                    "uses --continue, so re-running this script resumes. If quota persists, "
                    "wait before retrying or download the Drive folder manually."
                ) from exc
            time.sleep(retry_sleep_seconds)
    return {
        "status": "ok",
        "output": str(output),
        "attempts": completed_attempts,
        "json_label_count": len(list(output.glob("*/json_labels/*.json"))),
        "image_count": len(list(output.glob("*/Images/*/*")))
        + len(list(output.glob("*/images/*/*"))),
        "zip_label_count": len(list(output.glob("*/*.zip"))),
    }


def _ensure_gdown() -> None:
    if shutil.which("gdown") is not None:
        return
    try:
        __import__("gdown")
        return
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)


def _gdown_folder_command(url: str, output: Path, *, remaining_ok: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        url,
        "-O",
        str(output),
        "--continue",
    ]
    if remaining_ok:
        command.append("--remaining-ok")
    return command


def _gdown_supports_remaining_ok() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "gdown", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return True
    output = (result.stdout or "") + (result.stderr or "")
    return "--remaining-ok" in output


def _env_zip_urls() -> list[str]:
    raw = os.environ.get("LEUKEMIA_ZIP_URLS", "")
    urls: list[str] = []
    for chunk in raw.replace(",", "\n").splitlines():
        cleaned = chunk.strip().strip("'\"")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def _prepare_leukemia_from_zips(*, urls: list[str], output: Path) -> dict[str, Any]:
    download_dir = output.parent / f"{output.name}_takeout_zips"
    download_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for index, source in enumerate(urls, start=1):
        local = _existing_local_zip(source)
        if local is not None:
            parts.append(str(local))
            continue
        filename = _zip_filename(source, index)
        destination = download_dir / filename
        if (
            destination.exists()
            and destination.stat().st_size > 0
            and zipfile.is_zipfile(destination)
        ):
            parts.append(str(destination))
            continue
        _download_url_to_file(source, destination)
        parts.append(str(destination))
    for part in parts:
        _extract_zip_into(Path(part), output)
    return {
        "status": "ok",
        "method": "takeout_zip",
        "output": str(output),
        "zip_count": len(parts),
        "zip_parts": parts,
        "json_label_count": len(list(output.glob("*/json_labels/*.json"))),
        "image_count": len(list(output.glob("*/Images/*/*")))
        + len(list(output.glob("*/images/*/*"))),
        "zip_label_count": len(list(output.glob("*/*.zip"))),
    }


def _zip_filename(url: str, index: int) -> str:
    stem = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if not stem.lower().endswith(".zip") or len(stem) > 120:
        stem = f"leukemia_attri_takeout_part{index}.zip"
    return stem


def _download_url_to_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded zip is empty: {destination}")


def _existing_local_zip(source: str) -> Path | None:
    candidate = Path(source.strip().strip("'\""))
    if candidate.exists() and candidate.is_file():
        if not zipfile.is_zipfile(candidate):
            raise RuntimeError(_non_zip_error(candidate))
        return candidate
    return None


def _non_zip_error(zip_path: Path) -> str:
    size = zip_path.stat().st_size if zip_path.exists() else -1
    preview = ""
    try:
        with zip_path.open("rb") as handle:
            preview = handle.read(200).decode("utf-8", errors="replace")
    except OSError:
        preview = "<unreadable>"
    hint = ""
    if preview.lstrip().lower().startswith(("<!doctype", "<html")):
        hint = (
            " The file looks like an HTML page, so the link probably needs a Google "
            "login in a browser and Kaggle cannot fetch it. Download the Takeout zip "
            "on your own PC, upload it as a Kaggle dataset, and pass "
            "--leukemia-zip-file /kaggle/input/<dataset>/<file>.zip instead."
        )
    return (
        f"Not a zip archive: {zip_path} (size {size} bytes). First bytes: {preview[:200]!r}.{hint}"
    )


def _extract_zip_into(zip_path: Path, output: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(_non_zip_error(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        _assert_zip_members_safe(archive)
        archive.extractall(output)
    _normalize_extracted_tree(output)


def _assert_zip_members_safe(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename
        if not name or name.endswith("/"):
            continue
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise RuntimeError(f"Zip archive contains a symlink, refusing: {name}")
        normalized = name.replace("\\", "/")
        if normalized.startswith(("/", "../")) or "/../" in normalized:
            raise RuntimeError(f"Zip archive contains an unsafe path, refusing: {name}")
        first = normalized.split("/", 1)[0]
        if first.endswith(":") or first == "..":
            raise RuntimeError(f"Zip archive contains an unsafe path, refusing: {name}")


def _normalize_extracted_tree(output: Path) -> None:
    for _ in range(3):
        if list(output.glob("*/json_labels/*.json")):
            return
        subdirs = sorted(path for path in output.iterdir() if path.is_dir())
        if len(subdirs) != 1:
            return
        inner = subdirs[0]
        for child in sorted(inner.iterdir()):
            target = output / child.name
            if target.exists():
                if target.is_dir() and child.is_dir():
                    for nested in sorted(child.iterdir()):
                        shutil.move(str(nested), target / nested.name)
                else:
                    continue
            else:
                shutil.move(str(child), target)
        try:
            inner.rmdir()
        except OSError:
            return


def _ensure_git_lfs() -> None:
    result = subprocess.run(["git", "lfs", "version"], check=False, text=True, capture_output=True)
    if result.returncode == 0:
        return
    if shutil.which("apt-get") is None:
        raise RuntimeError("git-lfs is required for TXL-PBC but apt-get is unavailable")
    subprocess.run(["apt-get", "update", "-y"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)


def _count_lfs_pointer_files(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in root.glob("*/*"):
        if not path.is_file() or path.stat().st_size > 512:
            continue
        try:
            if path.read_text(encoding="utf-8").startswith("version https://git-lfs"):
                count += 1
        except UnicodeDecodeError:
            continue
    return count


def _run_text(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
