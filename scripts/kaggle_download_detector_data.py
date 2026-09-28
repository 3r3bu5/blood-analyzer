from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
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
        report["leukemia_attri"] = _prepare_leukemia_attri(
            url=args.leukemia_url,
            output=args.leukemia_output,
            dry_run=args.dry_run,
            attempts=args.leukemia_attempts,
            retry_sleep_seconds=args.leukemia_retry_sleep,
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
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "dry_run",
            "output": str(output),
            "command": _gdown_folder_command(url, output, remaining_ok=True),
        }
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
