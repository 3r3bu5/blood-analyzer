from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def capture_environment_report(project_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {name: _version_or_missing(name) for name in _tracked_packages()},
        "git": _git_info(root),
        "paths": {
            "project_root": str(root),
            "mll23_root_exists": (root / "data/raw/MLL23").exists(),
            "microscope_samples_exist": (root / "data/microscope_samples/raw_fields").exists(),
            "dinobloom_b_weights_exist": (root / "models/backbones/dinobloom-b.pth").exists(),
        },
    }


def _tracked_packages() -> list[str]:
    return [
        "torch",
        "torchvision",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "timm",
        "ultralytics",
        "albumentations",
        "pydantic",
        "PyYAML",
        "typer",
        "rich",
        "tqdm",
        "pytest",
        "mypy",
        "ruff",
    ]


def _version_or_missing(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_info(root: Path) -> dict[str, str | bool | None]:
    if shutil.which("git") is None:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "note": "git executable not found",
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return {"available": True, "commit": commit, "dirty": bool(status.strip()), "note": None}
    except subprocess.CalledProcessError:
        return {"available": False, "commit": None, "dirty": None, "note": "not a git repository"}
