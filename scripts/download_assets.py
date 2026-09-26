"""Thin wrapper: download a registered dataset via the bloodfilm package."""

from __future__ import annotations

import argparse
from pathlib import Path

from bloodfilm.data import download_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a registered dataset")
    parser.add_argument("name")
    parser.add_argument("--registry", type=Path, default=Path("configs/registry/assets.yaml"))
    parser.add_argument("--dest-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    download_dataset(args.name, args.registry, args.dest_root, args.report_output)
    print(args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
