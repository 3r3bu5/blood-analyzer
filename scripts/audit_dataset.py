"""Thin wrapper: audit a registered dataset and write a JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path

from bloodfilm.data import audit_dataset, resolve_dataset_paths
from bloodfilm.documents import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a registered dataset")
    parser.add_argument("name")
    parser.add_argument("--registry", type=Path, default=Path("configs/registry/assets.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, _ = resolve_dataset_paths(args.registry, args.name, dataset_root=args.dataset_root)
    write_json(args.report_output, audit_dataset(root))
    print(args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
