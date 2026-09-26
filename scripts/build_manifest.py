"""Thin wrapper: build an MLL23 image manifest from a registered dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from bloodfilm.data import resolve_dataset_paths, run_build_manifest
from bloodfilm.errors import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an MLL23 image manifest")
    parser.add_argument("name", nargs="?", default=None)
    parser.add_argument("--registry", type=Path, default=Path("configs/registry/assets.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invalid-output", type=Path, required=True)
    parser.add_argument(
        "--checksum-output", type=Path, default=Path("data/manifests/mll23_checksums.csv")
    )
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.name is not None:
        dataset_root, mapping = resolve_dataset_paths(
            args.registry, args.name, dataset_root=args.dataset_root, label_mapping=args.mapping
        )
        if mapping is None:
            raise ConfigError(f"Dataset {args.name!r} has no label mapping in the registry")
    else:
        if args.dataset_root is None or args.mapping is None:
            raise ConfigError("Provide a dataset name or both --dataset-root and --mapping")
        dataset_root, mapping = args.dataset_root, args.mapping
    print(
        run_build_manifest(
            dataset_root,
            mapping,
            output=args.output,
            invalid_output=args.invalid_output,
            checksum_output=args.checksum_output,
            report_output=args.report_output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
