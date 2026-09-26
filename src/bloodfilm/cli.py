from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from bloodfilm.classification.dinobloom import smoke_dinobloom_b
from bloodfilm.config import load_config
from bloodfilm.data import (
    audit_dataset,
    create_leakage_report,
    create_split_manifest,
    download_dataset,
    list_assets,
    read_manifest,
    resolve_dataset_paths,
    run_build_manifest,
    verify_downloads,
    write_split_manifest,
)
from bloodfilm.documents import write_json
from bloodfilm.environment import capture_environment_report
from bloodfilm.errors import BloodFilmError, ConfigError
from bloodfilm.imaging import assess_quality, load_image


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BloodFilmError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloodfilm")
    subcommands = parser.add_subparsers(required=True)

    environment = subcommands.add_parser(
        "environment", help="Capture environment and asset inventory"
    )
    environment.add_argument(
        "--output", type=Path, default=Path("outputs/reports/environment.json")
    )
    environment.set_defaults(handler=_environment)

    smoke = subcommands.add_parser("smoke", help="Run M0 smoke checks without substituting assets")
    smoke.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    smoke.add_argument("--image", type=Path, default=None)
    smoke.add_argument("--output", type=Path, default=Path("outputs/reports/smoke.json"))
    smoke.set_defaults(handler=_smoke)

    data = subcommands.add_parser(
        "dataset", aliases=["data"], help="Dataset manifest, audit and split workflows"
    )
    data_subcommands = data.add_subparsers(required=True)

    audit = data_subcommands.add_parser("audit", help="Audit a registered dataset")
    audit.add_argument("name")
    audit.add_argument("--registry", type=Path, default=Path("configs/registry/assets.yaml"))
    audit.add_argument("--dataset-root", type=Path, default=None)
    audit.add_argument("--report-output", type=Path, required=True)
    audit.set_defaults(handler=_audit)

    build_manifest = data_subcommands.add_parser("build-manifest", help="Build an MLL23 manifest")
    build_manifest.add_argument("name", nargs="?", default=None)
    build_manifest.add_argument("--dataset-root", type=Path, default=None)
    build_manifest.add_argument("--mapping", type=Path, default=None)
    build_manifest.add_argument(
        "--registry", type=Path, default=Path("configs/registry/assets.yaml")
    )
    build_manifest.add_argument("--output", type=Path, required=True)
    build_manifest.add_argument("--invalid-output", type=Path, required=True)
    build_manifest.add_argument(
        "--checksum-output", type=Path, default=Path("data/manifests/mll23_checksums.csv")
    )
    build_manifest.add_argument("--report-output", type=Path, required=True)
    build_manifest.set_defaults(handler=_build_manifest)

    assets = subcommands.add_parser("assets", help="Inspect and fetch registered assets")
    assets_subcommands = assets.add_subparsers(required=True)

    assets_list = assets_subcommands.add_parser("list", help="List registered datasets and models")
    assets_list.add_argument("--registry", type=Path, default=Path("configs/registry/assets.yaml"))
    assets_list.add_argument("--output", type=Path, required=True)
    assets_list.set_defaults(handler=_assets_list)

    assets_download = assets_subcommands.add_parser("download", help="Download a dataset")
    assets_download.add_argument("name")
    assets_download.add_argument(
        "--registry", type=Path, default=Path("configs/registry/assets.yaml")
    )
    assets_download.add_argument("--dest-root", type=Path, default=Path("data/raw"))
    assets_download.add_argument("--report-output", type=Path, required=True)
    assets_download.set_defaults(handler=_assets_download)

    assets_verify = assets_subcommands.add_parser("verify", help="Verify downloaded files")
    assets_verify.add_argument("name")
    assets_verify.add_argument(
        "--registry", type=Path, default=Path("configs/registry/assets.yaml")
    )
    assets_verify.add_argument("--dest-root", type=Path, default=Path("data/raw"))
    assets_verify.add_argument("--report-output", type=Path, required=True)
    assets_verify.set_defaults(handler=_assets_verify)

    split = data_subcommands.add_parser(
        "split", help="Create deterministic train/validation/test split"
    )
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument(
        "--report-output", type=Path, default=Path("outputs/reports/mll23_leakage_report.json")
    )
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(handler=_split)
    return parser


def _environment(args: argparse.Namespace) -> int:
    report = capture_environment_report(Path.cwd())
    write_json(args.output, report)
    print(args.output)
    return 0


def _smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report: dict[str, object] = {
        "config": str(args.config),
        "dinobloom_b": _dinobloom_smoke_report(config.classifier.weights),
        "sample_image": {
            "status": "blocked",
            "blocker": "No image supplied. Provide --image or place microscope fields under data/microscope_samples/raw_fields.",
        },
    }
    if args.image is not None:
        image = load_image(args.image)
        quality = assess_quality(image, config.quality)
        report["sample_image"] = {
            "status": "loaded",
            "path": str(args.image),
            "quality": asdict(quality),
        }
    write_json(args.output, report)
    print(args.output)
    return 0


def _dinobloom_smoke_report(weight_path: Path) -> dict[str, object]:
    try:
        return dict(asdict(smoke_dinobloom_b(weight_path)))
    except BloodFilmError as exc:
        return {
            "variant": "DinoBloom-B",
            "weight_path": str(weight_path),
            "status": "error",
            "error_code": exc.code,
            "blocker": str(exc),
        }


def _audit(args: argparse.Namespace) -> int:
    dataset_root = _resolve_dataset_root(args)
    write_json(args.report_output, audit_dataset(dataset_root))
    print(args.report_output)
    return 0


def _build_manifest(args: argparse.Namespace) -> int:
    name = getattr(args, "name", None)
    if name is not None:
        dataset_root, mapping = resolve_dataset_paths(
            args.registry, name, dataset_root=args.dataset_root, label_mapping=args.mapping
        )
        if mapping is None:
            raise ConfigError(f"Dataset {name!r} has no label mapping in the registry")
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


def _resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return Path(args.dataset_root)
    root, _ = resolve_dataset_paths(args.registry, args.name)
    return root


def _assets_list(args: argparse.Namespace) -> int:
    write_json(args.output, list_assets(args.registry))
    print(args.output)
    return 0


def _assets_download(args: argparse.Namespace) -> int:
    download_dataset(args.name, args.registry, args.dest_root, args.report_output)
    print(args.report_output)
    return 0


def _assets_verify(args: argparse.Namespace) -> int:
    verify_downloads(args.name, args.registry, args.dest_root, args.report_output)
    print(args.report_output)
    return 0


def _split(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    split_rows = create_split_manifest(rows, seed=args.seed)
    write_split_manifest(args.output, split_rows)
    write_json(args.report_output, create_leakage_report(split_rows))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
