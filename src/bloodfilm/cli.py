from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from bloodfilm.classification.dinobloom import smoke_dinobloom_b
from bloodfilm.config import load_config
from bloodfilm.data.audit import manifest_audit_report
from bloodfilm.data.manifests import (
    build_mll23_manifest,
    read_manifest,
    write_checksum_manifest,
    write_invalid_manifest,
    write_manifest,
)
from bloodfilm.data.splits import create_leakage_report, create_split_manifest, write_split_manifest
from bloodfilm.documents import write_json
from bloodfilm.environment import capture_environment_report
from bloodfilm.errors import BloodFilmError
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

    data = subcommands.add_parser("data", help="Dataset manifest and split workflows")
    data_subcommands = data.add_subparsers(required=True)

    build_manifest = data_subcommands.add_parser("build-manifest", help="Build an MLL23 manifest")
    build_manifest.add_argument("--dataset-root", type=Path, required=True)
    build_manifest.add_argument("--mapping", type=Path, default=Path("configs/mappings/mll23.yaml"))
    build_manifest.add_argument("--output", type=Path, required=True)
    build_manifest.add_argument("--invalid-output", type=Path, required=True)
    build_manifest.add_argument(
        "--checksum-output", type=Path, default=Path("data/manifests/mll23_checksums.csv")
    )
    build_manifest.add_argument("--report-output", type=Path, required=True)
    build_manifest.set_defaults(handler=_build_manifest)

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


def _build_manifest(args: argparse.Namespace) -> int:
    result = build_mll23_manifest(args.dataset_root, args.mapping)
    write_manifest(args.output, result.valid_rows)
    write_invalid_manifest(args.invalid_output, result.invalid_rows)
    write_checksum_manifest(args.checksum_output, result.valid_rows)
    write_json(args.report_output, manifest_audit_report(result))
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
