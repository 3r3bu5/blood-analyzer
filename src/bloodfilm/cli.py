from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bloodfilm.classification.dinobloom import (
    load_dinobloom_b_backbone,
    preprocess_crop,
    require_dinobloom_b_weights,
    smoke_dinobloom_b,
)
from bloodfilm.classification.embeddings import build_embedding_cache, save_embedding_cache
from bloodfilm.classification.heads import HEAD_NAMES
from bloodfilm.config import load_config
from bloodfilm.data import (
    audit_dataset,
    create_leakage_report,
    create_split_manifest,
    download_dataset,
    list_assets,
    read_manifest,
    require_complete_dataset,
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

    config = subcommands.add_parser("config", help="Configuration utilities")
    config_subcommands = config.add_subparsers(required=True)

    config_validate = config_subcommands.add_parser("validate", help="Validate a config file")
    config_validate.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    config_validate.set_defaults(handler=_config_validate)

    train = subcommands.add_parser("train", help="Model training entry points")
    train_subcommands = train.add_subparsers(required=True)

    train_classifier = train_subcommands.add_parser("classifier", help="Train the classifier head")
    train_classifier.add_argument(
        "--config", type=Path, default=Path("configs/classifier_mll23.yaml")
    )
    train_classifier.add_argument("--head", type=str, default=None)
    train_classifier.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings/mll23_embeddings.pt")
    )
    train_classifier.add_argument("--epochs", type=int, default=50)
    train_classifier.add_argument("--learning-rate", type=float, default=1e-3)
    train_classifier.add_argument("--seed", type=int, default=42)
    train_classifier.add_argument(
        "--evaluate-split", type=str, default="validation", choices=["validation", "test"]
    )
    train_classifier.add_argument(
        "--checkpoint-dir", type=Path, default=Path("outputs/checkpoints")
    )
    train_classifier.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/reports/mll23_head_comparison.json"),
    )
    train_classifier.set_defaults(handler=_train_classifier)

    embeddings = subcommands.add_parser("embeddings", help="Stage A embedding cache workflows")
    embeddings_subcommands = embeddings.add_subparsers(required=True)

    embeddings_extract = embeddings_subcommands.add_parser(
        "extract", help="Extract frozen-backbone embeddings over a split manifest"
    )
    embeddings_extract.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    embeddings_extract.add_argument(
        "--splits", type=Path, default=Path("data/manifests/mll23_splits.csv")
    )
    embeddings_extract.add_argument("--limit", type=int, default=None)
    embeddings_extract.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"]
    )
    embeddings_extract.add_argument(
        "--output", type=Path, default=Path("data/embeddings/mll23_embeddings.pt")
    )
    embeddings_extract.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/reports/mll23_embeddings.json"),
    )
    embeddings_extract.set_defaults(handler=_embeddings_extract)

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
        "dinobloom_b": _dinobloom_smoke_report(config.classifier.weights, args.image),
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


def _dinobloom_smoke_report(weight_path: Path, sample_image: Path | None) -> dict[str, object]:
    try:
        return dict(asdict(smoke_dinobloom_b(weight_path, sample_image=sample_image)))
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


def _config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    head = config.classifier.head or "linear"
    if head.strip().lower() not in HEAD_NAMES:
        raise ConfigError(f"Unknown classifier head {head!r}; expected one of {HEAD_NAMES}")
    if config.classifier.num_classes < 1:
        raise ConfigError("classifier.num_classes must be positive")
    summary = {
        "config": str(args.config),
        "backbone": config.classifier.backbone,
        "head": config.classifier.head,
        "num_classes": config.classifier.num_classes,
        "weights": str(config.classifier.weights),
        "class_names_file": str(config.classifier.class_names_file),
        "mll23_root": str(config.dataset.mll23_root),
        "seed": config.project.seed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _train_classifier(args: argparse.Namespace) -> int:
    from bloodfilm.classification.embeddings import load_embedding_cache
    from bloodfilm.schemas import MLL23_CANONICAL_CLASSES
    from bloodfilm.training.classifier import TrainConfig, run_head_training
    from bloodfilm.util import sha256_file

    config = load_config(args.config)
    require_complete_dataset(config.dataset.mll23_root, dataset="mll23")
    require_dinobloom_b_weights(config.classifier.weights)
    head = args.head or config.classifier.head or "linear"
    requested = list(HEAD_NAMES) if head == "all" else [head]
    for name in requested:
        if name not in HEAD_NAMES:
            raise ConfigError(f"Unknown classifier head {name!r}; expected one of {HEAD_NAMES}")
    if config.classifier.num_classes < 1:
        raise ConfigError("classifier.num_classes must be positive")
    cache = load_embedding_cache(args.embeddings)
    weight_sha256 = sha256_file(config.classifier.weights)
    if cache.metadata.get("weight_sha256") != weight_sha256:
        raise ConfigError(
            "Embedding cache was extracted with different backbone weights; "
            "re-run embeddings extract before training"
        )
    class_names = list(MLL23_CANONICAL_CLASSES[: config.classifier.num_classes])
    train_config = TrainConfig(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    heads: dict[str, Any] = {}
    for name in requested:
        run = run_head_training(
            cache,
            name,
            config.classifier.num_classes,
            class_names,
            train_config,
            eval_split=args.evaluate_split,
        )
        checkpoint_path = args.checkpoint_dir / f"{name}.pt"
        _save_head_checkpoint(checkpoint_path, name, config, class_names, train_config, run, cache)
        heads[name] = {
            "checkpoint": str(checkpoint_path),
            "eval_split": run["eval_split"],
            "eval_size": run["eval_size"],
            "macro_f1": run["eval_report"]["macro_f1"],
            "accuracy": run["eval_report"]["accuracy"],
            "balanced_accuracy": run["eval_report"]["balanced_accuracy"],
            "top2_accuracy": run["eval_top2_accuracy"],
            "ece_before": run["eval_ece_before"],
            "temperature": run["temperature"],
            "ece_after": run["eval_ece_after"],
            "final_loss": run["final_loss"],
            "epochs": run["epochs"],
        }
    selected = max(heads, key=lambda name: heads[name]["macro_f1"])
    write_json(
        args.report_output,
        {
            "config": str(args.config),
            "embeddings": str(args.embeddings),
            "weight_sha256": weight_sha256,
            "preprocessing_sha256": cache.metadata.get("preprocessing_sha256"),
            "heads": heads,
            "selected": selected,
            "selection_metric": "macro_f1",
        },
    )
    print(args.report_output)
    return 0


def _save_head_checkpoint(
    path: Path,
    head: str,
    config: Any,
    class_names: list[str],
    train_config: Any,
    run: Any,
    cache: Any,
) -> None:
    from bloodfilm.ml import require_torch

    torch = require_torch("Head training")
    torch.save(
        {
            "head": head,
            "num_classes": config.classifier.num_classes,
            "class_names": class_names,
            "train_config": {
                "epochs": train_config.epochs,
                "learning_rate": train_config.learning_rate,
                "weight_decay": train_config.weight_decay,
                "imbalance": train_config.imbalance,
                "seed": train_config.seed,
            },
            "eval_split": run["eval_split"],
            "temperature": run["temperature"],
            "cache_metadata": cache.metadata,
            "model_state": run["model_state"],
        },
        path,
    )


def _embeddings_extract(args: argparse.Namespace) -> int:
    from bloodfilm.data import read_split_manifest
    from bloodfilm.util import sha256_file

    config = load_config(args.config)
    require_complete_dataset(config.dataset.mll23_root, dataset="mll23")
    require_dinobloom_b_weights(config.classifier.weights)
    rows = read_split_manifest(args.splits)
    backbone = load_dinobloom_b_backbone(config.classifier.weights, device=_resolve_device(args))
    cache = build_embedding_cache(
        rows,
        dataset_root=config.dataset.mll23_root,
        backbone=backbone,
        preprocess=preprocess_crop,
        weight_sha256=sha256_file(config.classifier.weights),
        limit=args.limit,
        device=_resolve_device(args),
    )
    save_embedding_cache(cache, args.output)
    write_json(
        args.report_output,
        {
            "cache": str(args.output),
            "splits": str(args.splits),
            "weights": str(config.classifier.weights),
            **cache.metadata,
        },
    )
    print(args.report_output)
    return 0


def _resolve_device(args: argparse.Namespace) -> str:
    if args.device == "auto":
        from bloodfilm.ml import require_torch

        torch = require_torch("Embedding extraction")
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(args.device)


def _split(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    split_rows = create_split_manifest(rows, seed=args.seed)
    write_split_manifest(args.output, split_rows)
    write_json(args.report_output, create_leakage_report(split_rows))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
