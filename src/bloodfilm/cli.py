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
from bloodfilm.classification.parity import DEFAULT_PARITY_CROPS
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
from bloodfilm.errors import BloodFilmError, ConfigError, InputNotFoundError
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
    audit.add_argument("--format", choices=["auto", "mll23", "txl-pbc"], default="auto")
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

    detector = subcommands.add_parser("detector", help="Detector dataset and bundle workflows")
    detector_subcommands = detector.add_subparsers(required=True)

    detector_prepare = detector_subcommands.add_parser(
        "prepare-yolo", help="Create a WBC-only YOLO derivative from TXL-PBC"
    )
    detector_prepare.add_argument("--dataset-root", type=Path, required=True)
    detector_prepare.add_argument("--output-root", type=Path, required=True)
    detector_prepare.add_argument("--wbc-class-id", type=int, required=True)
    detector_prepare.add_argument("--report-output", type=Path, required=True)
    detector_prepare.set_defaults(handler=_detector_prepare_yolo)

    detector_preprocess = detector_subcommands.add_parser(
        "preprocess", help="Detect microscope-field ROI preprocessing metadata"
    )
    detector_preprocess.add_argument("--input", type=Path, required=True)
    detector_preprocess.add_argument("--dark-threshold", type=int, default=20)
    detector_preprocess.add_argument("--minimum-area-ratio", type=float, default=0.30)
    detector_preprocess.add_argument("--padding-ratio", type=float, default=0.02)
    detector_preprocess.add_argument("--report-output", type=Path, required=True)
    detector_preprocess.set_defaults(handler=_detector_preprocess)

    detector_audit = detector_subcommands.add_parser(
        "audit", help="Audit detector annotation directories"
    )
    detector_audit.add_argument(
        "--format", choices=["leukemia-attri-yolo", "standard-yolo"], required=True
    )
    detector_audit.add_argument("--image-root", type=Path, required=True)
    detector_audit.add_argument("--label-root", type=Path, required=True)
    detector_audit.add_argument("--class-names", action="append", default=[])
    detector_audit.add_argument("--class-mapping", action="append", default=[])
    detector_audit.add_argument(
        "--annotation-completeness",
        choices=["fully_annotated", "sparsely_annotated", "unknown"],
        default="fully_annotated",
    )
    detector_audit.add_argument("--report-output", type=Path, required=True)
    detector_audit.set_defaults(handler=_detector_audit)

    detector_package = detector_subcommands.add_parser(
        "package-bundle", help="Package trained detector weights and reports"
    )
    detector_package.add_argument("--weights", type=Path, required=True)
    detector_package.add_argument(
        "--config", type=Path, default=Path("configs/detector_txl_pbc.yaml")
    )
    detector_package.add_argument("--metrics", type=Path, required=True)
    detector_package.add_argument("--thresholds", type=Path, required=True)
    detector_package.add_argument("--audit-report", type=Path, required=True)
    detector_package.add_argument(
        "--bundle", type=Path, default=Path("models/txl-pbc-yolo26n-v0.1")
    )
    detector_package.add_argument("--name", type=str, default="txl-pbc-yolo26n-v0.1")
    detector_package.set_defaults(handler=_detector_package_bundle)

    detector_inspect = detector_subcommands.add_parser(
        "inspect-bundle", help="Inspect a packaged detector bundle"
    )
    detector_inspect.add_argument("--bundle", type=Path, required=True)
    detector_inspect.add_argument("--output", type=Path, default=None)
    detector_inspect.set_defaults(handler=_detector_inspect_bundle)

    detector_sweep = detector_subcommands.add_parser(
        "sweep", help="Sweep detector confidence thresholds over field images"
    )
    detector_sweep.add_argument("--weights", type=Path, required=True)
    detector_sweep.add_argument("--input", type=Path, required=True)
    detector_sweep.add_argument("--thresholds", type=str, default="0.05,0.10,0.15,0.25,0.35,0.50")
    detector_sweep.add_argument(
        "--mode",
        choices=["full_field", "field_roi", "tiled", "field_roi_tiled"],
        default="full_field",
    )
    detector_sweep.add_argument("--iou", type=float, default=0.5)
    detector_sweep.add_argument("--imgsz", type=int, default=640)
    detector_sweep.add_argument("--labels", type=Path, default=None)
    detector_sweep.add_argument("--target-recall", type=float, default=0.95)
    detector_sweep.add_argument("--max-false-positives-per-image", type=float, default=2.0)
    detector_sweep.add_argument("--report-output", type=Path, required=True)
    detector_sweep.set_defaults(handler=_detector_sweep)

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

    classifier = subcommands.add_parser(
        "classifier", help="Classifier bundle and inference workflows"
    )
    classifier_subcommands = classifier.add_subparsers(required=True)

    inspect_bundle = classifier_subcommands.add_parser(
        "inspect-bundle", help="Inspect a packaged classifier bundle"
    )
    inspect_bundle.add_argument("--bundle", type=Path, required=True)
    inspect_bundle.add_argument("--output", type=Path, default=None)
    inspect_bundle.set_defaults(handler=_classifier_inspect_bundle)

    package_bundle = classifier_subcommands.add_parser(
        "package-bundle", help="Package a trained classifier head as a research-only bundle"
    )
    package_bundle.add_argument("--checkpoint", type=Path, required=True)
    package_bundle.add_argument(
        "--comparison-report",
        type=Path,
        default=Path("outputs/reports/mll23_head_comparison.json"),
    )
    package_bundle.add_argument(
        "--thresholds", type=Path, default=Path("configs/classifier_uncertainty.yaml")
    )
    package_bundle.add_argument(
        "--test-report", type=Path, default=Path("outputs/reports/mll23_test_evaluation.json")
    )
    package_bundle.add_argument(
        "--bundle", type=Path, default=Path("models/mll23-dinobloom-b-mlp-v0.1")
    )
    package_bundle.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/mll23_manifest.csv")
    )
    package_bundle.add_argument(
        "--invalid-manifest", type=Path, default=Path("data/manifests/mll23_invalid.csv")
    )
    package_bundle.set_defaults(handler=_classifier_package_bundle)

    validation_reports = classifier_subcommands.add_parser(
        "validation-reports", help="Write post-training classifier validation reports"
    )
    validation_reports.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/mll23_manifest.csv")
    )
    validation_reports.add_argument(
        "--splits", type=Path, default=Path("data/manifests/mll23_splits.csv")
    )
    validation_reports.add_argument(
        "--invalid-manifest", type=Path, default=Path("data/manifests/mll23_invalid.csv")
    )
    validation_reports.add_argument(
        "--embeddings-report", type=Path, default=Path("outputs/reports/mll23_embeddings.json")
    )
    validation_reports.add_argument(
        "--comparison-report", type=Path, default=Path("outputs/reports/mll23_head_comparison.json")
    )
    validation_reports.add_argument(
        "--bundle-inspection",
        type=Path,
        default=Path("outputs/reports/mll23_bundle_inspection.json"),
    )
    validation_reports.add_argument(
        "--thresholds", type=Path, default=Path("configs/classifier_uncertainty.yaml")
    )
    validation_reports.add_argument(
        "--test-evaluation", type=Path, default=Path("outputs/reports/mll23_test_evaluation.json")
    )
    validation_reports.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    validation_reports.add_argument("--parity-evidence", type=Path, default=None)
    validation_reports.add_argument("--verification-report", type=Path, default=None)
    validation_reports.set_defaults(handler=_classifier_validation_reports)

    evaluate_cache = classifier_subcommands.add_parser(
        "evaluate-cache", help="Evaluate a trained head on one cached embedding split"
    )
    evaluate_cache.add_argument("--checkpoint", type=Path, required=True)
    evaluate_cache.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings/mll23_embeddings.pt")
    )
    evaluate_cache.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    evaluate_cache.add_argument("--output", type=Path, required=True)
    evaluate_cache.add_argument("--artifacts-dir", type=Path, default=Path("outputs/reports"))
    evaluate_cache.add_argument(
        "--thresholds", type=Path, default=Path("configs/classifier_uncertainty.yaml")
    )
    evaluate_cache.set_defaults(handler=_classifier_evaluate_cache)

    parity_live = classifier_subcommands.add_parser(
        "parity-live", help="Check live backbone/head outputs against cached embeddings"
    )
    parity_live.add_argument(
        "--weights", type=Path, default=Path("models/backbones/dinobloom-b.pth")
    )
    parity_live.add_argument("--dataset-root", type=Path, default=Path("data/raw/MLL23"))
    parity_live.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings/mll23_embeddings.pt")
    )
    parity_live.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/mlp.pt"))
    parity_live.add_argument("--max-crops", type=int, default=DEFAULT_PARITY_CROPS)
    parity_live.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parity_live.add_argument("--output", type=Path, required=True)
    parity_live.set_defaults(handler=_classifier_parity_live)

    classify_crop = classifier_subcommands.add_parser(
        "classify-crop", help="Classify one already-cropped WBC image"
    )
    classify_crop.add_argument("--bundle", type=Path, required=True)
    classify_crop.add_argument("--image", type=Path, required=True)
    classify_crop.add_argument(
        "--backbone-weights", type=Path, default=Path("models/backbones/dinobloom-b.pth")
    )
    classify_crop.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    classify_crop.add_argument("--output", type=Path, required=True)
    classify_crop.set_defaults(handler=_classifier_classify_crop)

    classify_folder = classifier_subcommands.add_parser(
        "classify-folder", help="Classify every image crop in a folder"
    )
    classify_folder.add_argument("--bundle", type=Path, required=True)
    classify_folder.add_argument("--input", type=Path, required=True)
    classify_folder.add_argument(
        "--backbone-weights", type=Path, default=Path("models/backbones/dinobloom-b.pth")
    )
    classify_folder.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    classify_folder.add_argument("--output", type=Path, required=True)
    classify_folder.set_defaults(handler=_classifier_classify_folder)

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
    if args.format == "txl-pbc" or (args.format == "auto" and args.name.lower() == "txl-pbc"):
        from bloodfilm.data.txl_pbc import audit_txl_pbc_dataset

        report = audit_txl_pbc_dataset(dataset_root)
    else:
        report = audit_dataset(dataset_root)
    write_json(args.report_output, report)
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


def _detector_prepare_yolo(args: argparse.Namespace) -> int:
    from bloodfilm.detection.dataset import prepare_wbc_yolo_dataset

    report = prepare_wbc_yolo_dataset(
        args.dataset_root, args.output_root, wbc_class_id=args.wbc_class_id
    )
    write_json(args.report_output, report)
    print(args.report_output)
    return 0


def _detector_preprocess(args: argparse.Namespace) -> int:
    from bloodfilm.detection.roi import detect_field_roi

    roi = detect_field_roi(
        args.input,
        dark_threshold=args.dark_threshold,
        minimum_area_ratio=args.minimum_area_ratio,
        padding_ratio=args.padding_ratio,
    )
    report = {
        "schema_version": 1,
        "research_only": True,
        "input": str(args.input),
        "roi": roi.to_dict(),
    }
    write_json(args.report_output, report)
    print(args.report_output)
    return 0


def _detector_audit(args: argparse.Namespace) -> int:
    from bloodfilm.detection.annotations import audit_yolo_directory

    report = audit_yolo_directory(
        image_root=args.image_root,
        label_root=args.label_root,
        class_names=_parse_int_mapping(args.class_names),
        class_mapping=_parse_str_mapping(args.class_mapping),
        annotation_completeness=args.annotation_completeness,
    )
    report["format"] = args.format
    write_json(args.report_output, report)
    print(args.report_output)
    return 0


def _detector_package_bundle(args: argparse.Namespace) -> int:
    from bloodfilm.detection.bundle import write_detector_bundle

    write_detector_bundle(
        args.bundle,
        weights=args.weights,
        config=args.config,
        metrics=args.metrics,
        thresholds=args.thresholds,
        audit_report=args.audit_report,
        model_name=args.name,
    )
    print(args.bundle)
    return 0


def _detector_inspect_bundle(args: argparse.Namespace) -> int:
    from bloodfilm.detection.bundle import inspect_detector_bundle

    report = inspect_detector_bundle(args.bundle)
    if args.output is not None:
        write_json(args.output, report)
        print(args.output)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _detector_sweep(args: argparse.Namespace) -> int:
    from bloodfilm.detection.detector import Detection
    from bloodfilm.detection.sweep import (
        collect_images,
        load_yolo_truth,
        parse_thresholds,
        run_prediction_sweep,
        score_thresholds_with_truth,
        write_sweep_report,
    )

    thresholds = parse_thresholds(args.thresholds)
    images = collect_images(args.input)
    report = run_prediction_sweep(
        args.weights,
        images,
        thresholds,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        preprocessing_mode=args.mode,
    )
    if args.labels is not None:
        # YOLO filtering is monotonic in confidence, so the lowest-threshold
        # candidate holds the full prediction set for rescoring.
        widest = min(report["candidates"], key=lambda row: row["confidence_threshold"])
        full_predictions: dict[str, list[Detection]] = {
            image_stem: [
                Detection(
                    x1=box["x1"],
                    y1=box["y1"],
                    x2=box["x2"],
                    y2=box["y2"],
                    score=box["score"],
                    label="wbc_candidate",
                )
                for box in boxes
            ]
            for image_stem, boxes in widest["boxes_by_image"].items()
        }
        truth = load_yolo_truth(args.labels, images)
        report["scoring"] = score_thresholds_with_truth(
            full_predictions,
            truth,
            thresholds,
            iou_threshold=args.iou,
            target_recall=args.target_recall,
            max_false_positives_per_image=args.max_false_positives_per_image,
        )
    write_sweep_report(report, args.report_output)
    print(args.report_output)
    return 0


def _parse_str_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ConfigError(f"Expected KEY=VALUE mapping, got {value!r}")
        key, mapped = value.split("=", 1)
        mapping[key] = mapped
    return mapping


def _parse_int_mapping(values: list[str]) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for key, value in _parse_str_mapping(values).items():
        try:
            parsed[int(key)] = value
        except ValueError as exc:
            raise ConfigError(f"Expected integer mapping key, got {key!r}") from exc
    return parsed


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


def _classifier_inspect_bundle(args: argparse.Namespace) -> int:
    from bloodfilm.classification.bundle import inspect_bundle

    report = inspect_bundle(args.bundle)
    if args.output is not None:
        write_json(args.output, report)
        print(args.output)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _classifier_package_bundle(args: argparse.Namespace) -> int:
    from bloodfilm.classification.bundle import write_bundle_metadata
    from bloodfilm.classification.uncertainty import UncertaintyPolicy
    from bloodfilm.documents import load_config_document
    from bloodfilm.ml import require_torch

    torch = require_torch("Classifier bundle packaging")
    comparison = load_config_document(args.comparison_report)
    selected = str(comparison["selected"])
    head_metrics = comparison["heads"][selected]
    thresholds = load_config_document(args.thresholds)
    test_report = load_config_document(args.test_report) if args.test_report.exists() else None
    if not args.checkpoint.exists():
        raise InputNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = _checkpoint_class_names(checkpoint)
    if checkpoint.get("head") != selected:
        raise ConfigError("Checkpoint head does not match selected comparison head")
    if "model_state" not in checkpoint:
        raise ConfigError("Checkpoint is missing model_state")
    metrics: dict[str, object] = {
        "selection_report": str(args.comparison_report),
        "selection_metric": comparison.get("selection_metric", "macro_f1"),
        "validation": head_metrics,
        "all_heads": comparison["heads"],
    }
    if test_report is not None:
        metrics["test"] = test_report
    write_bundle_metadata(
        args.bundle,
        head_checkpoint=args.checkpoint,
        class_names=class_names,
        head_name=selected,
        backbone_sha256=str(comparison["weight_sha256"]),
        preprocessing_sha256=str(comparison["preprocessing_sha256"]),
        temperature=float(head_metrics["temperature"]),
        validation_metrics=metrics,
        uncertainty_policy=UncertaintyPolicy(**thresholds),
        dataset_metadata=_dataset_metadata(args.manifest, args.invalid_manifest),
    )
    print(args.bundle)
    return 0


def _classifier_validation_reports(args: argparse.Namespace) -> int:
    from bloodfilm.classification.reports import write_post_training_reports

    paths = write_post_training_reports(
        manifest=args.manifest,
        splits=args.splits,
        invalid_manifest=args.invalid_manifest,
        embeddings_report=args.embeddings_report,
        comparison_report=args.comparison_report,
        bundle_inspection=args.bundle_inspection,
        thresholds=args.thresholds,
        test_evaluation=args.test_evaluation,
        parity_evidence=args.parity_evidence,
        verification_report=args.verification_report,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


def _classifier_evaluate_cache(args: argparse.Namespace) -> int:
    from bloodfilm.classification.embeddings import load_embedding_cache
    from bloodfilm.classification.heads import build_head
    from bloodfilm.classification.reports import write_evaluation_artifacts
    from bloodfilm.classification.uncertainty import UncertaintyPolicy
    from bloodfilm.documents import load_config_document
    from bloodfilm.errors import InputNotFoundError
    from bloodfilm.evaluation.classifier import expected_calibration_error
    from bloodfilm.ml import require_torch
    from bloodfilm.training.classifier import cache_split_tensors, evaluate_head

    if not args.checkpoint.exists():
        raise InputNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.embeddings.exists():
        raise InputNotFoundError(f"Embedding cache not found: {args.embeddings}")
    torch = require_torch("Classifier cache evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cache = load_embedding_cache(args.embeddings)
    ok_entries = [entry for entry in cache.entries if entry.status == "ok"]
    if tuple(cache.embeddings.shape) != (len(ok_entries), 768):
        raise ConfigError("Embedding cache shape does not match ok entries and 768 dimensions")
    if not bool(torch.isfinite(cache.embeddings).all()):
        raise ConfigError("Embedding cache contains NaN or infinite values")
    head_name = str(checkpoint["head"])
    class_names = [str(name) for name in checkpoint["class_names"]]
    model = build_head(head_name, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    embeddings, labels = cache_split_tensors(cache, args.split)
    evaluation = evaluate_head(model, embeddings, labels, class_names)
    temperature = float(checkpoint.get("temperature", 1.0))
    probabilities_before = torch.softmax(evaluation["logits"], dim=1).tolist()
    scaled = torch.softmax(evaluation["logits"] / temperature, dim=1).tolist()
    split_entries = [entry for entry in ok_entries if entry.split == args.split]
    if len(split_entries) != len(labels):
        raise ConfigError("Embedding labels do not align with split entries")
    report = {
        "checkpoint": str(args.checkpoint),
        "embeddings": str(args.embeddings),
        "split": args.split,
        "head": head_name,
        "size": len(labels),
        "temperature": temperature,
        "metrics": evaluation["report"],
        "top2_accuracy": evaluation["top2_accuracy"],
        "ece_before": evaluation["ece"],
        "ece_after": expected_calibration_error(scaled, labels, bins=15),
    }
    write_json(args.output, report)
    write_evaluation_artifacts(
        output_dir=args.artifacts_dir,
        class_names=class_names,
        labels=labels,
        image_ids=[entry.image_id for entry in split_entries],
        image_paths=[entry.image_path for entry in split_entries],
        probabilities_before=[[float(value) for value in row] for row in probabilities_before],
        probabilities_after=[[float(value) for value in row] for row in scaled],
        temperature=temperature,
        evaluation_split=args.split,
        temperature_fit_split=str(checkpoint.get("eval_split", "validation")),
        uncertainty_policy=UncertaintyPolicy(**load_config_document(args.thresholds)),
    )
    print(args.output)
    return 0


def _classifier_parity_live(args: argparse.Namespace) -> int:
    from bloodfilm.classification.parity import run_live_parity

    evidence = run_live_parity(
        weights=args.weights,
        dataset_root=args.dataset_root,
        cache_path=args.embeddings,
        checkpoint_path=args.checkpoint,
        max_crops=args.max_crops,
        device=args.device,
    )
    write_json(args.output, evidence)
    print(args.output)
    return 0


def _classifier_classify_crop(args: argparse.Namespace) -> int:
    from bloodfilm.classification.inference import classify_crop

    prediction = classify_crop(
        bundle_dir=args.bundle,
        image_path=args.image,
        backbone_weights=args.backbone_weights,
        device=args.device,
    )
    write_json(args.output, prediction)
    print(args.output)
    return 0


def _classifier_classify_folder(args: argparse.Namespace) -> int:
    from bloodfilm.classification.inference import classify_crop

    if not args.input.exists():
        raise ConfigError(f"Input folder not found: {args.input}")
    if not args.input.is_dir():
        raise ConfigError(f"Input path is not a folder: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = [path for path in sorted(args.input.iterdir()) if path.is_file()]
    written: list[str] = []
    for image_path in image_paths:
        try:
            prediction = classify_crop(
                bundle_dir=args.bundle,
                image_path=image_path,
                backbone_weights=args.backbone_weights,
                device=args.device,
            )
        except BloodFilmError as exc:
            prediction = {
                "research_only": True,
                "image": str(image_path),
                "decision": {"status": "unknown", "label": None, "reasons": [exc.code]},
                "error": {"code": exc.code, "message": str(exc)},
            }
        output = args.output / f"{image_path.stem}.json"
        write_json(output, prediction)
        written.append(str(output))
    write_json(args.output / "index.json", {"count": len(written), "predictions": written})
    print(args.output)
    return 0


def _checkpoint_class_names(checkpoint: object) -> list[str]:
    if not isinstance(checkpoint, dict):
        raise ConfigError("Checkpoint must be a metadata dictionary")
    raw = checkpoint.get("class_names")
    if not isinstance(raw, list) or not all(isinstance(name, str) for name in raw):
        raise ConfigError("Checkpoint is missing class_names")
    if not raw:
        raise ConfigError("Checkpoint class_names must not be empty")
    return list(raw)


def _dataset_metadata(manifest: Path, invalid_manifest: Path) -> dict[str, object]:
    from bloodfilm.util import sha256_file

    metadata: dict[str, object] = {"name": "MLL23"}
    if manifest.exists():
        metadata["valid_image_count"] = _csv_data_row_count(manifest)
        metadata["manifest_sha256"] = sha256_file(manifest)
    if invalid_manifest.exists():
        metadata["invalid_item_count"] = _csv_data_row_count(invalid_manifest)
        metadata["invalid_manifest_sha256"] = sha256_file(invalid_manifest)
    return metadata


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


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
