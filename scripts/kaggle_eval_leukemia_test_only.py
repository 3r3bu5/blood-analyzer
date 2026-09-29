from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bloodfilm.detection.field_test import build_source_test_subset, score_weights_on_subset
from bloodfilm.documents import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score detector checkpoints on the LeukemiaAttri source-test "
        "field subset without retraining."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/detector_multidomain_manifest.csv"),
    )
    parser.add_argument(
        "--data-yaml", type=Path, default=Path("data/detection/multidomain/data.yaml")
    )
    parser.add_argument("--weights", type=Path, action="append", default=[])
    parser.add_argument("--source", type=str, default="LeukemiaAttri")
    parser.add_argument("--source-split", type=str, default="test")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/detector_leukemia_test_only"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/reports/detector_leukemia_test_only.json"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.weights:
        parser.error("Provide at least one --weights checkpoint.")

    subset = build_source_test_subset(
        manifest_path=args.manifest,
        data_yaml_path=args.data_yaml,
        output_root=args.output_root,
        source_dataset=args.source,
        source_split=args.source_split,
    )
    evaluations: list[dict[str, Any]] = []
    if not args.dry_run:
        for weights in args.weights:
            evaluations.append(
                score_weights_on_subset(
                    weights=weights,
                    data_yaml=subset["data_yaml"],
                    split="test",
                    image_size=args.imgsz,
                )
            )
    report = {
        **subset,
        "weights": [str(item) for item in args.weights],
        "image_size": args.imgsz,
        "evaluations": evaluations,
        "pairwise_deltas": _pairwise_deltas(evaluations),
    }
    write_json(args.report_output, report)
    print(args.report_output)
    for evaluation in evaluations:
        metrics = evaluation["metrics"]
        print(
            f"{evaluation['weights']}: "
            f"P={metrics.get('metrics/precision(B)', float('nan')):.3f} "
            f"R={metrics.get('metrics/recall(B)', float('nan')):.3f} "
            f"mAP50={metrics.get('metrics/mAP50(B)', float('nan')):.3f} "
            f"mAP50-95={metrics.get('metrics/mAP50-95(B)', float('nan')):.3f}"
        )
    return 0


def _pairwise_deltas(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evaluations) != 2:
        return {"status": "needs_exactly_two_checkpoints"}
    first, second = (item["metrics"] for item in evaluations)
    keys = sorted(set(first) & set(second))
    return {
        "status": "ok",
        "first": evaluations[0]["weights"],
        "second": evaluations[1]["weights"],
        "delta_second_minus_first": {
            key: round(float(second[key]) - float(first[key]), 5) for key in keys
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
