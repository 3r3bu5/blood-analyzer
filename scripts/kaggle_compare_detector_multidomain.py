from __future__ import annotations

import argparse
from pathlib import Path

from bloodfilm.detection.comparison import compare_kaggle_detector_runs, require_plan_exists


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare M3.1 Kaggle detector experiments without selecting a winner."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("outputs/reports/detector_multidomain_kaggle_plan.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/detector_multidomain_comparison.json"),
    )
    args = parser.parse_args()

    require_plan_exists(args.plan)
    compare_kaggle_detector_runs(plan_path=args.plan, output_path=args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
