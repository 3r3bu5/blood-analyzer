from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bloodfilm.detection.training_plan import build_kaggle_training_plan, write_kaggle_training_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or plan M3.1 multidomain detector training experiments on Kaggle."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/detector_multidomain.yaml"))
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path("outputs/detector_multidomain"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("outputs/reports/detector_multidomain_kaggle_plan.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_kaggle_training_plan(
        config_path=args.config,
        data_yaml=args.data_yaml,
        project_dir=args.project_dir,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
    )
    write_kaggle_training_plan(plan, args.plan_output)
    print(args.plan_output)
    if args.dry_run:
        for command in plan["commands"]:
            print(command["train_command"])
            print(command["val_command"])
        return 0
    for command in plan["commands"]:
        subprocess.run(command["train_command"].split(), check=True)
        subprocess.run(command["val_command"].split(), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
