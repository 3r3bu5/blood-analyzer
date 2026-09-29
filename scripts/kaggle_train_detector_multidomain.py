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
    parser.add_argument(
        "--experiment",
        choices=["all", "A_finetune_txl_pbc", "B_clean_pretrained"],
        default="all",
        help="Run both experiments by default, or only one selected experiment.",
    )
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
        experiment=args.experiment,
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
        val_command = _val_command_with_existing_weights(command)
        subprocess.run(val_command.split(), check=True)
    return 0


def _val_command_with_existing_weights(command: dict[str, object]) -> str:
    val_command = str(command["val_command"])
    planned_weights = str(command["expected_best_weights"])
    if Path(planned_weights).exists():
        return val_command
    yolo_runs_weights = str(command.get("yolo_runs_best_weights", ""))
    if yolo_runs_weights and Path(yolo_runs_weights).exists():
        return val_command.replace(f"model={planned_weights}", f"model={yolo_runs_weights}")
    return val_command


if __name__ == "__main__":
    raise SystemExit(main())
