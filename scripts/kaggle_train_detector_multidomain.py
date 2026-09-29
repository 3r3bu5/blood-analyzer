from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bloodfilm.detection.training_plan import build_kaggle_training_plan, write_kaggle_training_plan


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run validation from existing weights "
        "(resume mode: reuses a previously trained best.pt without retraining).",
    )
    args = parser.parse_args(argv)

    plan = build_kaggle_training_plan(
        config_path=args.config,
        data_yaml=args.data_yaml,
        project_dir=args.project_dir,
        experiment=args.experiment,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
    )
    plan["train_skipped"] = args.skip_train
    write_kaggle_training_plan(plan, args.plan_output)
    print(args.plan_output)
    if args.dry_run:
        for command in plan["commands"]:
            if args.skip_train:
                print(f"SKIP train: {command['train_command']}")
            else:
                print(command["train_command"])
            print(command["val_command"])
        return 0
    for command in plan["commands"]:
        if args.skip_train:
            val_command = _require_existing_weights(command)
            subprocess.run(val_command.split(), check=True)
        else:
            subprocess.run(command["train_command"].split(), check=True)
            val_command = _val_command_with_existing_weights(command)
            subprocess.run(val_command.split(), check=True)
    return 0


def _require_existing_weights(command: dict[str, object]) -> str:
    """Resolve the val command against existing weights, failing fast if absent."""
    val_command = str(command["val_command"])
    planned_weights = str(command.get("expected_best_weights", ""))
    if planned_weights and Path(planned_weights).exists():
        return val_command
    yolo_runs_weights = str(command.get("yolo_runs_best_weights", ""))
    if yolo_runs_weights and Path(yolo_runs_weights).exists():
        return val_command.replace(f"model={planned_weights}", f"model={yolo_runs_weights}")
    experiment = str(command.get("experiment", "unknown"))
    raise SystemExit(
        f"Cannot skip train for {experiment}: no existing weights found.\n"
        f"Checked:\n  {planned_weights}\n  {yolo_runs_weights}\n"
        f"Restore best.pt first (e.g. from ../blood_data) or run without --skip-train."
    )


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
