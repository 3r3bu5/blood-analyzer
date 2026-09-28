from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bloodfilm.documents import load_config_document, write_json
from bloodfilm.errors import ConfigError


@dataclass(frozen=True)
class DetectorExperiment:
    name: str
    initial_weights: str
    purpose: str
    output_name: str


def build_kaggle_training_plan(
    *,
    config_path: Path | str,
    data_yaml: Path | str,
    project_dir: Path | str,
    epochs: int | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build reproducible Ultralytics commands for the two M3.1 detector experiments."""
    config = load_config_document(config_path)
    detector = _object(config, "detector")
    training = _object(config, "training")
    experiments = _experiments(config)
    selected_epochs = int(epochs if epochs is not None else training.get("epochs", 100))
    selected_image_size = int(
        image_size if image_size is not None else detector.get("image_size", 640)
    )
    selected_batch = batch_size if batch_size is not None else training.get("batch_size")
    commands = []
    output_root = Path(project_dir)
    for experiment in experiments:
        train_parts = [
            "yolo",
            "detect",
            "train",
            f"model={experiment.initial_weights}",
            f"data={data_yaml}",
            f"imgsz={selected_image_size}",
            f"epochs={selected_epochs}",
            f"project={output_root}",
            f"name={experiment.output_name}",
            "exist_ok=True",
            "seed=42",
        ]
        if selected_batch is not None:
            train_parts.append(f"batch={int(selected_batch)}")
        val_parts = [
            "yolo",
            "detect",
            "val",
            f"model={output_root / experiment.output_name / 'weights' / 'best.pt'}",
            f"data={data_yaml}",
            "split=test",
            f"imgsz={selected_image_size}",
            f"project={output_root}",
            f"name={experiment.output_name}-test",
            "exist_ok=True",
        ]
        commands.append(
            {
                "experiment": experiment.name,
                "purpose": experiment.purpose,
                "initial_weights": experiment.initial_weights,
                "train_command": " ".join(str(part) for part in train_parts),
                "val_command": " ".join(str(part) for part in val_parts),
                "expected_best_weights": str(
                    output_root / experiment.output_name / "weights" / "best.pt"
                ),
                "expected_test_dir": str(output_root / f"{experiment.output_name}-test"),
            }
        )
    return {
        "schema_version": 1,
        "research_only": True,
        "status": "planned_not_trained",
        "config": str(config_path),
        "data_yaml": str(data_yaml),
        "project_dir": str(project_dir),
        "epochs": selected_epochs,
        "image_size": selected_image_size,
        "batch_size": selected_batch,
        "commands": commands,
        "artifacts_to_download": [
            str(output_root),
            "outputs/reports/detector_multidomain_kaggle_plan.json",
            "outputs/reports/detector_multidomain_comparison.json",
            "outputs/reports/detector_multidomain_evaluation.json",
            "outputs/reports/detector_target_smoke.json",
        ],
        "blocked_until": [
            "LeukemiaAttri license and source layout are verified",
            "unified fully annotated YOLO data.yaml exists",
            "CUDA runner is available",
        ],
    }


def write_kaggle_training_plan(plan: dict[str, Any], output: Path | str) -> Path:
    output_path = Path(output)
    write_json(output_path, plan)
    return output_path


def _object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _experiments(config: dict[str, Any]) -> list[DetectorExperiment]:
    raw = _object(config, "experiments")
    required = ["A_finetune_txl_pbc", "B_clean_pretrained"]
    experiments: list[DetectorExperiment] = []
    for name in required:
        value = raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"experiments.{name} must be configured")
        initial_weights = str(value.get("initial_weights", ""))
        if not initial_weights:
            raise ConfigError(f"experiments.{name}.initial_weights is required")
        experiments.append(
            DetectorExperiment(
                name=name,
                initial_weights=initial_weights,
                purpose=str(value.get("purpose", "")),
                output_name=_experiment_output_name(name),
            )
        )
    return experiments


def _experiment_output_name(name: str) -> str:
    return "wbc-detector-multidomain-yolo26n-" + name.split("_", 1)[0].lower()
