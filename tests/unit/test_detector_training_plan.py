from pathlib import Path

from bloodfilm.detection.training_plan import build_kaggle_training_plan


def test_kaggle_training_plan_contains_two_required_experiments() -> None:
    plan = build_kaggle_training_plan(
        config_path=Path("configs/detector_multidomain.yaml"),
        data_yaml=Path("data/detection/multidomain/data.yaml"),
        project_dir=Path("outputs/detector_multidomain"),
        epochs=2,
        batch_size=4,
    )

    names = [command["experiment"] for command in plan["commands"]]
    assert names == ["A_finetune_txl_pbc", "B_clean_pretrained"]
    assert "models/txl-pbc-yolo26n-v0.1/weights.pt" in plan["commands"][0]["train_command"]
    assert "model=yolo26n.pt" in plan["commands"][1]["train_command"]
    assert "epochs=2" in plan["commands"][0]["train_command"]
    assert "batch=4" in plan["commands"][0]["train_command"]
    assert plan["status"] == "planned_not_trained"


def test_kaggle_training_plan_records_artifacts_to_download() -> None:
    plan = build_kaggle_training_plan(
        config_path=Path("configs/detector_multidomain.yaml"),
        data_yaml=Path("data/detection/multidomain/data.yaml"),
        project_dir=Path("outputs/detector_multidomain"),
    )

    assert "outputs/detector_multidomain" in plan["artifacts_to_download"]
    assert plan["commands"][0]["expected_best_weights"].endswith("weights/best.pt")
    assert plan["commands"][0]["yolo_runs_best_weights"].endswith("weights/best.pt")
    assert plan["commands"][0]["yolo_runs_best_weights"].startswith("runs/detect/")


def test_kaggle_training_plan_can_select_experiment_b_only() -> None:
    plan = build_kaggle_training_plan(
        config_path=Path("configs/detector_multidomain.yaml"),
        data_yaml=Path("data/detection/multidomain/data.yaml"),
        project_dir=Path("outputs/detector_multidomain"),
        experiment="B_clean_pretrained",
    )

    assert plan["experiment"] == "B_clean_pretrained"
    assert [command["experiment"] for command in plan["commands"]] == ["B_clean_pretrained"]
    assert "model=yolo26n.pt" in plan["commands"][0]["train_command"]
    assert "wbc-detector-multidomain-yolo26n-b" in plan["commands"][0]["train_command"]
