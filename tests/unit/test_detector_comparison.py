import json
from pathlib import Path

from bloodfilm.detection.comparison import compare_kaggle_detector_runs
from bloodfilm.documents import write_json


def test_compare_kaggle_detector_runs_writes_comparison_without_selection(
    tmp_path: Path,
) -> None:
    project = tmp_path / "outputs" / "detector_multidomain"
    plan = tmp_path / "outputs" / "reports" / "plan.json"
    output = tmp_path / "outputs" / "reports" / "comparison.json"
    _write_run(project / "wbc-detector-multidomain-yolo26n-a", recall=0.8, map50=0.7)
    _write_run(project / "wbc-detector-multidomain-yolo26n-a-test", recall=0.82, map50=0.72)
    _write_run(project / "wbc-detector-multidomain-yolo26n-b", recall=0.9, map50=0.75)
    _write_run(project / "wbc-detector-multidomain-yolo26n-b-test", recall=0.91, map50=0.77)
    write_json(
        plan,
        {
            "schema_version": 1,
            "data_yaml": "data/detection/multidomain/data.yaml",
            "commands": [
                _command(project, "A_finetune_txl_pbc", "wbc-detector-multidomain-yolo26n-a"),
                _command(project, "B_clean_pretrained", "wbc-detector-multidomain-yolo26n-b"),
            ],
        },
    )

    report = compare_kaggle_detector_runs(plan_path=plan, output_path=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "comparison_only"
    assert payload["selection"] is None
    assert payload["packageable"] is False
    assert payload["requires_human_review"] is True
    assert [row["artifact_status"] for row in payload["experiments"]] == [
        "complete",
        "complete",
    ]
    assert payload["experiments"][1]["test_metrics"]["metrics"]["recall"] == 0.91
    assert payload["pairwise_deltas"][0]["delta_is_right_minus_left"]["recall"] == 0.09
    assert payload["pairwise_deltas"][0]["metric_source"] == "test_results_csv"
    assert payload["pairwise_deltas"][0]["selection_made"] is False


def test_compare_kaggle_detector_runs_reports_missing_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "outputs" / "detector_multidomain"
    plan = tmp_path / "outputs" / "reports" / "plan.json"
    output = tmp_path / "outputs" / "reports" / "comparison.json"
    write_json(
        plan,
        {
            "schema_version": 1,
            "commands": [
                _command(project, "A_finetune_txl_pbc", "wbc-detector-multidomain-yolo26n-a"),
                _command(project, "B_clean_pretrained", "wbc-detector-multidomain-yolo26n-b"),
            ],
        },
    )

    report = compare_kaggle_detector_runs(plan_path=plan, output_path=output)

    assert [row["artifact_status"] for row in report["experiments"]] == [
        "missing_best_weights",
        "missing_best_weights",
    ]
    assert report["pairwise_deltas"] == []


def _write_run(path: Path, *, recall: float, map50: float) -> None:
    (path / "weights").mkdir(parents=True, exist_ok=True)
    (path / "weights" / "best.pt").write_bytes(b"weights")
    (path / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        f"1,0.5,{recall},{map50},0.4\n",
        encoding="utf-8",
    )


def _command(project: Path, experiment: str, output_name: str) -> dict[str, str]:
    return {
        "experiment": experiment,
        "purpose": "test",
        "initial_weights": "model.pt",
        "expected_best_weights": str(project / output_name / "weights" / "best.pt"),
        "expected_test_dir": str(project / f"{output_name}-test"),
    }
