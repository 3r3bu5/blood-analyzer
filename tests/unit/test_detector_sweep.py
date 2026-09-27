"""Unit tests for prediction-level detector threshold sweeps."""

import sys
import types
from pathlib import Path

import pytest

from bloodfilm.detection.detector import Detection
from bloodfilm.detection.sweep import (
    collect_images,
    load_yolo_truth,
    parse_thresholds,
    run_prediction_sweep,
    score_thresholds_with_truth,
)
from bloodfilm.errors import ConfigError, InputNotFoundError
from tests.helpers.png import write_rgb_png


def test_parse_thresholds_sorts_and_dedupes() -> None:
    assert parse_thresholds("0.25, 0.05,0.25,0.5") == [0.05, 0.25, 0.5]


def test_parse_thresholds_rejects_invalid() -> None:
    with pytest.raises(ConfigError):
        parse_thresholds("0.25,banana")
    with pytest.raises(ConfigError):
        parse_thresholds("1.5")
    with pytest.raises(ConfigError):
        parse_thresholds("  ,  ")


def test_collect_images_accepts_single_file(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    write_rgb_png(image, 2, 2, [(1, 2, 3)] * 4)

    assert collect_images(image) == [image]


def test_collect_images_collects_directory(tmp_path: Path) -> None:
    write_rgb_png(tmp_path / "a.png", 2, 2, [(1, 2, 3)] * 4)
    write_rgb_png(tmp_path / "b.jpg", 2, 2, [(1, 2, 3)] * 4)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    assert [path.name for path in collect_images(tmp_path)] == ["a.png", "b.jpg"]


def test_collect_images_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        collect_images(tmp_path / "absent.png")


def test_load_yolo_truth_converts_normalized_boxes(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    write_rgb_png(image, 100, 50, [(1, 2, 3)] * (100 * 50))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "field.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    truth = load_yolo_truth(labels, [image])

    assert len(truth["field"]) == 1
    box = truth["field"][0]
    assert (box.x1, box.y1, box.x2, box.y2) == (40.0, 15.0, 60.0, 35.0)
    assert box.label == "wbc_candidate"


def test_load_yolo_truth_rejects_malformed_labels(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    write_rgb_png(image, 10, 10, [(1, 2, 3)] * 100)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "field.txt").write_text("0 0.5\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_yolo_truth(labels, [image])


def test_score_thresholds_selects_low_threshold_for_recall() -> None:
    predictions = {
        "field": [
            Detection(0, 0, 10, 10, 0.9),
            Detection(20, 20, 30, 30, 0.12),
        ]
    }
    truth = {
        "field": [
            Detection(0, 0, 10, 10, 1.0),
            Detection(20, 20, 30, 30, 1.0),
        ]
    }

    report = score_thresholds_with_truth(
        predictions,
        truth,
        [0.1, 0.5],
        iou_threshold=0.5,
        target_recall=0.95,
        max_false_positives_per_image=2.0,
    )

    assert report["selected"]["confidence_threshold"] == 0.1
    assert [row["recall"] for row in report["candidates"]] == [1.0, 0.5]


def test_run_prediction_sweep_counts_boxes_per_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    image = tmp_path / "field.png"
    write_rgb_png(image, 20, 20, [(1, 2, 3)] * 400)
    _install_fake_yolo(monkeypatch, scores_by_threshold={0.1: [0.9, 0.2], 0.5: [0.9]})

    report = run_prediction_sweep(weights, [image], [0.1, 0.5])

    counts = {row["confidence_threshold"]: row["detection_count"] for row in report["candidates"]}
    assert counts == {0.1: 2, 0.5: 1}
    assert report["candidates"][0]["boxes_by_image"]["field"][0]["score"] == 0.9


def test_run_prediction_sweep_rejects_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_yolo(monkeypatch, scores_by_threshold={})
    image = tmp_path / "field.png"
    write_rgb_png(image, 2, 2, [(1, 2, 3)] * 4)

    with pytest.raises(InputNotFoundError):
        run_prediction_sweep(tmp_path / "absent.pt", [image], [0.25])


def _install_fake_yolo(
    monkeypatch: pytest.MonkeyPatch, *, scores_by_threshold: dict[float, list[float]]
) -> None:
    module = types.ModuleType("ultralytics")

    class FakeBox:
        def __init__(self, score: float) -> None:
            self.xyxy = [[0.0, 0.0, 10.0, 10.0]]
            self.conf = [score]

    class FakeResult:
        def __init__(self, scores: list[float]) -> None:
            self.boxes = [FakeBox(score) for score in scores] or None

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(
            self,
            source: str,
            *,
            conf: float,
            iou: float,
            imgsz: int,
            verbose: bool,
        ) -> list[FakeResult]:
            assert source and iou and imgsz and verbose is False
            return [FakeResult(scores_by_threshold.get(round(conf, 4), []))]

    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_cli_sweep_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bloodfilm.cli import main

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    image = tmp_path / "field.png"
    write_rgb_png(image, 20, 20, [(1, 2, 3)] * 400)
    output = tmp_path / "sweep.json"
    _install_fake_yolo(monkeypatch, scores_by_threshold={0.25: [0.8]})

    exit_code = main(
        [
            "detector",
            "sweep",
            "--weights",
            str(weights),
            "--input",
            str(image),
            "--thresholds",
            "0.25",
            "--report-output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()
