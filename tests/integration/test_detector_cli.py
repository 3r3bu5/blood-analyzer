import json
from pathlib import Path

from bloodfilm.cli import main
from tests.helpers.png import write_rgb_png


def test_detector_prepare_yolo_cli_writes_wbc_derivative(tmp_path: Path) -> None:
    source = tmp_path / "TXL-PBC"
    (source / "images" / "train").mkdir(parents=True)
    (source / "labels" / "train").mkdir(parents=True)
    write_rgb_png(source / "images" / "train" / "field.png", 2, 2, [(1, 1, 1)] * 4)
    (source / "labels" / "train" / "field.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    output = tmp_path / "wbc-yolo"
    report = tmp_path / "report.json"

    exit_code = main(
        [
            "detector",
            "prepare-yolo",
            "--dataset-root",
            str(source),
            "--output-root",
            str(output),
            "--wbc-class-id",
            "0",
            "--report-output",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["wbc_boxes"] == 1
    assert (output / "data.yaml").exists()


def test_detector_package_bundle_cli_writes_inspectable_bundle(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text('{"detector": {}}', encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"recall": 1.0, "precision": 0.5, "mAP50": 0.8}', encoding="utf-8")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"confidence_threshold": 0.25}', encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text('{"status": "ok", "wbc_class_id": 0}', encoding="utf-8")
    bundle = tmp_path / "bundle"
    inspect = tmp_path / "inspect.json"

    exit_code = main(
        [
            "detector",
            "package-bundle",
            "--weights",
            str(weights),
            "--config",
            str(config),
            "--metrics",
            str(metrics),
            "--thresholds",
            str(thresholds),
            "--audit-report",
            str(audit),
            "--bundle",
            str(bundle),
        ]
    )
    inspect_code = main(
        ["detector", "inspect-bundle", "--bundle", str(bundle), "--output", str(inspect)]
    )

    assert exit_code == 0
    assert inspect_code == 0
    assert json.loads(inspect.read_text(encoding="utf-8"))["status"] == "ok"
