import json
from pathlib import Path

import pytest

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


def test_detector_prepare_cli_writes_multidomain_yolo_dataset(tmp_path: Path) -> None:
    txl = tmp_path / "txl"
    leukemia = tmp_path / "leukemia"
    (txl / "images" / "train").mkdir(parents=True)
    (txl / "labels" / "train").mkdir(parents=True)
    (leukemia / "images" / "val").mkdir(parents=True)
    (leukemia / "labels" / "val").mkdir(parents=True)
    write_rgb_png(txl / "images" / "train" / "txl.png", 20, 20, [(255, 255, 255)] * 400)
    write_rgb_png(leukemia / "images" / "val" / "leuk.png", 20, 20, [(255, 255, 255)] * 400)
    (txl / "labels" / "train" / "txl.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (leukemia / "labels" / "val" / "leuk.txt").write_text(
        "2 0.5 0.5 0.3 0.3 attrs\n", encoding="utf-8"
    )
    output = tmp_path / "multidomain"
    report = tmp_path / "report.json"
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.csv"
    leakage = tmp_path / "leakage.json"

    exit_code = main(
        [
            "detector",
            "prepare",
            "--txl-pbc-root",
            str(txl),
            "--leukemia-attri-root",
            str(leukemia),
            "--output-root",
            str(output),
            "--manifest-output",
            str(manifest),
            "--splits-output",
            str(splits),
            "--leakage-output",
            str(leakage),
            "--report-output",
            str(report),
            "--txl-class-names",
            "0=wbc",
            "--txl-class-mapping",
            "wbc=candidate_wbc",
            "--leukemia-class-names",
            "2=Neutrophil",
            "--leukemia-class-mapping",
            "Neutrophil=candidate_wbc",
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["sample_count"] == 2
    assert (output / "data.yaml").exists()
    assert manifest.exists()
    assert splits.exists()
    assert json.loads(leakage.read_text(encoding="utf-8"))["locked_target_violations"] == []


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


def test_detector_preprocess_cli_writes_roi_report(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    pixels = []
    for y in range(50):
        for x in range(50):
            pixels.append((240, 240, 240) if 10 <= x <= 39 and 10 <= y <= 39 else (0, 0, 0))
    write_rgb_png(image, 50, 50, pixels)
    report = tmp_path / "roi.json"

    exit_code = main(
        [
            "detector",
            "preprocess",
            "--input",
            str(image),
            "--report-output",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["roi"]["method"] == "auto_illuminated_region"


def test_detector_audit_cli_writes_yolo_audit(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    write_rgb_png(image_root / "field.png", 20, 20, [(255, 255, 255)] * 400)
    (label_root / "field.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    report = tmp_path / "audit.json"

    exit_code = main(
        [
            "detector",
            "audit",
            "--format",
            "leukemia-attri-yolo",
            "--image-root",
            str(image_root),
            "--label-root",
            str(label_root),
            "--class-names",
            "0=Neutrophil",
            "--class-mapping",
            "Neutrophil=candidate_wbc",
            "--report-output",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["canonical_class_counts"] == {"candidate_wbc": 1}


def test_detector_sweep_cli_records_preprocessing_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_detector_sweep import _install_fake_yolo

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    image = tmp_path / "field.png"
    write_rgb_png(image, 20, 20, [(1, 2, 3)] * 400)
    report = tmp_path / "sweep.json"
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
            "--mode",
            "field_roi_tiled",
            "--report-output",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["preprocessing_mode"] == "field_roi_tiled"
