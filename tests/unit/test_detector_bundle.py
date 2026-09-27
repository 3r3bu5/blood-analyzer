import json
from pathlib import Path

from bloodfilm.detection.bundle import inspect_detector_bundle, write_detector_bundle


def test_write_detector_bundle_records_weights_config_metrics_and_checksums(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text('{"detector": {"confidence_threshold": 0.25}}', encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"recall": 0.98, "precision": 0.7, "mAP50": 0.9}', encoding="utf-8")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"confidence_threshold": 0.25}', encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text('{"status": "ok", "wbc_class_id": 0}', encoding="utf-8")
    bundle = tmp_path / "models" / "txl-pbc-yolo26n-v0.1"

    write_detector_bundle(
        bundle,
        weights=weights,
        config=config,
        metrics=metrics,
        thresholds=thresholds,
        audit_report=audit,
        model_name="txl-pbc-yolo26n-v0.1",
    )
    report = inspect_detector_bundle(bundle)

    assert report["status"] == "ok"
    bundle_doc = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert bundle_doc["name"] == "txl-pbc-yolo26n-v0.1"
    assert bundle_doc["weights_file"] == "weights.pt"
    assert bundle_doc["research_only"] is True
    assert (bundle / "model_card.md").read_text(encoding="utf-8").startswith("# txl-pbc")
