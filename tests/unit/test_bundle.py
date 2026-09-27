import json
from pathlib import Path

import pytest

from bloodfilm.classification.bundle import inspect_bundle, write_bundle_metadata
from bloodfilm.classification.uncertainty import UncertaintyPolicy
from bloodfilm.cli import main


def _write_complete_bundle(path: Path) -> None:
    head = path / "head.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"checkpoint")
    write_bundle_metadata(
        path,
        head_checkpoint=head,
        class_names=["basophil", "eosinophil"],
        head_name="mlp",
        backbone_sha256="f" * 64,
        preprocessing_sha256="3" * 64,
        temperature=1.1,
        validation_metrics={"macro_f1": 0.8},
        uncertainty_policy=UncertaintyPolicy(min_accept_confidence=0.75),
    )


def test_inspect_bundle_reports_required_files_and_hashes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)

    report = inspect_bundle(bundle)

    assert report["status"] == "ok"
    assert report["bundle"]["head"] == "mlp"
    assert report["bundle"]["num_classes"] == 2
    assert report["files"]["head.pt"]["exists"] is True
    assert len(report["files"]["head.pt"]["sha256"]) == 64


def test_inspect_bundle_reports_missing_required_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "thresholds.json").unlink()

    report = inspect_bundle(bundle)

    assert report["status"] == "incomplete"
    assert "thresholds.json" in report["missing_files"]


def test_inspect_bundle_reports_checksum_mismatch_as_invalid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "head.pt").write_bytes(b"tampered")

    report = inspect_bundle(bundle)

    assert report["status"] == "invalid"
    assert report["checksum_errors"] != []


def test_inspect_bundle_cli_writes_json(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    output = tmp_path / "inspect.json"

    exit_code = main(
        [
            "classifier",
            "inspect-bundle",
            "--bundle",
            str(bundle),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_classify_crop_cli_reports_missing_backbone(tmp_path: Path) -> None:
    from tests.helpers.png import write_rgb_png

    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    image = tmp_path / "cell.png"
    write_rgb_png(image, 1, 1, [(255, 255, 255)])

    exit_code = main(
        [
            "classifier",
            "classify-crop",
            "--bundle",
            str(bundle),
            "--image",
            str(image),
            "--backbone-weights",
            str(tmp_path / "missing.pth"),
            "--output",
            str(tmp_path / "prediction.json"),
        ]
    )

    assert exit_code == 2


def test_classify_crop_cli_reports_wrong_backbone_checksum(tmp_path: Path) -> None:
    from tests.helpers.png import write_rgb_png

    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    image = tmp_path / "cell.png"
    weights = tmp_path / "wrong.pth"
    weights.write_bytes(b"wrong")
    write_rgb_png(image, 1, 1, [(255, 255, 255)])

    exit_code = main(
        [
            "classifier",
            "classify-crop",
            "--bundle",
            str(bundle),
            "--image",
            str(image),
            "--backbone-weights",
            str(weights),
            "--output",
            str(tmp_path / "prediction.json"),
        ]
    )

    assert exit_code == 2


def test_package_bundle_cli_writes_required_files(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "mlp.pt"
    torch.save(
        {
            "head": "mlp",
            "class_names": ["basophil", "eosinophil"],
            "model_state": {"probe": torch.tensor([1.0])},
        },
        checkpoint,
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "selected": "mlp",
                "weight_sha256": "f" * 64,
                "preprocessing_sha256": "3" * 64,
                "heads": {"mlp": {"temperature": 1.2, "macro_f1": 0.83}},
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"

    exit_code = main(
        [
            "classifier",
            "package-bundle",
            "--checkpoint",
            str(checkpoint),
            "--comparison-report",
            str(comparison),
            "--bundle",
            str(bundle),
        ]
    )

    assert exit_code == 0
    assert json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))["head"] == "mlp"
    assert json.loads((bundle / "taxonomy.json").read_text(encoding="utf-8"))["class_count"] == 2
    assert (bundle / "sha256sums.txt").exists()


def test_evaluate_cache_cli_reports_missing_checkpoint(tmp_path: Path) -> None:
    exit_code = main(
        [
            "classifier",
            "evaluate-cache",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--embeddings",
            str(tmp_path / "embeddings.pt"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 2
