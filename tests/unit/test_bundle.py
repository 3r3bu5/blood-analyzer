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


def test_model_card_records_metrics_limitations_and_thresholds(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)

    card = (bundle / "model_card.md").read_text(encoding="utf-8")

    assert "research-only" in card.lower()
    assert "1.1" in card
    assert "lymphocyte_reactive" in card
    assert "insufficiently validated" in card
    assert "accepted" in card and "review_required" in card and "unknown" in card
    assert "threshold" in card.lower()


def test_inspect_bundle_reports_missing_required_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "thresholds.json").unlink()

    report = inspect_bundle(bundle)

    assert report["status"] == "incomplete"
    assert "thresholds.json" in report["missing_files"]


def test_verify_bundle_head_accepts_matching_checksum(tmp_path: Path) -> None:
    from bloodfilm.classification.bundle import verify_bundle_head

    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    bundle_doc = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))

    verify_bundle_head(bundle_doc, bundle / "head.pt")


def test_verify_backbone_checksum_rejects_mismatch(tmp_path: Path) -> None:
    from bloodfilm.classification.bundle import verify_backbone_checksum
    from bloodfilm.errors import ModelLoadError

    weights = tmp_path / "wrong.pth"
    weights.write_bytes(b"wrong")

    with pytest.raises(ModelLoadError, match="checksum"):
        verify_backbone_checksum({"backbone_sha256": "f" * 64}, weights)


def test_verify_backbone_checksum_accepts_declared_alternate(tmp_path: Path) -> None:
    from bloodfilm.classification.bundle import verify_backbone_checksum
    from bloodfilm.util import sha256_file

    weights = tmp_path / "official.pth"
    weights.write_bytes(b"official")

    verify_backbone_checksum(
        {"backbone_sha256": "f" * 64, "accepted_backbone_sha256": [sha256_file(weights)]},
        weights,
    )


def test_verify_backbone_checksum_accepts_official_dinobloom_b_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bloodfilm.classification import bundle

    weights = tmp_path / "DinoBloom-B.pth"
    weights.write_bytes(b"official")
    monkeypatch.setattr(bundle, "sha256_file", lambda _: bundle.OFFICIAL_DINOBLOOM_B_SHA256)

    bundle.verify_backbone_checksum(
        {"backbone": "DinoBloom-B", "backbone_sha256": "f" * 64},
        weights,
    )


def test_inspect_bundle_reports_checksum_mismatch_as_invalid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "head.pt").write_bytes(b"tampered")

    report = inspect_bundle(bundle)

    assert report["status"] == "invalid"
    assert report["checksum_errors"] != []


def test_load_bundle_documents_rejects_missing_head(tmp_path: Path) -> None:
    from bloodfilm.classification.bundle import load_bundle_documents
    from bloodfilm.errors import ConfigError

    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    (bundle / "head.pt").unlink()

    with pytest.raises(ConfigError, match="incomplete"):
        load_bundle_documents(bundle)


def test_classify_folder_writes_unknown_per_file_json(tmp_path: Path) -> None:
    import json

    from tests.helpers.png import write_rgb_png

    bundle = tmp_path / "bundle"
    _write_complete_bundle(bundle)
    crops = tmp_path / "crops"
    crops.mkdir()
    write_rgb_png(crops / "a.png", 1, 1, [(255, 255, 255)])
    write_rgb_png(crops / "b.png", 1, 1, [(0, 0, 0)])
    output = tmp_path / "predictions"

    exit_code = main(
        [
            "classifier",
            "classify-folder",
            "--bundle",
            str(bundle),
            "--input",
            str(crops),
            "--backbone-weights",
            str(tmp_path / "missing.pth"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["count"] == 2
    for prediction_path in (output / "a.json", output / "b.json"):
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        assert payload["research_only"] is True
        assert payload["decision"]["status"] == "unknown"


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


@pytest.mark.parametrize("flag", ["--weights", "--dataset-root", "--embeddings", "--checkpoint"])
def test_parity_live_cli_reports_each_missing_asset(tmp_path: Path, flag: str) -> None:
    present = tmp_path / "present"
    present.mkdir()
    (present / "weights.pth").write_bytes(b"weights")
    (present / "embeddings.pt").write_bytes(b"cache")
    (present / "mlp.pt").write_bytes(b"checkpoint")
    (present / "data").mkdir()
    locations = {
        "--weights": str(present / "weights.pth"),
        "--dataset-root": str(present / "data"),
        "--embeddings": str(present / "embeddings.pt"),
        "--checkpoint": str(present / "mlp.pt"),
    }
    locations[flag] = str(tmp_path / "missing-asset")

    exit_code = main(
        [
            "classifier",
            "parity-live",
            "--weights",
            locations["--weights"],
            "--dataset-root",
            locations["--dataset-root"],
            "--embeddings",
            locations["--embeddings"],
            "--checkpoint",
            locations["--checkpoint"],
            "--output",
            str(tmp_path / "parity.json"),
        ]
    )

    assert exit_code == 2


def test_parity_tolerances_match_canonical_values() -> None:
    from bloodfilm.classification import parity as parity_module

    assert parity_module.EMBEDDING_COSINE_MIN == 0.9999
    assert parity_module.LOGITS_ATOL == 1e-4
    assert parity_module.LOGITS_RTOL == 1e-4
    assert parity_module.LOGITS_TOLERANCE_LABEL == "1e-4"


def test_parity_live_defaults_to_all_mll23_classes() -> None:
    from bloodfilm.classification import parity as parity_module

    assert parity_module.DEFAULT_PARITY_CROPS == 18


def test_validation_reports_cli_reports_missing_input_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "classifier",
            "validation-reports",
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--splits",
            str(tmp_path / "missing.csv"),
            "--invalid-manifest",
            str(tmp_path / "missing.csv"),
            "--embeddings-report",
            str(tmp_path / "missing.json"),
            "--comparison-report",
            str(tmp_path / "missing.json"),
            "--bundle-inspection",
            str(tmp_path / "missing.json"),
            "--thresholds",
            str(tmp_path / "missing.json"),
            "--test-evaluation",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 2
    assert "INPUT_NOT_FOUND" in capsys.readouterr().err


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
