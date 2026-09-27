import json
from pathlib import Path

from bloodfilm.cli import main
from bloodfilm.data.txl_pbc import audit_txl_pbc_dataset
from tests.helpers.png import write_rgb_png


def _write_yolo_fixture(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    write_rgb_png(root / "images" / "train" / "field.png", 10, 10, [(0, 0, 0)] * 100)
    (root / "labels" / "train" / "field.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nnames:\n  0: RBC\n  1: WBC\n  2: platelet\n",
        encoding="utf-8",
    )


def test_audit_txl_pbc_dataset_reports_wbc_yolo_labels(tmp_path: Path) -> None:
    _write_yolo_fixture(tmp_path)

    report = audit_txl_pbc_dataset(tmp_path)

    assert report["status"] == "ok"
    assert report["images"]["count"] == 1
    assert report["labels"]["count"] == 1
    assert report["classes"]["wbc"]["class_id"] == 1
    assert report["classes"]["wbc"]["box_count"] == 1
    assert report["invalid_labels"] == []


def test_dataset_audit_cli_supports_txl_pbc_format(tmp_path: Path) -> None:
    _write_yolo_fixture(tmp_path)
    output = tmp_path / "audit.json"

    exit_code = main(
        [
            "dataset",
            "audit",
            "TXL-PBC",
            "--dataset-root",
            str(tmp_path),
            "--format",
            "txl-pbc",
            "--report-output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dataset"] == "TXL-PBC"
    assert report["classes"]["wbc"]["box_count"] == 1
