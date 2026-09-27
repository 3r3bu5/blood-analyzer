import json
from pathlib import Path

from bloodfilm.detection.dataset import prepare_wbc_yolo_dataset
from tests.helpers.png import write_rgb_png


def _write_txl_fixture(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    write_rgb_png(root / "images" / "train" / "field.png", 4, 4, [(1, 2, 3)] * 16)
    (root / "labels" / "train" / "field.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.5 0.5\n2 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )


def test_prepare_wbc_yolo_dataset_keeps_only_wbc_labels(tmp_path: Path) -> None:
    source = tmp_path / "TXL-PBC"
    output = tmp_path / "wbc-yolo"
    _write_txl_fixture(source)

    report = prepare_wbc_yolo_dataset(source, output, wbc_class_id=0)

    assert report["status"] == "ok"
    assert report["images_copied"] == 1
    assert report["wbc_boxes"] == 1
    assert (output / "images" / "train" / "field.png").exists()
    assert (output / "labels" / "train" / "field.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2\n"
    )
    data = json.loads((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {"0": "wbc_candidate"}
