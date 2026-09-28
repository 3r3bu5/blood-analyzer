import csv
import json
from pathlib import Path

import pytest

from tests.helpers.png import write_rgb_png


def test_prepare_multidomain_yolo_dataset_writes_data_yaml_manifest_and_labels(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset

    txl = tmp_path / "txl"
    leukemia = tmp_path / "leukemia"
    _write_yolo_sample(
        txl,
        split="train",
        stem="txl_field",
        label_text="0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.5 0.5\n",
    )
    _write_yolo_sample(
        leukemia,
        split="val",
        stem="leuk_field",
        label_text="2 0.4 0.4 0.3 0.3 blast reviewed\n3 0.5 0.5 0.2 0.2 artifact\n",
    )
    output = tmp_path / "multidomain"
    stale_image = output / "images" / "train" / "stale.png"
    stale_label = output / "labels" / "train" / "stale.txt"
    stale_image.parent.mkdir(parents=True)
    stale_label.parent.mkdir(parents=True)
    stale_image.write_bytes(b"stale")
    stale_label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.csv"
    leakage = tmp_path / "leakage.json"
    report = tmp_path / "report.json"

    result = prepare_multidomain_yolo_dataset(
        txl_pbc_root=txl,
        leukemia_attri_root=leukemia,
        output_root=output,
        manifest_output=manifest,
        splits_output=splits,
        leakage_output=leakage,
        report_output=report,
        txl_class_names={0: "wbc", 1: "rbc"},
        txl_class_mapping={"wbc": "candidate_wbc"},
        leukemia_class_names={2: "Neutrophil", 3: "Artifact"},
        leukemia_class_mapping={"Neutrophil": "candidate_wbc", "Artifact": "artifact"},
        locked_hashes=set(),
    )

    assert result["status"] == "ok"
    assert json.loads((output / "data.yaml").read_text(encoding="utf-8")) == {
        "names": {"0": "candidate_wbc"},
        "path": str(output),
        "test": "images/test",
        "train": "images/train",
        "val": "images/val",
    }
    assert (output / "labels" / "train" / "txl_pbc__train__txl_field.txt").read_text(
        encoding="utf-8"
    ) == "0 0.500000 0.500000 0.200000 0.200000\n"
    assert (output / "labels" / "val" / "leukemia_attri__val__leuk_field.txt").read_text(
        encoding="utf-8"
    ) == "0 0.400000 0.400000 0.300000 0.300000\n"
    assert not stale_image.exists()
    assert not stale_label.exists()
    rows = list(csv.DictReader(manifest.read_text(encoding="utf-8").splitlines()))
    assert {row["source_dataset"] for row in rows} == {"LeukemiaAttri", "TXL-PBC"}
    assert rows[0]["canonical_classes"] == "candidate_wbc"
    assert json.loads(leakage.read_text(encoding="utf-8"))["locked_target_violations"] == []


def test_prepare_multidomain_yolo_dataset_rejects_sparse_leukemia_annotations(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset
    from bloodfilm.errors import ConfigError

    txl = tmp_path / "txl"
    leukemia = tmp_path / "leukemia"
    _write_yolo_sample(txl, split="train", stem="txl_field", label_text="0 0.5 0.5 0.2 0.2\n")
    _write_yolo_sample(leukemia, split="train", stem="leuk_field", label_text="0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ConfigError, match="fully annotated"):
        prepare_multidomain_yolo_dataset(
            txl_pbc_root=txl,
            leukemia_attri_root=leukemia,
            output_root=tmp_path / "out",
            manifest_output=tmp_path / "manifest.csv",
            splits_output=tmp_path / "splits.csv",
            leakage_output=tmp_path / "leakage.json",
            report_output=tmp_path / "report.json",
            txl_class_names={0: "wbc"},
            txl_class_mapping={"wbc": "candidate_wbc"},
            leukemia_class_names={0: "Neutrophil"},
            leukemia_class_mapping={"Neutrophil": "candidate_wbc"},
            locked_hashes=set(),
            leukemia_annotation_completeness="sparsely_annotated",
        )


def test_prepare_multidomain_yolo_dataset_rejects_missing_labels(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset
    from bloodfilm.errors import ConfigError

    txl = tmp_path / "txl"
    leukemia = tmp_path / "leukemia"
    _write_yolo_sample(txl, split="train", stem="txl_field", label_text="0 0.5 0.5 0.2 0.2\n")
    _write_yolo_sample(leukemia, split="train", stem="leuk_field", label_text="0 0.5 0.5 0.2 0.2\n")
    write_rgb_png(
        leukemia / "images" / "train" / "missing_label.png",
        20,
        20,
        [(255, 255, 255)] * 400,
    )

    with pytest.raises(ConfigError, match="images without labels"):
        prepare_multidomain_yolo_dataset(
            txl_pbc_root=txl,
            leukemia_attri_root=leukemia,
            output_root=tmp_path / "out",
            manifest_output=tmp_path / "manifest.csv",
            splits_output=tmp_path / "splits.csv",
            leakage_output=tmp_path / "leakage.json",
            report_output=tmp_path / "report.json",
            txl_class_names={0: "wbc"},
            txl_class_mapping={"wbc": "candidate_wbc"},
            leukemia_class_names={0: "Neutrophil"},
            leukemia_class_mapping={"Neutrophil": "candidate_wbc"},
            locked_hashes=set(),
        )


def test_prepare_multidomain_yolo_dataset_accepts_leukemia_coco_domain_layout(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset

    txl = tmp_path / "txl"
    leukemia = tmp_path / "LeukemiaAttri"
    _write_yolo_sample(txl, split="train", stem="txl_field", label_text="0 0.5 0.5 0.2 0.2\n")
    image = leukemia / "H_100X_C1" / "Images" / "test" / "leuk_field.png"
    image.parent.mkdir(parents=True)
    write_rgb_png(image, 100, 100, [(255, 255, 255)] * 10000)
    annotation = leukemia / "H_100X_C1" / "json_labels" / "test.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "leuk_field.png", "width": 100, "height": 100}],
                "categories": [
                    {"id": 3, "name": "neutrophil"},
                    {"id": 14, "name": "none"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 3, "bbox": [40, 40, 20, 20]},
                    {"id": 11, "image_id": 1, "category_id": 14, "bbox": [0, 0, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )

    prepare_multidomain_yolo_dataset(
        txl_pbc_root=txl,
        leukemia_attri_root=leukemia,
        output_root=tmp_path / "out",
        manifest_output=tmp_path / "manifest.csv",
        splits_output=tmp_path / "splits.csv",
        leakage_output=tmp_path / "leakage.json",
        report_output=tmp_path / "report.json",
        txl_class_names={0: "wbc"},
        txl_class_mapping={"wbc": "candidate_wbc"},
        leukemia_class_names={},
        leukemia_class_mapping={"neutrophil": "candidate_wbc", "none": "artifact"},
        locked_hashes=set(),
        leukemia_annotation_format="coco_domain",
    )

    assert (
        tmp_path / "out" / "labels" / "val" / "leukemia_attri__H_100X_C1__test__leuk_field.txt"
    ).read_text(encoding="utf-8") == "0 0.500000 0.500000 0.200000 0.200000\n"
    rows = list(
        csv.DictReader((tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines())
    )
    leukemia_row = next(row for row in rows if row["source_dataset"] == "LeukemiaAttri")
    assert leukemia_row["source_split"] == "test"
    assert leukemia_row["unified_split"] == "val"
    assert leukemia_row["acquisition_domain"] == "H_100X_C1"


def test_prepare_multidomain_yolo_dataset_rejects_unmapped_leukemia_coco_classes(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset
    from bloodfilm.errors import ConfigError

    txl = tmp_path / "txl"
    leukemia = tmp_path / "LeukemiaAttri"
    _write_yolo_sample(txl, split="train", stem="txl_field", label_text="0 0.5 0.5 0.2 0.2\n")
    image = leukemia / "H_100X_C1" / "Images" / "test" / "leuk_field.png"
    image.parent.mkdir(parents=True)
    write_rgb_png(image, 100, 100, [(255, 255, 255)] * 10000)
    annotation = leukemia / "H_100X_C1" / "json_labels" / "test.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "leuk_field.png", "width": 100, "height": 100}],
                "categories": [{"id": 3, "name": "neutrophil"}],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 3, "bbox": [40, 40, 20, 20]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unmapped COCO categories"):
        prepare_multidomain_yolo_dataset(
            txl_pbc_root=txl,
            leukemia_attri_root=leukemia,
            output_root=tmp_path / "out",
            manifest_output=tmp_path / "manifest.csv",
            splits_output=tmp_path / "splits.csv",
            leakage_output=tmp_path / "leakage.json",
            report_output=tmp_path / "report.json",
            txl_class_names={0: "wbc"},
            txl_class_mapping={"wbc": "candidate_wbc"},
            leukemia_class_names={},
            leukemia_class_mapping={},
            locked_hashes=set(),
            leukemia_annotation_format="coco_domain",
        )


def test_prepare_multidomain_yolo_dataset_blocks_locked_target_training_leakage(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset
    from bloodfilm.errors import ConfigError
    from bloodfilm.util import sha256_file

    txl = tmp_path / "txl"
    leukemia = tmp_path / "leukemia"
    image = _write_yolo_sample(txl, split="train", stem="locked", label_text="0 0.5 0.5 0.2 0.2\n")
    _write_yolo_sample(leukemia, split="train", stem="leuk_field", label_text="0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ConfigError, match="locked target"):
        prepare_multidomain_yolo_dataset(
            txl_pbc_root=txl,
            leukemia_attri_root=leukemia,
            output_root=tmp_path / "out",
            manifest_output=tmp_path / "manifest.csv",
            splits_output=tmp_path / "splits.csv",
            leakage_output=tmp_path / "leakage.json",
            report_output=tmp_path / "report.json",
            txl_class_names={0: "wbc"},
            txl_class_mapping={"wbc": "candidate_wbc"},
            leukemia_class_names={0: "Neutrophil"},
            leukemia_class_mapping={"Neutrophil": "candidate_wbc"},
            locked_hashes={sha256_file(image)},
        )


def test_prepare_multidomain_yolo_dataset_skips_degenerate_coco_boxes_with_report(
    tmp_path: Path,
) -> None:
    from bloodfilm.detection.multidomain import prepare_multidomain_yolo_dataset

    txl = tmp_path / "txl"
    leukemia = tmp_path / "LeukemiaAttri"
    _write_yolo_sample(txl, split="train", stem="txl_field", label_text="0 0.5 0.5 0.2 0.2\n")
    image = leukemia / "H_10X_C1" / "Images" / "test" / "leuk_field.png"
    image.parent.mkdir(parents=True)
    write_rgb_png(image, 100, 100, [(255, 255, 255)] * 10000)
    annotation = leukemia / "H_10X_C1" / "json_labels" / "test.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "leuk_field.png", "width": 100, "height": 100}],
                "categories": [{"id": 3, "name": "neutrophil"}],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 3, "bbox": [40, 40, 20, 20]},
                    {"id": 1388, "image_id": 1, "category_id": 3, "bbox": [10, 10, 0, 5]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_multidomain_yolo_dataset(
        txl_pbc_root=txl,
        leukemia_attri_root=leukemia,
        output_root=tmp_path / "out",
        manifest_output=tmp_path / "manifest.csv",
        splits_output=tmp_path / "splits.csv",
        leakage_output=tmp_path / "leakage.json",
        report_output=tmp_path / "report.json",
        txl_class_names={0: "wbc"},
        txl_class_mapping={"wbc": "candidate_wbc"},
        leukemia_class_names={},
        leukemia_class_mapping={"neutrophil": "candidate_wbc"},
        locked_hashes=set(),
        leukemia_annotation_format="coco_domain",
    )

    assert result["status"] == "ok"
    assert result["degenerate_boxes_skipped"]["total"] == 1
    assert (
        tmp_path / "out" / "labels" / "val" / "leukemia_attri__H_10X_C1__test__leuk_field.txt"
    ).read_text(encoding="utf-8") == "0 0.500000 0.500000 0.200000 0.200000\n"


def _write_yolo_sample(root: Path, *, split: str, stem: str, label_text: str) -> Path:
    image = root / "images" / split / f"{stem}.png"
    label = root / "labels" / split / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    write_rgb_png(image, 100, 100, [(255, 255, 255)] * 10000)
    label.write_text(label_text, encoding="utf-8")
    return image
