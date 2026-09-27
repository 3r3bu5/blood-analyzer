import json
from pathlib import Path

import pytest

from bloodfilm.errors import ConfigError
from tests.helpers.png import write_rgb_png


def test_standard_yolo_parser_maps_and_preserves_source_class(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import parse_yolo_annotations

    label = tmp_path / "field.txt"
    label.write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")

    records = parse_yolo_annotations(
        label,
        image_width=200,
        image_height=100,
        class_names={2: "Neutrophil"},
        class_mapping={"Neutrophil": "candidate_wbc"},
    )

    assert len(records) == 1
    assert records[0].box.to_dict() == {"x1": 75.0, "y1": 25.0, "x2": 125.0, "y2": 75.0}
    assert records[0].canonical_class == "candidate_wbc"
    assert records[0].source_class == "Neutrophil"
    assert records[0].attributes == []


def test_extended_yolo_parser_preserves_extra_attributes(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import parse_yolo_annotations

    label = tmp_path / "field.txt"
    label.write_text("0 0.5 0.5 0.2 0.2 blast high-risk microscope-a\n", encoding="utf-8")

    records = parse_yolo_annotations(
        label,
        image_width=100,
        image_height=100,
        class_names={0: "Myeloblast"},
        class_mapping={"Myeloblast": "candidate_wbc"},
    )

    assert records[0].attributes == ["blast", "high-risk", "microscope-a"]
    assert records[0].metadata["source_class"] == "Myeloblast"


def test_yolo_parser_reports_malformed_line_with_file_and_line(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import parse_yolo_annotations

    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="bad.txt:1"):
        parse_yolo_annotations(
            label,
            image_width=100,
            image_height=100,
            class_names={0: "Neutrophil"},
            class_mapping={"Neutrophil": "candidate_wbc"},
        )


def test_yolo_parser_warns_for_unknown_source_class(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import parse_yolo_annotations

    label = tmp_path / "field.txt"
    label.write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    records = parse_yolo_annotations(
        label,
        image_width=100,
        image_height=100,
        class_names={9: "Unverified"},
        class_mapping={"Neutrophil": "candidate_wbc"},
    )

    assert records[0].canonical_class == "unknown"
    assert records[0].warnings == ["unmapped_source_class:Unverified"]


def test_coco_parser_reads_boxes_and_preserves_attributes(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import parse_coco_annotations

    coco = tmp_path / "annotations.json"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "field.png", "width": 100, "height": 80}],
                "categories": [{"id": 5, "name": "Lymphoblast"}],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": [10, 20, 30, 40],
                        "attributes": {"magnification": "100x"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = parse_coco_annotations(coco, class_mapping={"Lymphoblast": "candidate_wbc"})

    assert records["field.png"][0].box.to_dict() == {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 40.0,
        "y2": 60.0,
    }
    assert records["field.png"][0].metadata["attributes"] == {"magnification": "100x"}


def test_sparse_annotations_are_rejected_for_training_preparation() -> None:
    from bloodfilm.detection.annotations import AnnotationCompleteness, require_fully_annotated

    require_fully_annotated(AnnotationCompleteness.FULLY_ANNOTATED)
    with pytest.raises(ConfigError, match="fully annotated"):
        require_fully_annotated(AnnotationCompleteness.SPARSELY_ANNOTATED)


def test_annotation_audit_detects_field_sized_box(tmp_path: Path) -> None:
    from bloodfilm.detection.annotations import audit_yolo_directory

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()
    write_rgb_png(image_root / "field.png", 100, 100, [(255, 255, 255)] * 10000)
    (label_root / "field.txt").write_text("0 0.5 0.5 0.95 0.95\n", encoding="utf-8")

    report = audit_yolo_directory(
        image_root=image_root,
        label_root=label_root,
        class_names={0: "Neutrophil"},
        class_mapping={"Neutrophil": "candidate_wbc"},
        annotation_completeness="fully_annotated",
    )

    assert report["status"] == "review_required"
    assert report["field_sized_boxes"] == 1
    assert report["canonical_class_counts"] == {"candidate_wbc": 1}
