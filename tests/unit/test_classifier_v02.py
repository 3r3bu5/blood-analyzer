from pathlib import Path

import pytest

from bloodfilm.classification.v02 import (
    MLL23_CLASSES,
    acceptance_status,
    assert_hashes_unchanged,
    decide_source_mapping,
    group_aware_split,
    partial_label_loss,
    probability_schema,
    split_leakage_report,
    square_crop_geometry,
    verify_mll23_class_order,
)
from bloodfilm.errors import ConfigError


def test_verify_mll23_class_order_rejects_reordered_taxonomy() -> None:
    verify_mll23_class_order(MLL23_CLASSES)
    with pytest.raises(ConfigError):
        verify_mll23_class_order([*MLL23_CLASSES[1:], MLL23_CLASSES[0]])


def test_leukemiaattri_mapping_marks_exact_partial_and_ood() -> None:
    assert decide_source_mapping("Basophil").to_dict() == {
        "source_class": "Basophil",
        "normalized_source_class": "basophil",
        "mapping_type": "exact",
        "target_class": "basophil",
        "partial_target_classes": [],
        "requires_provenance_review": False,
    }
    lymphocyte = decide_source_mapping("Lymphocyte")
    assert lymphocyte.target_class == "lymphocyte_typical"
    assert lymphocyte.requires_provenance_review is True
    neutrophil = decide_source_mapping("Neutrophil")
    assert neutrophil.mapping_type == "partial"
    assert neutrophil.partial_target_classes == ("neutrophil_band", "neutrophil_segmented")
    assert decide_source_mapping("Lymphoblast").mapping_type == "unmapped_ood"


def test_partial_label_loss_uses_probability_mass_for_allowed_classes() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32)
    loss = partial_label_loss(logits, [1, 2], torch=torch)
    expected = -torch.logsumexp(
        torch.nn.functional.log_softmax(logits, dim=-1)[:, [1, 2]], dim=-1
    ).mean()
    assert torch.isclose(loss, expected)


def test_square_crop_geometry_clips_and_records_padding() -> None:
    crop = square_crop_geometry((0, 5, 10, 15), image_width=20, image_height=20, padding_ratio=0.25)
    assert crop.output_size == 15
    assert crop.crop_box == (0, 2, 12, 17)
    assert crop.padding[0] == 3
    assert crop.paste_box == (3, 0, 15, 15)


def test_group_aware_split_keeps_groups_together() -> None:
    rows = [{"group": "a"}, {"group": "a"}, {"group": "b"}, {"group": "c"}]
    splits = group_aware_split(rows, group_key="group", seed=1, train=0.5, validation=0.25)
    by_group: dict[str, set[str]] = {}
    for row, split in zip(rows, splits):
        by_group.setdefault(row["group"], set()).add(split)
    assert all(len(values) == 1 for values in by_group.values())
    assert set(splits) <= {"train", "validation", "test"}


def test_split_leakage_report_detects_cross_split_hashes() -> None:
    rows = [
        {"split": "train", "image_sha256": "same"},
        {"split": "test", "image_sha256": "same"},
    ]
    report = split_leakage_report(rows, "split", ["image_sha256"])
    assert report["status"] == "fail"
    assert report["violations"][0]["field"] == "image_sha256"


def test_assert_hashes_unchanged_reports_protected_mutation() -> None:
    assert_hashes_unchanged({"a": "1"}, {"a": "1"})
    with pytest.raises(ConfigError):
        assert_hashes_unchanged({"a": "1"}, {"a": "2"})


def test_probability_schema_requires_all_probabilities() -> None:
    payload = probability_schema([0.8, 0.2], ["a", "b"])
    assert payload["top_class"] == "a"
    assert payload["top_probability"] == 0.8
    assert len(payload["probabilities"]) == 2
    with pytest.raises(ValueError):
        probability_schema([0.8, 0.1], ["a", "b"])


def test_acceptance_status_reports_specific_rejections() -> None:
    gates = {
        "mll23_accuracy_max_drop": 0.005,
        "mll23_macro_f1_max_drop": 0.010,
        "external_exact_macro_f1_min_improvement": 0.05,
    }
    accepted = acceptance_status(
        {
            "mll23_accuracy_drop": 0.0,
            "mll23_macro_f1_drop": 0.0,
            "external_exact_macro_f1_improvement": 0.06,
            "all_18_output_classes_required": True,
        },
        gates,
    )
    assert accepted["status"] == "accepted_candidate"
    rejected = acceptance_status(
        {
            "mll23_accuracy_drop": 0.006,
            "mll23_macro_f1_drop": 0.0,
            "external_exact_macro_f1_improvement": 0.06,
            "all_18_output_classes_required": True,
        },
        gates,
    )
    assert rejected["status"] == "rejected_mll23_regression"
