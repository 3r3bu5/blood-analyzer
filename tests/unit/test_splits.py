from bloodfilm.data.splits import create_leakage_report, create_split_manifest
from bloodfilm.schemas import ManifestRow


def test_split_manifest_keeps_groups_in_one_split() -> None:
    rows = [
        ManifestRow(
            image_id=f"id-{index}",
            image_path=f"class/cell_{index}.png",
            source_folder="class",
            canonical_label="basophil",
            sha256=f"sha-{index}",
            width=2,
            height=2,
            mode="RGB",
            patient_or_source_group="group-a",
        )
        for index in range(4)
    ]

    split_rows = create_split_manifest(rows, seed=42)
    report = create_leakage_report(split_rows)

    assert len({row.split for row in split_rows}) == 1
    assert report["leaking_groups"] == []
    assert report["leaking_group_count"] == 0
    assert report["leakage_free"] is True
    assert report["independence_claim"] is False


def test_leakage_report_marks_surrogate_groups_unassessable() -> None:
    rows = [
        ManifestRow(
            image_id=f"sha-{index}",
            image_path=f"class/cell_{index}.png",
            source_folder="class",
            canonical_label="basophil",
            sha256=f"sha-{index}",
            width=2,
            height=2,
            mode="RGB",
            patient_or_source_group=f"ungrouped:sha-{index}",
        )
        for index in range(4)
    ]

    split_rows = create_split_manifest(rows, seed=42)
    report = create_leakage_report(split_rows)

    assert report["leakage_free"] is None
    assert report["independence_claim"] is False
    assert report["grouping_status"] == "unverified_image_level_surrogate"


def test_leakage_report_still_assesses_real_groups_in_mixed_manifest() -> None:
    rows = [
        ManifestRow(
            image_id="real-1",
            image_path="class/cell_real.png",
            source_folder="class",
            canonical_label="basophil",
            sha256="real-1",
            width=2,
            height=2,
            mode="RGB",
            patient_or_source_group="group-a",
        ),
        ManifestRow(
            image_id="surrogate-1",
            image_path="class/cell_surrogate.png",
            source_folder="class",
            canonical_label="basophil",
            sha256="surrogate-1",
            width=2,
            height=2,
            mode="RGB",
            patient_or_source_group="ungrouped:surrogate-1",
        ),
    ]

    report = create_leakage_report(create_split_manifest(rows, seed=42))

    assert report["leaking_groups"] == []
    assert report["leakage_free"] is True
