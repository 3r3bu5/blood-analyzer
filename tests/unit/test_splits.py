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
