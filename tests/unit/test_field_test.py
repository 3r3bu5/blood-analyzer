import csv
import json
from pathlib import Path

import pytest

from bloodfilm.detection.field_test import build_source_test_subset, score_weights_on_subset
from bloodfilm.errors import ConfigError, InputNotFoundError


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "sample_id",
        "image_path",
        "label_path",
        "image_sha256",
        "source_dataset",
        "source_split",
        "unified_split",
        "patient_id",
        "slide_id",
        "acquisition_domain",
        "microscope",
        "camera",
        "magnification",
        "annotation_completeness",
        "source_classes",
        "canonical_classes",
        "locked_for_validation",
        "license_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _row(sample_id: str, image: Path, label: Path, source: str, split: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "image_path": str(image),
        "label_path": str(label),
        "source_dataset": source,
        "source_split": split,
        "unified_split": "val" if split == "test" else split,
        "acquisition_domain": "H_100X_C1",
    }


def test_build_source_test_subset_copies_only_matching_rows(tmp_path: Path) -> None:
    unified = tmp_path / "unified"
    (unified / "images" / "val").mkdir(parents=True)
    (unified / "labels" / "val").mkdir(parents=True)
    (unified / "images" / "train").mkdir(parents=True)
    (unified / "images" / "test").mkdir(parents=True)
    kept_images = []
    for stem in ("field_a", "field_b"):
        image = unified / "images" / "val" / f"{stem}.png"
        label = unified / "labels" / "val" / f"{stem}.txt"
        image.write_bytes(b"fake-image")
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        kept_images.append((image, label))
    other_image = unified / "images" / "val" / "other.png"
    other_label = unified / "labels" / "val" / "other.txt"
    other_image.write_bytes(b"fake-image")
    other_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("leuk_a__test__a", *kept_images[0], "LeukemiaAttri", "test"),
            _row("leuk_a__test__b", *kept_images[1], "LeukemiaAttri", "test"),
            _row("leuk_a__train__c", other_image, other_label, "LeukemiaAttri", "train"),
            _row("txl__test__d", other_image, other_label, "TXL-PBC", "test"),
        ],
    )
    base_yaml = tmp_path / "base_data.yaml"
    base_yaml.write_text(
        json.dumps(
            {
                "path": str(unified),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {"0": "candidate_wbc"},
            }
        ),
        encoding="utf-8",
    )

    report = build_source_test_subset(
        manifest_path=manifest,
        data_yaml_path=base_yaml,
        output_root=tmp_path / "subset",
    )

    assert report["sample_count"] == 2
    assert report["source_dataset"] == "LeukemiaAttri"
    subset_images = sorted((tmp_path / "subset" / "images" / "test").iterdir())
    subset_labels = sorted((tmp_path / "subset" / "labels" / "test").iterdir())
    assert [item.name for item in subset_images] == ["field_a.png", "field_b.png"]
    assert [item.name for item in subset_labels] == ["field_a.txt", "field_b.txt"]
    subset_yaml = json.loads((tmp_path / "subset" / "data.yaml").read_text(encoding="utf-8"))
    assert subset_yaml["test"] == "images/test"
    assert subset_yaml["names"] == {"0": "candidate_wbc"}


def test_build_source_test_subset_rejects_empty_filter(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [])
    base_yaml = tmp_path / "base_data.yaml"
    base_yaml.write_text(
        json.dumps({"path": str(tmp_path), "train": "t", "val": "v"}), encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="No manifest rows"):
        build_source_test_subset(
            manifest_path=manifest,
            data_yaml_path=base_yaml,
            output_root=tmp_path / "subset",
        )


def test_build_source_test_subset_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError, match="manifest not found"):
        build_source_test_subset(
            manifest_path=tmp_path / "missing.csv",
            data_yaml_path=tmp_path / "missing.yaml",
            output_root=tmp_path / "subset",
        )


def test_score_weights_requires_existing_weights(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError, match="weights not found"):
        score_weights_on_subset(
            weights=tmp_path / "missing.pt", data_yaml=tmp_path / "missing.yaml"
        )


def test_eval_script_dry_run_builds_subset_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.kaggle_eval_leukemia_test_only as eval_module

    unified = tmp_path / "unified"
    (unified / "images" / "val").mkdir(parents=True)
    (unified / "labels" / "val").mkdir(parents=True)
    (unified / "images" / "train").mkdir(parents=True)
    (unified / "images" / "test").mkdir(parents=True)
    image = unified / "images" / "val" / "field.png"
    label = unified / "labels" / "val" / "field.txt"
    image.write_bytes(b"fake-image")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("leuk__test__a", image, label, "LeukemiaAttri", "test")])
    base_yaml = tmp_path / "base_data.yaml"
    base_yaml.write_text(
        json.dumps(
            {
                "path": str(unified),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {"0": "candidate_wbc"},
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    report_output = tmp_path / "report.json"

    monkeypatch.chdir(tmp_path)
    exit_code = eval_module.main(
        [
            "--manifest",
            str(manifest),
            "--data-yaml",
            str(base_yaml),
            "--weights",
            str(weights),
            "--output-root",
            str(tmp_path / "subset"),
            "--report-output",
            str(report_output),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["evaluations"] == []
