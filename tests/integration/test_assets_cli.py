import json
from pathlib import Path

from bloodfilm.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry_with_empty_files(tmp_path: Path) -> Path:
    source = (REPO_ROOT / "configs/registry/assets.yaml").read_text(encoding="utf-8")
    data = json.loads(source)
    for dataset in data.get("datasets", []):
        dataset["files"] = []
    registry = tmp_path / "assets.yaml"
    registry.write_text(json.dumps(data), encoding="utf-8")
    return registry


def test_assets_list_reports_registry_entries(tmp_path: Path) -> None:
    output = tmp_path / "assets.json"

    exit_code = main(
        [
            "assets",
            "list",
            "--registry",
            str(REPO_ROOT / "configs/registry/assets.yaml"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "MLL23" in report["datasets"]
    assert "DinoBloom-B" in report["models"]


def test_assets_download_blocked_without_file_entries(tmp_path: Path) -> None:
    registry = _registry_with_empty_files(tmp_path)
    report_path = tmp_path / "download.json"

    exit_code = main(
        [
            "assets",
            "download",
            "mll23",
            "--registry",
            str(registry),
            "--dest-root",
            str(tmp_path / "raw"),
            "--report-output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "blocked"
    assert "files" in report["blocker"].lower() or "file entries" in report["blocker"].lower()


def test_assets_verify_blocked_without_file_entries(tmp_path: Path) -> None:
    registry = _registry_with_empty_files(tmp_path)
    report_path = tmp_path / "verify.json"

    exit_code = main(
        [
            "assets",
            "verify",
            "mll23",
            "--registry",
            str(registry),
            "--dest-root",
            str(tmp_path / "raw"),
            "--report-output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "blocked"


def test_dataset_audit_and_build_manifest_from_name(tmp_path: Path) -> None:
    from tests.helpers.png import write_rgb_png

    dataset_root = tmp_path / "mll23"
    (dataset_root / "Basophil").mkdir(parents=True)
    write_rgb_png(dataset_root / "Basophil" / "cell.png", 2, 2, [(1, 2, 3)] * 4)
    registry = tmp_path / "assets.yaml"
    mapping = str(REPO_ROOT / "configs/mappings/mll23.yaml")
    registry.write_text(
        '{"schema_version": 1,'
        '"datasets": [{"name": "MLL23",'
        f'"local_path": {str(dataset_root)!r},'
        f'"label_mapping": {mapping!r},'
        '"grouping_field": "patient_or_source_group",'
        '"files": []}],'
        '"models": []}'.replace("'", '"'),
        encoding="utf-8",
    )
    audit_report = tmp_path / "audit.json"
    manifest_path = tmp_path / "manifest.csv"

    assert (
        main(
            [
                "dataset",
                "audit",
                "mll23",
                "--registry",
                str(registry),
                "--report-output",
                str(audit_report),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "dataset",
                "build-manifest",
                "mll23",
                "--registry",
                str(registry),
                "--output",
                str(manifest_path),
                "--invalid-output",
                str(tmp_path / "invalid.csv"),
                "--checksum-output",
                str(tmp_path / "checksums.csv"),
                "--report-output",
                str(tmp_path / "manifest_audit.json"),
            ]
        )
        == 0
    )

    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    assert audit["total_files"] == 1
    assert manifest_path.exists()
