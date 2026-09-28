import json
import subprocess
from pathlib import Path

import pytest

import scripts.kaggle_download_detector_data as download_module
from scripts.kaggle_download_detector_data import main


def test_kaggle_download_detector_data_dry_run_writes_resume_commands(tmp_path: Path) -> None:
    report = tmp_path / "download.json"

    exit_code = main(
        [
            "--dry-run",
            "--txl-output",
            str(tmp_path / "TXL-PBC"),
            "--leukemia-output",
            str(tmp_path / "LeukemiaAttri"),
            "--report-output",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["txl_pbc"]["command"][:2] == ["git", "clone"]
    assert ["git", "lfs", "pull"] in payload["txl_pbc"]["post_clone_commands"]
    assert "gdown" in payload["leukemia_attri"]["command"]
    assert "--continue" in payload["leukemia_attri"]["command"]
    assert "--remaining-ok" in payload["leukemia_attri"]["command"]


def test_prepare_leukemia_attri_retries_drive_quota_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0, "sleeps": 0}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.CalledProcessError(1, "gdown")
        return subprocess.CompletedProcess(args="gdown", returncode=0)

    def fake_sleep(seconds: float) -> None:
        calls["sleeps"] += 1

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    monkeypatch.setattr(download_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(download_module, "_ensure_gdown", lambda: None)
    monkeypatch.setattr(download_module, "_gdown_supports_remaining_ok", lambda: True)

    report = download_module._prepare_leukemia_attri(
        url="https://example.invalid/folder",
        output=tmp_path / "LeukemiaAttri",
        dry_run=False,
        attempts=3,
        retry_sleep_seconds=1.0,
    )

    assert report["status"] == "ok"
    assert report["attempts"] == 2
    assert calls == {"count": 2, "sleeps": 1}


def test_prepare_leukemia_attri_raises_helpful_error_after_exhausted_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, "gdown")

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    monkeypatch.setattr(download_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(download_module, "_ensure_gdown", lambda: None)
    monkeypatch.setattr(download_module, "_gdown_supports_remaining_ok", lambda: True)

    with pytest.raises(RuntimeError, match="quota"):
        download_module._prepare_leukemia_attri(
            url="https://example.invalid/folder",
            output=tmp_path / "LeukemiaAttri",
            dry_run=False,
            attempts=2,
            retry_sleep_seconds=0.0,
        )


def test_prepare_leukemia_attri_uses_takeout_zip_instead_of_gdown(tmp_path: Path) -> None:
    import zipfile

    source = tmp_path / "source"
    domain_images = source / "LeukemiaAttri_Dataset" / "H_100X_C1" / "Images" / "test"
    domain_labels = source / "LeukemiaAttri_Dataset" / "H_100X_C1" / "json_labels"
    domain_images.mkdir(parents=True)
    domain_labels.mkdir(parents=True)
    (domain_images / "field.png").write_bytes(b"fake-png")
    (domain_labels / "test.json").write_text('{"images": []}', encoding="utf-8")
    zip_path = tmp_path / "takeout.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))

    report = download_module._prepare_leukemia_attri(
        url="https://example.invalid/folder",
        output=tmp_path / "LeukemiaAttri",
        dry_run=False,
        zip_urls=[zip_path.as_uri()],
    )

    assert report["status"] == "ok"
    assert report["method"] == "takeout_zip"
    assert report["json_label_count"] == 1
    assert report["image_count"] == 1
    assert (tmp_path / "LeukemiaAttri" / "H_100X_C1" / "json_labels" / "test.json").exists()


def test_extract_zip_into_rejects_path_traversal(tmp_path: Path) -> None:
    import zipfile

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "evil")

    with pytest.raises(RuntimeError, match="unsafe path"):
        download_module._extract_zip_into(zip_path, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_env_zip_urls_parses_multiline_and_comma_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LEUKEMIA_ZIP_URLS",
        "https://example.invalid/a.zip,\nhttps://example.invalid/b.zip\nhttps://example.invalid/a.zip",
    )

    assert download_module._env_zip_urls() == [
        "https://example.invalid/a.zip",
        "https://example.invalid/b.zip",
    ]
