import json
from pathlib import Path

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
