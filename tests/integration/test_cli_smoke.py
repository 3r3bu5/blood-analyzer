from pathlib import Path
import json

from bloodfilm.cli import main


def test_smoke_command_writes_missing_asset_and_sample_gap_report(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "smoke.json"
    config_path.write_text(
        '{"classifier":{"weights":"models/backbones/dinobloom-b.pth"},'
        '"quality":{"min_width":1,"min_height":1}}',
        encoding="utf-8",
    )

    exit_code = main(["smoke", "--config", str(config_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dinobloom_b"]["status"] == "blocked"
    assert "weights are not present" in report["dinobloom_b"]["blocker"]
    assert report["sample_image"]["status"] == "blocked"
