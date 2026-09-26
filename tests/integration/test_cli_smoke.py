import json
from pathlib import Path

from bloodfilm.cli import main


def test_smoke_command_writes_missing_asset_and_sample_gap_report(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "smoke.json"
    config_path.write_text(
        '{"classifier":{"weights":' + json.dumps(str(tmp_path / "absent.pth")) + "},"
        '"quality":{"min_width":1,"min_height":1}}',
        encoding="utf-8",
    )

    exit_code = main(["smoke", "--config", str(config_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dinobloom_b"]["status"] == "blocked"
    assert "weights are not present" in report["dinobloom_b"]["blocker"]
    assert report["sample_image"]["status"] == "blocked"


def test_smoke_command_embeds_model_error_instead_of_aborting(tmp_path: Path) -> None:
    weights = tmp_path / "dinobloom-b.pth"
    weights.write_bytes(b"not-a-real-checkpoint")
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "smoke.json"
    config_path.write_text(
        f'{{"classifier":{{"weights":{str(weights)!r}}},'
        '"quality":{"min_width":1,"min_height":1}}'.replace("'", '"'),
        encoding="utf-8",
    )

    exit_code = main(["smoke", "--config", str(config_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dinobloom_b"]["status"] == "error"
    assert report["dinobloom_b"]["error_code"] == "MODEL_LOAD_ERROR"
