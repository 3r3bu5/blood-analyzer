import json
import subprocess
from pathlib import Path

import pytest

import scripts.kaggle_train_detector_multidomain as train_module


def _record_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = list(args[0]) if args else []
        calls.append([str(part) for part in command])
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(train_module.subprocess, "run", fake_run)
    return calls


def test_train_runs_train_then_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_run(monkeypatch)
    plan_output = tmp_path / "plan.json"

    exit_code = train_module.main(
        [
            "--data-yaml",
            "data/detection/multidomain/data.yaml",
            "--project-dir",
            str(tmp_path / "runs"),
            "--experiment",
            "B_clean_pretrained",
            "--plan-output",
            str(plan_output),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][:3] == ["yolo", "detect", "train"]
    assert "model=yolo26n.pt" in calls[0]
    assert calls[1][:3] == ["yolo", "detect", "val"]
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert payload["train_skipped"] is False


def test_skip_train_runs_only_val_from_existing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_run(monkeypatch)
    project = tmp_path / "runs"
    weights = project / "wbc-detector-multidomain-yolo26n-a" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake-weights")
    plan_output = tmp_path / "plan.json"

    exit_code = train_module.main(
        [
            "--data-yaml",
            "data/detection/multidomain/data.yaml",
            "--project-dir",
            str(project),
            "--experiment",
            "A_finetune_txl_pbc",
            "--plan-output",
            str(plan_output),
            "--skip-train",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][:3] == ["yolo", "detect", "val"]
    assert f"model={weights}" in calls[0]
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert payload["train_skipped"] is True


def test_skip_train_fails_fast_without_existing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_run(monkeypatch)

    with pytest.raises(SystemExit, match="no existing weights"):
        train_module.main(
            [
                "--data-yaml",
                "data/detection/multidomain/data.yaml",
                "--project-dir",
                str(tmp_path / "runs"),
                "--experiment",
                "A_finetune_txl_pbc",
                "--plan-output",
                str(tmp_path / "plan.json"),
                "--skip-train",
            ]
        )
