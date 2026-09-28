import json
from pathlib import Path

NOTEBOOK = Path("notebooks/05_kaggle_detector_recovery.ipynb")


def _text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_kaggle_detector_recovery_notebook_documents_end_to_end_flow() -> None:
    text = _text()

    for heading in [
        "# 05 - Kaggle M3.1 detector recovery",
        "## 0. Required Kaggle inputs",
        "## 4. Locate datasets",
        "## 6. Audit LeukemiaAttri once layout is known",
        "## 7. Build or attach unified YOLO data.yaml",
        "## 8. Dry-run the two training experiments",
        "## 9. Run training on GPU",
        "## 11. Download these outputs back to the repo",
    ]:
        assert heading in text


def test_kaggle_detector_recovery_notebook_contains_real_commands_and_guards() -> None:
    text = _text()

    for token in [
        "kaggle_train_detector_multidomain.py",
        "detector audit",
        "leukemia_attri_audit.json",
        "data/detection/multidomain/data.yaml",
        "RUN_TRAINING = False",
        "models/txl-pbc-yolo26n-v0.1/weights.pt",
        "detector_sweep_microscope_wbc2_field_roi.json",
        "Do not package",
    ]:
        assert token in text
