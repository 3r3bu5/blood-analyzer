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
        "## 4. Locate or download datasets",
        "## 6. Inspect LeukemiaAttri COCO-domain layout",
        "## 7. Build or attach unified YOLO data.yaml",
        "## 8. Dry-run the two training experiments",
        "## 9. Run training on GPU",
        "## 10. Compare experiments without selecting",
        "## 12. Download these outputs back to the repo",
    ]:
        assert heading in text


def test_kaggle_detector_recovery_notebook_contains_real_commands_and_guards() -> None:
    text = _text()

    for token in [
        "kaggle_train_detector_multidomain.py",
        "kaggle_compare_detector_multidomain.py",
        "RUN_COMPARISON = False",
        "detector_multidomain_comparison.json",
        "kaggle_download_detector_data.py",
        "RUN_DATA_DOWNLOAD = False",
        "--leukemia-annotation-format coco_domain",
        "data/detection/multidomain/data.yaml",
        "RUN_TRAINING = False",
        "models/txl-pbc-yolo26n-v0.1/weights.pt",
        "detector_sweep_microscope_wbc2_field_roi.json",
        "Do not package",
    ]:
        assert token in text
