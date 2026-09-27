import json
from pathlib import Path

NOTEBOOK = Path("notebooks/04_e2e_field_analysis.ipynb")


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_e2e_notebook_documents_each_pipeline_stage() -> None:
    text = _notebook_text()

    for heading in [
        "# 04 - End-to-end field analysis",
        "## 1. Environment and assets",
        "## 2. Select one field image",
        "## 3. Load detector and classifier",
        "## 4. Detect WBC candidates",
        "## 5. Visualize boxes and heatmaps",
        "## 6. Crop and classify detected cells",
        "## 7. Insights and export",
    ]:
        assert heading in text


def test_e2e_notebook_contains_runtime_and_visualization_code() -> None:
    text = _notebook_text()

    for token in [
        "YOLO(",
        "classify_crop(",
        "models/txl-pbc-yolo26n-v0.1",
        "models/mll23-dinobloom-b-mlp-v0.1",
        "models/backbones/dinobloom-b.pth",
        "imshow",
        "Rectangle",
        "heatmap",
        "outputs/e2e/field_analysis.json",
        "all 18 probabilities",
    ]:
        assert token in text
