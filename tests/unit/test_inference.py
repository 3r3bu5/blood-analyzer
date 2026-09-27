from pathlib import Path

import pytest

from bloodfilm.classification.inference import classify_probabilities
from bloodfilm.classification.uncertainty import UncertaintyPolicy


def test_classify_probabilities_returns_ranked_research_only_payload() -> None:
    result = classify_probabilities(
        probabilities=[0.82, 0.12, 0.06],
        class_names=["basophil", "eosinophil", "monocyte"],
        policy=UncertaintyPolicy(min_accept_confidence=0.80),
        image_path=Path("cell.png"),
        bundle_name="mll23-dinobloom-b-mlp-v0.1",
    )

    assert result["research_only"] is True
    assert result["image"] == "cell.png"
    assert result["decision"]["status"] == "accepted"
    assert result["decision"]["label"] == "basophil"
    assert [row["class_code"] for row in result["top_predictions"]] == [
        "basophil",
        "eosinophil",
        "monocyte",
    ]


def test_classify_probabilities_returns_all_18_probabilities() -> None:
    class_names = [f"class_{index}" for index in range(18)]
    probabilities = [0.0] * 18
    probabilities[17] = 1.0

    result = classify_probabilities(
        probabilities=probabilities,
        class_names=class_names,
        policy=UncertaintyPolicy(min_accept_confidence=0.80),
        image_path=Path("cell.png"),
        bundle_name="mll23-dinobloom-b-mlp-v0.1",
    )

    assert len(result["probabilities"]) == 18
    assert [row["class_code"] for row in result["probabilities"]] == class_names
    assert result["top_predictions"][0]["class_code"] == "class_17"


def test_preprocess_crop_rejects_corrupt_image(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    from bloodfilm.classification.dinobloom import preprocess_crop
    from bloodfilm.errors import ModelLoadError

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    with pytest.raises(ModelLoadError, match="decode"):
        preprocess_crop(corrupt)


def test_preprocess_crop_is_deterministic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    from bloodfilm.classification.dinobloom import preprocess_crop
    from tests.helpers.png import write_rgb_png

    image = tmp_path / "cell.png"
    write_rgb_png(image, 4, 4, [(200, 30, 30)] * 16)

    first = preprocess_crop(image)
    second = preprocess_crop(image)

    assert torch.equal(first, second)


def test_classification_modules_do_not_import_torch_at_top_level() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for relative in [
        "src/bloodfilm/cli.py",
        "src/bloodfilm/classification/inference.py",
        "src/bloodfilm/classification/bundle.py",
        "src/bloodfilm/classification/reports.py",
    ]:
        text = (repo_root / relative).read_text(encoding="utf-8")
        assert "\nimport torch\n" not in text
        assert "\nimport torch " not in text
        assert "\nfrom torch" not in text
