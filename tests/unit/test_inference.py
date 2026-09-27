from pathlib import Path

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
