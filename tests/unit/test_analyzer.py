from pathlib import Path

from bloodfilm.analysis import CropClassifier, analyze_field_image
from bloodfilm.detection import Detection
from tests.helpers.png import write_rgb_png


class FixtureDetector:
    version = "fixture-detector-v1"

    def detect(self, image_rgb: object) -> list[Detection]:
        return [Detection(x1=1, y1=1, x2=5, y2=5, score=0.91)]


class ReviewDetector:
    version = "review-detector-v1"

    def detect(self, image_rgb: object) -> list[Detection]:
        return [
            Detection(
                x1=1,
                y1=1,
                x2=5,
                y2=5,
                score=0.91,
                box_status="review",
                box_review_reasons=("localization_quality",),
            )
        ]


class FixtureClassifier:
    version = "fixture-classifier-v1"

    def classify_crop(self, image_path: Path) -> CropClassifier:
        probabilities = [0.0] * 18
        probabilities[4] = 0.87
        return {
            "decision": {
                "status": "accepted",
                "label": "monocyte",
                "confidence": 0.87,
                "uncertainty_reasons": [],
            },
            "probabilities": [
                {"class_index": index, "class_code": f"class_{index}", "probability": value}
                for index, value in enumerate(probabilities)
            ],
        }


def test_analyze_field_image_returns_structured_cell_json(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    write_rgb_png(image, 8, 8, [(128, 128, 128)] * 64)
    crops = tmp_path / "crops"

    result = analyze_field_image(
        image,
        detector=FixtureDetector(),
        classifier=FixtureClassifier(),
        crop_dir=crops,
        crop_padding_ratio=0.25,
    )

    assert result["status"] == "succeeded"
    assert result["source_image"] == str(image)
    assert result["detector_version"] == "fixture-detector-v1"
    assert result["classifier_version"] == "fixture-classifier-v1"
    assert result["latency_ms"] >= 0
    assert len(result["cells"]) == 1
    cell = result["cells"][0]
    assert cell["cell_id"] == "field_d0000"
    assert cell["box"] == {"x1": 1, "y1": 1, "x2": 5, "y2": 5}
    assert cell["detector_confidence"] == 0.91
    assert cell["predicted_class"] == "monocyte"
    assert cell["classification_confidence"] == 0.87
    assert cell["box_status"] == "accepted"
    assert cell["box_review_reasons"] == []
    assert cell["classification_status"] == "accepted"
    assert cell["decision_status"] == "accepted"
    assert cell["detector_version"] == "fixture-detector-v1"
    assert cell["classifier_version"] == "fixture-classifier-v1"
    assert len(cell["probabilities"]) == 18
    assert Path(cell["crop_path"]).exists()


def test_analyze_field_image_keeps_flagged_localization_in_review(tmp_path: Path) -> None:
    image = tmp_path / "field.png"
    write_rgb_png(image, 8, 8, [(128, 128, 128)] * 64)

    result = analyze_field_image(
        image,
        detector=ReviewDetector(),
        classifier=FixtureClassifier(),
        crop_dir=tmp_path / "crops",
    )

    cell = result["cells"][0]
    assert cell["predicted_class"] == "monocyte"
    assert cell["classification_confidence"] == 0.87
    assert cell["box_status"] == "review"
    assert cell["box_review_reasons"] == ["localization_quality"]
    assert cell["classification_status"] == "review"
    assert cell["decision_status"] == "review"
    assert "localization_quality" in cell["uncertainty_reasons"]
