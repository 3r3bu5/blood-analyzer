import pytest

from bloodfilm.detection import Detection, validate_detections
from bloodfilm.errors import ConfigError


def test_detection_to_dict_uses_wbc_candidate_label() -> None:
    detection = Detection(x1=1, y1=2, x2=11, y2=12, score=0.9)

    assert detection.to_dict() == {
        "x1": 1,
        "y1": 2,
        "x2": 11,
        "y2": 12,
        "score": 0.9,
        "label": "wbc_candidate",
    }


def test_validate_detections_accepts_bounded_boxes() -> None:
    detections = [Detection(x1=0, y1=0, x2=10, y2=10, score=1.0)]

    assert validate_detections(detections, image_width=10, image_height=10) == detections


@pytest.mark.parametrize(
    "detection",
    [
        Detection(x1=-1, y1=0, x2=10, y2=10, score=0.5),
        Detection(x1=0, y1=0, x2=11, y2=10, score=0.5),
        Detection(x1=1, y1=0, x2=1, y2=10, score=0.5),
        Detection(x1=0, y1=0, x2=10, y2=10, score=1.1),
    ],
)
def test_validate_detections_rejects_invalid_boxes(detection: Detection) -> None:
    with pytest.raises(ConfigError):
        validate_detections([detection], image_width=10, image_height=10)
