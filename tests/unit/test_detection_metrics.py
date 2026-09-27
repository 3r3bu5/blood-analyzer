from bloodfilm.detection import Detection
from bloodfilm.detection.metrics import evaluate_detections, select_confidence_threshold


def test_evaluate_detections_reports_recall_precision_and_map() -> None:
    report = evaluate_detections(
        predictions_by_image={
            "field_a": [
                Detection(0, 0, 10, 10, 0.9),
                Detection(20, 20, 30, 30, 0.4),
            ],
            "field_b": [Detection(0, 0, 5, 5, 0.8)],
        },
        truth_by_image={
            "field_a": [Detection(0, 0, 10, 10, 1.0)],
            "field_b": [Detection(20, 20, 25, 25, 1.0)],
        },
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )

    assert report["recall"] == 0.5
    assert report["precision"] == 0.5
    assert report["false_positives_per_image"] == 0.5
    assert report["missed_wbc_rate"] == 0.5
    assert report["mAP50"] == report["ap50"]
    assert 0.0 <= report["mAP50_95"] <= report["mAP50"] <= 1.0
    assert report["missed_wbc_count"] == 1


def test_select_confidence_threshold_prioritizes_recall_under_false_crop_budget() -> None:
    candidates = [
        {"confidence_threshold": 0.1, "recall": 1.0, "false_positives_per_image": 4.0},
        {"confidence_threshold": 0.25, "recall": 1.0, "false_positives_per_image": 1.5},
        {"confidence_threshold": 0.5, "recall": 0.8, "false_positives_per_image": 0.2},
    ]

    selection = select_confidence_threshold(
        candidates,
        target_recall=0.95,
        max_false_positives_per_image=2.0,
    )

    assert selection["confidence_threshold"] == 0.25
    assert selection["selection_reason"] == "meets_recall_and_false_crop_budget"
