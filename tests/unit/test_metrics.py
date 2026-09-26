import pytest

from bloodfilm.evaluation.classifier import classification_report, confusion_matrix

LABELS = [0, 0, 1, 1, 2]
PREDS = [0, 1, 1, 1, 2]


def test_confusion_matrix_counts_label_pairs() -> None:
    assert confusion_matrix(PREDS, LABELS, num_classes=3) == [
        [1, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ]


def test_classification_report_matches_worked_example() -> None:
    report = classification_report(PREDS, LABELS, ["a", "b", "c"])

    assert report["accuracy"] == pytest.approx(0.8)
    assert report["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert report["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)
    assert report["weighted_f1"] == pytest.approx((2 / 3 * 2 + 0.8 * 2 + 1.0) / 5)
    per_class = {row["class_code"]: row for row in report["per_class"]}
    assert per_class["a"]["support"] == 2
    assert per_class["a"]["precision"] == pytest.approx(1.0)
    assert per_class["a"]["recall"] == pytest.approx(0.5)
    assert per_class["b"]["f1"] == pytest.approx(0.8)
    assert per_class["c"]["f1"] == pytest.approx(1.0)


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        classification_report([0, 1], [0], ["a", "b"])
