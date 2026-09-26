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


def test_expected_calibration_error_is_zero_for_matched_confidence() -> None:
    from bloodfilm.evaluation.classifier import expected_calibration_error

    probabilities = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    labels = [0, 1, 0, 1]

    assert expected_calibration_error(probabilities, labels, bins=1) == 0.0


def test_expected_calibration_error_is_one_for_fully_wrong_confident_predictions() -> None:
    from bloodfilm.evaluation.classifier import expected_calibration_error

    probabilities = [[1.0, 0.0], [1.0, 0.0]]
    labels = [1, 1]

    assert expected_calibration_error(probabilities, labels, bins=1) == 1.0


def test_expected_calibration_error_rejects_mismatched_lengths() -> None:
    from bloodfilm.evaluation.classifier import expected_calibration_error

    with pytest.raises(ValueError):
        expected_calibration_error([[1.0, 0.0]], [], bins=1)


def test_fit_temperature_reduces_nll_on_fit_set() -> None:
    torch = pytest.importorskip("torch")
    from bloodfilm.evaluation.classifier import fit_temperature

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 3, generator=generator) * 3.0
    labels = torch.tensor([0, 1, 2] * 13 + [0])

    temperature = fit_temperature(logits, labels)

    assert temperature > 0.0
    assert torch.isfinite(torch.tensor(temperature))
    before = torch.nn.functional.cross_entropy(logits, labels).item()
    after = torch.nn.functional.cross_entropy(logits / temperature, labels).item()
    assert after <= before + 1e-6


def test_top_k_accuracy_counts_top2_hits() -> None:
    torch = pytest.importorskip("torch")
    from bloodfilm.evaluation.classifier import top_k_accuracy

    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.1, 1.0]])
    labels = torch.tensor([1, 2])

    assert top_k_accuracy(logits, labels, k=1) == 0.5
    assert top_k_accuracy(logits, labels, k=2) == 1.0
