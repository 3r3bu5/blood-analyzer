import pytest

torch = pytest.importorskip("torch")

from bloodfilm.classification.heads import build_head
from bloodfilm.training.classifier import (
    TrainConfig,
    compute_class_weights,
    head_accuracy,
    train_head,
)
from tests.helpers.fake_backbone import fake_embedding_bank


def test_weighted_loss_balances_rare_classes() -> None:
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)

    assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])


def test_linear_head_fits_separable_embeddings() -> None:
    embeddings, labels = fake_embedding_bank()
    model = build_head("linear", num_classes=3)

    history = train_head(
        model, embeddings, labels, TrainConfig(epochs=30, learning_rate=0.05), num_classes=3
    )

    assert history["loss"][-1] < history["loss"][0]
    assert head_accuracy(model, embeddings, labels) == pytest.approx(1.0)


def test_head_accuracy_rejects_empty_labels() -> None:
    model = build_head("linear", num_classes=3)

    with pytest.raises(ValueError):
        head_accuracy(model, torch.zeros(0, 768), [])
