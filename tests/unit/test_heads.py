import pytest

from bloodfilm.classification.heads import build_head
from bloodfilm.errors import ConfigError

torch = pytest.importorskip("torch")


def test_linear_head_maps_768_to_classes() -> None:
    head = build_head("linear", num_classes=18)

    assert list(head(torch.zeros(4, 768)).shape) == [4, 18]


def test_mlp_head_matches_tdd_structure() -> None:
    head = build_head("mlp", num_classes=18)

    kinds = [type(module).__name__ for module in head]
    assert kinds == ["LayerNorm", "Linear", "GELU", "Dropout", "Linear"]
    assert head[1].in_features == 768
    assert head[1].out_features == 512
    assert list(head(torch.zeros(4, 768)).shape) == [4, 18]


def test_cosine_head_scales_normalized_similarities() -> None:
    head = build_head("cosine", num_classes=18)

    assert float(head.log_scale.exp()) > 0
    assert list(head(torch.zeros(4, 768)).shape) == [4, 18]


def test_unknown_head_name_raises() -> None:
    with pytest.raises(ConfigError):
        build_head("transformer", num_classes=18)
