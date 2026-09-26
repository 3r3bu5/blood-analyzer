from pathlib import Path

import pytest

from bloodfilm.data.dataset import ManifestImageDataset, default_image_loader
from bloodfilm.errors import ConfigError, ModelLoadError
from bloodfilm.schemas import ManifestRow
from tests.helpers.png import write_rgb_png


def _ml_available() -> bool:
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _row(image_id: str, label: str) -> ManifestRow:
    return ManifestRow(
        image_id=image_id,
        image_path=f"class/{image_id}.png",
        source_folder="class",
        canonical_label=label,
        sha256=image_id,
        width=2,
        height=2,
        mode="RGB",
        patient_or_source_group=f"ungrouped:{image_id}",
    )


def test_manifest_dataset_uses_loader_and_label_index() -> None:
    rows = [_row("aaa", "monocyte"), _row("bbb", "basophil")]
    seen: list[str] = []

    def recording_loader(row: ManifestRow) -> str:
        seen.append(row.image_id)
        return "pixels"

    dataset = ManifestImageDataset(rows, ["basophil", "monocyte"], loader=recording_loader)

    assert len(dataset) == 2
    pixels, image_id, label_index = dataset[1]
    assert (pixels, image_id) == ("pixels", "bbb")
    assert label_index == 0
    assert seen == ["bbb"]


def test_manifest_dataset_rejects_unknown_label() -> None:
    with pytest.raises(ModelLoadError):
        ManifestImageDataset([_row("aaa", "mystery")], ["basophil"])


def test_default_loader_requires_ml_dependencies() -> None:
    if _ml_available():
        pytest.skip("ML dependencies are installed; see the operational test below")
    with pytest.raises((ConfigError, ModelLoadError)) as exc_info:
        default_image_loader(_row("aaa", "basophil"))
    assert "ml extras" in str(exc_info.value).lower()


def test_default_loader_returns_channel_first_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    image_path = tmp_path / "cell.png"
    write_rgb_png(image_path, 2, 2, [(1, 2, 3)] * 4)
    row = ManifestRow(
        image_id="cell",
        image_path=str(image_path),
        source_folder="class",
        canonical_label="basophil",
        sha256="cell",
        width=2,
        height=2,
        mode="RGB",
        patient_or_source_group="ungrouped:cell",
    )

    tensor = default_image_loader(row)

    assert list(tensor.shape) == [3, 2, 2]
    assert tensor.dtype == torch.uint8
    assert tensor[:, 0, 0].tolist() == [1, 2, 3]
