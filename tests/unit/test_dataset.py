import pytest

from bloodfilm.data.dataset import ManifestImageDataset, default_image_loader
from bloodfilm.errors import ModelLoadError
from bloodfilm.schemas import ManifestRow


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
    try:
        default_image_loader(_row("aaa", "basophil"))
    except ModelLoadError as exc:
        assert "torch" in str(exc).lower() or "pillow" in str(exc).lower()
    else:
        pytest.skip("ML dependencies are installed; default loader is operational")
