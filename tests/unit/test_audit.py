from pathlib import Path

from bloodfilm.data.audit import audit_dataset
from tests.helpers.png import write_rgb_png


def test_audit_counts_classes_dimensions_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "mll23"
    (root / "Basophil").mkdir(parents=True)
    (root / "Monocyte").mkdir(parents=True)
    write_rgb_png(root / "Basophil" / "cell_a.png", 2, 2, [(1, 2, 3)] * 4)
    write_rgb_png(root / "Basophil" / "cell_b.png", 2, 2, [(1, 2, 3)] * 4)
    write_rgb_png(root / "Monocyte" / "cell_c.png", 4, 1, [(9, 9, 9)] * 4)
    (root / "Monocyte" / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / "stray.txt").write_text("top-level file", encoding="utf-8")

    report = audit_dataset(root)

    assert report["total_files"] == 5
    assert report["class_counts"] == {"Basophil": 2, "Monocyte": 2, "top_level": 1}
    assert report["dimensions"]["min_width"] == 2
    assert report["dimensions"]["max_width"] == 4
    assert len(report["duplicates"]) == 1
    assert sorted(report["duplicates"][0]) == [
        str(root / "Basophil" / "cell_a.png"),
        str(root / "Basophil" / "cell_b.png"),
    ]
    assert report["unreadable"] == sorted(
        [str(root / "Monocyte" / "notes.txt"), str(root / "stray.txt")]
    )
