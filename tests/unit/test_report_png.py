from bloodfilm.classification.png import write_matrix_png, write_reliability_png


def test_write_matrix_png_produces_png_signature(tmp_path) -> None:
    path = tmp_path / "matrix.png"
    write_matrix_png(path, [[2, 0], [0, 1]])

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_reliability_png_produces_png_signature(tmp_path) -> None:
    path = tmp_path / "reliability.png"
    write_reliability_png(
        path,
        [
            {
                "bin_index": 0,
                "low": 0.0,
                "high": 0.5,
                "count": 1,
                "accuracy": 1.0,
                "confidence": 0.8,
                "absolute_gap": 0.2,
                "weight": 0.5,
            }
        ],
    )

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
