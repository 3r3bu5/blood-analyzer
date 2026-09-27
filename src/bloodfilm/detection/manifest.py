from __future__ import annotations

import csv
from pathlib import Path


def assert_no_locked_target_leakage(
    manifest_path: Path | str, *, locked_hashes: set[str]
) -> list[str]:
    """Return sample IDs whose image hash leaks locked validation images into training."""
    violations: list[str] = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_sha256") in locked_hashes and row.get("unified_split") == "train":
                violations.append(row.get("sample_id", "<unknown>"))
    return violations
