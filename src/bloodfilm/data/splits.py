from __future__ import annotations

import hashlib
from pathlib import Path

from bloodfilm.schemas import ManifestRow, SplitName, SplitRow, is_ungrouped_group
from bloodfilm.util import write_csv_rows


def create_split_manifest(
    rows: list[ManifestRow], *, seed: int = 42, train: float = 0.70, validation: float = 0.15
) -> list[SplitRow]:
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError(
            "train and validation proportions must be positive and leave room for test"
        )
    split_rows: list[SplitRow] = []
    for row in rows:
        fraction = _stable_fraction(f"{seed}:{row.patient_or_source_group}")
        split: SplitName
        if fraction < train:
            split = "train"
        elif fraction < train + validation:
            split = "validation"
        else:
            split = "test"
        split_rows.append(
            SplitRow(
                image_id=row.image_id,
                image_path=row.image_path,
                canonical_label=row.canonical_label,
                patient_or_source_group=row.patient_or_source_group,
                split=split,
            )
        )
    return split_rows


def create_leakage_report(rows: list[SplitRow]) -> dict[str, object]:
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row.patient_or_source_group, set()).add(row.split)
    leaking = [
        {"patient_or_source_group": group, "splits": sorted(splits)}
        for group, splits in sorted(splits_by_group.items())
        if len(splits) > 1
    ]
    assessable = any(not is_ungrouped_group(group) for group in splits_by_group)
    return {
        "group_count": len(splits_by_group),
        "leaking_group_count": len(leaking),
        "leaking_groups": leaking,
        "leakage_free": (not leaking) if assessable else None,
        "grouping_status": _grouping_status(splits_by_group),
        "independence_claim": False,
    }


def _grouping_status(splits_by_group: dict[str, set[str]]) -> str:
    if not splits_by_group:
        return "empty_manifest"
    if all(is_ungrouped_group(group) for group in splits_by_group):
        return "unverified_image_level_surrogate"
    return "caller_provided_group_keys_not_independently_verified"


def write_split_manifest(path: Path | str, rows: list[SplitRow]) -> None:
    write_csv_rows(path, list(SplitRow.__dataclass_fields__), [row.__dict__ for row in rows])


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(0xFFFFFFFFFFFFFFFF)
