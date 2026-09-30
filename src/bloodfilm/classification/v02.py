from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from bloodfilm.errors import ConfigError
from bloodfilm.util import sha256_file

MLL23_CLASSES = [
    "basophil",
    "eosinophil",
    "neutrophil_band",
    "neutrophil_segmented",
    "monocyte",
    "lymphocyte_typical",
    "lymphocyte_reactive",
    "lymphocyte_large_granular",
    "lymphocyte_neoplastic_other",
    "hairy_cell",
    "plasma_cell",
    "smudge_cell",
    "myeloblast",
    "promyelocyte",
    "promyelocyte_atypical",
    "myelocyte",
    "metamyelocyte",
    "normoblast",
]

EXACT_SOURCE_MAPPINGS = {
    "basophil": "basophil",
    "eosinophil": "eosinophil",
    "monocyte": "monocyte",
    "myeloblast": "myeloblast",
    "myelocyte": "myelocyte",
    "metamyelocyte": "metamyelocyte",
    "lymphocyte": "lymphocyte_typical",
    "abnormal_promyelocyte": "promyelocyte_atypical",
}

PROVENANCE_REVIEW_MAPPINGS = {
    "lymphocyte": "lymphocyte_typical",
    "abnormal_promyelocyte": "promyelocyte_atypical",
}

PARTIAL_SOURCE_MAPPINGS = {"neutrophil": ("neutrophil_band", "neutrophil_segmented")}

UNMAPPED_SOURCE_CLASSES = {
    "lymphoblast",
    "atypical_lymphocyte",
    "promonocyte",
    "monoblast",
    "none",
    "artifact",
    "none_artifact",
}


@dataclass(frozen=True)
class MappingDecision:
    source_class: str
    normalized_source_class: str
    mapping_type: str
    target_class: str | None
    partial_target_classes: tuple[str, ...]
    requires_provenance_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_class": self.source_class,
            "normalized_source_class": self.normalized_source_class,
            "mapping_type": self.mapping_type,
            "target_class": self.target_class,
            "partial_target_classes": list(self.partial_target_classes),
            "requires_provenance_review": self.requires_provenance_review,
        }


@dataclass(frozen=True)
class SquareCrop:
    crop_box: tuple[int, int, int, int]
    paste_box: tuple[int, int, int, int]
    output_size: int
    padding: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_box": list(self.crop_box),
            "paste_box": list(self.paste_box),
            "output_size": self.output_size,
            "padding": list(self.padding),
        }


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def verify_mll23_class_order(class_names: Sequence[str]) -> None:
    actual = list(class_names)
    if actual != MLL23_CLASSES:
        raise ConfigError(
            "Protected classifier taxonomy order differs from the expected 18-class MLL23 order: "
            f"expected={MLL23_CLASSES!r} actual={actual!r}"
        )


def decide_source_mapping(source_class: str) -> MappingDecision:
    normalized = normalize_class_name(source_class)
    if normalized in EXACT_SOURCE_MAPPINGS:
        target = EXACT_SOURCE_MAPPINGS[normalized]
        return MappingDecision(
            source_class=source_class,
            normalized_source_class=normalized,
            mapping_type="exact",
            target_class=target,
            partial_target_classes=(),
            requires_provenance_review=PROVENANCE_REVIEW_MAPPINGS.get(normalized) == target,
        )
    if normalized in PARTIAL_SOURCE_MAPPINGS:
        return MappingDecision(
            source_class=source_class,
            normalized_source_class=normalized,
            mapping_type="partial",
            target_class=None,
            partial_target_classes=PARTIAL_SOURCE_MAPPINGS[normalized],
        )
    return MappingDecision(
        source_class=source_class,
        normalized_source_class=normalized,
        mapping_type="unmapped_ood"
        if normalized in UNMAPPED_SOURCE_CLASSES
        else "unmapped_unknown",
        target_class=None,
        partial_target_classes=(),
    )


def partial_label_loss(logits: Any, target_indices: Sequence[int], torch: Any | None = None) -> Any:
    if not target_indices:
        raise ValueError("target_indices must not be empty")
    if torch is None:
        import torch as torch_module

        torch = torch_module
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_probs[:, list(target_indices)]
    return -torch.logsumexp(selected, dim=-1).mean()


def square_crop_geometry(
    box: Sequence[float], image_width: int, image_height: int, padding_ratio: float
) -> SquareCrop:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")
    x1, y1, x2, y2 = [float(value) for value in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area")
    width = x2 - x1
    height = y2 - y1
    expanded = max(width, height) * (1.0 + 2.0 * padding_ratio)
    side = max(1, int(math.ceil(expanded)))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    raw_x1 = int(math.floor(center_x - side / 2.0))
    raw_y1 = int(math.floor(center_y - side / 2.0))
    raw_x2 = raw_x1 + side
    raw_y2 = raw_y1 + side
    crop_x1 = max(0, raw_x1)
    crop_y1 = max(0, raw_y1)
    crop_x2 = min(image_width, raw_x2)
    crop_y2 = min(image_height, raw_y2)
    pad_left = crop_x1 - raw_x1
    pad_top = crop_y1 - raw_y1
    pad_right = raw_x2 - crop_x2
    pad_bottom = raw_y2 - crop_y2
    paste_box = (pad_left, pad_top, pad_left + crop_x2 - crop_x1, pad_top + crop_y2 - crop_y1)
    return SquareCrop(
        crop_box=(crop_x1, crop_y1, crop_x2, crop_y2),
        paste_box=paste_box,
        output_size=side,
        padding=(pad_left, pad_top, pad_right, pad_bottom),
    )


def group_aware_split(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
) -> list[str]:
    if not rows:
        return []
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError("split proportions must leave a non-empty test proportion")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = str(row.get(group_key) or row.get("source_image") or index)
        groups.setdefault(group, []).append(index)
    ordered_groups = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered_groups)
    total = len(rows)
    train_target = total * train
    validation_target = total * validation
    assignments = ["test"] * total
    train_count = 0
    validation_count = 0
    for group in ordered_groups:
        indices = groups[group]
        if train_count < train_target:
            split = "train"
            train_count += len(indices)
        elif validation_count < validation_target:
            split = "validation"
            validation_count += len(indices)
        else:
            split = "test"
        for index in indices:
            assignments[index] = split
    return assignments


def split_leakage_report(
    rows: Sequence[dict[str, Any]], split_key: str, leakage_keys: Iterable[str]
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for key in leakage_keys:
        seen: dict[str, set[str]] = {}
        for row in rows:
            value = row.get(key)
            split = row.get(split_key)
            if value in (None, "") or split in (None, ""):
                continue
            seen.setdefault(str(value), set()).add(str(split))
        for value, splits in sorted(seen.items()):
            if len(splits) > 1:
                violations.append({"field": key, "value": value, "splits": sorted(splits)})
    return {"status": "fail" if violations else "ok", "violations": violations}


def hash_files(root: Path | str) -> dict[str, str]:
    path = Path(root)
    if path.is_file():
        return {path.name: sha256_file(path)}
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def assert_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        raise ConfigError("Protected artifact hashes changed")


def probability_schema(
    probabilities: Sequence[float], class_names: Sequence[str], *, tolerance: float = 1e-4
) -> dict[str, Any]:
    if len(probabilities) != len(class_names):
        raise ValueError("probability count must match class count")
    if any(not math.isfinite(float(value)) for value in probabilities):
        raise ValueError("probabilities must be finite")
    total = sum(float(value) for value in probabilities)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"probabilities must sum to 1 within {tolerance}, got {total}")
    ranked = sorted(enumerate(probabilities), key=lambda item: float(item[1]), reverse=True)
    top_index, top_probability = ranked[0]
    return {
        "predicted_class": None,
        "top_class": class_names[top_index],
        "top_probability": float(top_probability),
        "top_k": [
            {"class_index": index, "class_code": class_names[index], "probability": float(value)}
            for index, value in ranked[:5]
        ],
        "probabilities": [
            {"class_index": index, "class_code": name, "probability": float(probabilities[index])}
            for index, name in enumerate(class_names)
        ],
    }


def acceptance_status(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if metrics.get("protected_hashes_changed"):
        failures.append("protected_hashes_changed")
    if metrics.get("split_leakage_detected"):
        failures.append("split_leakage_detected")
    if metrics.get("all_18_output_classes_required") is False:
        failures.append("missing_18_output_classes")
    accuracy_drop = float(metrics.get("mll23_accuracy_drop", 0.0))
    if accuracy_drop > float(gates.get("mll23_accuracy_max_drop", 0.005)):
        failures.append("mll23_accuracy_regression")
    macro_f1_drop = float(metrics.get("mll23_macro_f1_drop", 0.0))
    if macro_f1_drop > float(gates.get("mll23_macro_f1_max_drop", 0.010)):
        failures.append("mll23_macro_f1_regression")
    external_gain = float(metrics.get("external_exact_macro_f1_improvement", 0.0))
    if external_gain < float(gates.get("external_exact_macro_f1_min_improvement", 0.05)):
        failures.append("external_no_improvement")
    if gates.get("high_confidence_error_must_not_increase", True) and metrics.get(
        "high_confidence_error_increased"
    ):
        failures.append("high_confidence_error_increased")
    if gates.get("incorrect_accepted_must_not_increase", True) and metrics.get(
        "incorrect_accepted_increased"
    ):
        failures.append("incorrect_accepted_increased")
    if metrics.get("calibration_materially_worse"):
        failures.append("calibration_materially_worse")
    if not failures:
        status = "accepted_candidate"
    elif any(item.startswith("mll23") for item in failures):
        status = "rejected_mll23_regression"
    elif "external_no_improvement" in failures:
        status = "rejected_external_no_improvement"
    elif "calibration_materially_worse" in failures:
        status = "rejected_calibration"
    else:
        status = "rejected_integrity"
    return {"status": status, "failures": failures}


def stable_sample_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
