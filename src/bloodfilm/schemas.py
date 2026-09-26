from __future__ import annotations

from dataclasses import dataclass

MLL23_CANONICAL_CLASSES = [
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


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    source_folder: str
    canonical_label: str
    sha256: str
    width: int
    height: int
    mode: str
    patient_or_source_group: str


@dataclass(frozen=True)
class InvalidManifestRow:
    image_path: str
    source_folder: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SplitRow:
    image_id: str
    image_path: str
    canonical_label: str
    patient_or_source_group: str
    split: str
