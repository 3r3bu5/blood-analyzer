from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bloodfilm.documents import load_config_document
from bloodfilm.errors import MappingError


@dataclass(frozen=True)
class LabelMapping:
    source: str
    canonical_classes: list[str]
    mappings: dict[str, str]

    def canonical_for(self, source_label: str) -> str | None:
        return self.mappings.get(source_label) or self.mappings.get(source_label.strip())


def load_label_mapping(path: Path | str) -> LabelMapping:
    raw = load_config_document(path)
    canonical_raw = raw.get("canonical_classes", [])
    mappings_raw = raw.get("mappings", {})
    if not isinstance(canonical_raw, list) or not all(
        isinstance(item, str) for item in canonical_raw
    ):
        raise MappingError("canonical_classes must be a list of strings")
    if not isinstance(mappings_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mappings_raw.items()
    ):
        raise MappingError("mappings must be an object of string keys and values")
    canonical_classes = list(canonical_raw)
    unknown_targets = sorted(
        {value for value in mappings_raw.values() if value not in canonical_classes}
    )
    if unknown_targets:
        raise MappingError(f"Mappings reference unknown canonical classes: {unknown_targets}")
    return LabelMapping(
        source=str(raw.get("source", "unknown")),
        canonical_classes=canonical_classes,
        mappings=dict(mappings_raw),
    )
