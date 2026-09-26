from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bloodfilm.errors import ConfigError


def load_config_document(path: Path | str) -> dict[str, Any]:
    document_path = Path(path)
    try:
        text = document_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Document not found: {document_path}") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{document_path} must be JSON-compatible YAML in the M0 scaffold"
        ) from exc

    if not isinstance(value, dict):
        raise ConfigError(f"{document_path} must contain an object at the top level")
    return value


def write_json(path: Path | str, value: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
