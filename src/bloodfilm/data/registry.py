from __future__ import annotations

from pathlib import Path
from typing import Any

from bloodfilm.documents import load_document


def load_asset_registry(path: Path | str) -> dict[str, Any]:
    return load_document(path)
