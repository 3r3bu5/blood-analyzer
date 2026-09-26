"""Stage A embedding cache: extract frozen-backbone embeddings once (TDD 9.4)."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bloodfilm.classification.dinobloom import PREPROCESSING
from bloodfilm.errors import InputNotFoundError
from bloodfilm.ml import require_torch
from bloodfilm.schemas import MLL23_CANONICAL_CLASSES, SplitName, SplitRow

CacheEntryStatus = Literal["ok", "skipped"]


@dataclass(frozen=True)
class CacheEntry:
    image_id: str
    image_path: str
    label_index: int
    split: SplitName
    status: CacheEntryStatus
    reason: str = ""


@dataclass(frozen=True)
class EmbeddingCache:
    entries: list[CacheEntry]
    embeddings: Any
    metadata: dict[str, Any]


def preprocessing_checksum() -> str:
    canonical = json.dumps(PREPROCESSING, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_embedding_cache(
    rows: list[SplitRow],
    *,
    dataset_root: Path | str,
    backbone: Callable[[Any], Any],
    preprocess: Callable[[Path], Any],
    weight_sha256: str,
    limit: int | None = None,
    progress_every: int = 5000,
) -> EmbeddingCache:
    torch = require_torch("Embedding extraction")
    root = Path(dataset_root)
    if not root.exists():
        raise InputNotFoundError(f"Dataset root not found: {root}")
    selected = rows if limit is None else rows[:limit]
    entries: list[CacheEntry] = []
    tensors: list[Any] = []
    for scanned, row in enumerate(selected, start=1):
        if scanned % progress_every == 0:
            print(f"embeddings: extracted {scanned}/{len(selected)} crops", file=sys.stderr)
        entries.append(_extract_row(row, root, backbone, preprocess, tensors))
    embeddings = torch.stack(tensors) if tensors else torch.empty((0, 768))
    ok_count = sum(1 for entry in entries if entry.status == "ok")
    metadata: dict[str, Any] = {
        "weight_sha256": weight_sha256,
        "preprocessing_sha256": preprocessing_checksum(),
        "row_count": len(entries),
        "ok_count": ok_count,
        "skipped_count": len(entries) - ok_count,
    }
    return EmbeddingCache(entries=entries, embeddings=embeddings, metadata=metadata)


def _extract_row(
    row: SplitRow,
    root: Path,
    backbone: Callable[[Any], Any],
    preprocess: Callable[[Path], Any],
    tensors: list[Any],
) -> CacheEntry:
    try:
        label_index = MLL23_CANONICAL_CLASSES.index(row.canonical_label)
    except ValueError:
        return CacheEntry(
            image_id=row.image_id,
            image_path=row.image_path,
            label_index=-1,
            split=row.split,
            status="skipped",
            reason=f"unknown label: {row.canonical_label}",
        )
    torch = require_torch("Embedding extraction")
    try:
        # No autograd: each stored embedding must not pin its forward graph,
        # or memory grows with every crop until the OOM killer intervenes.
        with torch.no_grad():
            batch = preprocess(root / row.image_path)
            embedding = backbone(batch)
        shape = [int(dim) for dim in embedding.shape]
    except Exception as exc:  # noqa: BLE001 - one bad crop must not abort the cache run
        return CacheEntry(
            image_id=row.image_id,
            image_path=row.image_path,
            label_index=label_index,
            split=row.split,
            status="skipped",
            reason=f"unreadable image: {exc}",
        )
    if shape != [1, 768]:
        return CacheEntry(
            image_id=row.image_id,
            image_path=row.image_path,
            label_index=label_index,
            split=row.split,
            status="skipped",
            reason=f"unexpected embedding shape: {shape}",
        )
    tensors.append(embedding[0])
    return CacheEntry(
        image_id=row.image_id,
        image_path=row.image_path,
        label_index=label_index,
        split=row.split,
        status="ok",
    )


def save_embedding_cache(cache: EmbeddingCache, path: Path | str) -> Path:
    torch = require_torch("Embedding extraction")
    payload = {
        "embeddings": cache.embeddings,
        "entries": [entry.__dict__ for entry in cache.entries],
        "metadata": cache.metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_embedding_cache(path: Path | str) -> EmbeddingCache:
    torch = require_torch("Embedding extraction")
    location = Path(path)
    if not location.exists():
        raise InputNotFoundError(f"Embedding cache not found: {location}")
    payload = torch.load(location, map_location="cpu", weights_only=True)
    entries = [CacheEntry(**entry) for entry in payload["entries"]]
    return EmbeddingCache(
        entries=entries, embeddings=payload["embeddings"], metadata=payload["metadata"]
    )
