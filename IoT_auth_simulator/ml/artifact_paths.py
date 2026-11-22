"""Venue-neutral path resolution for frozen scientific artifacts.

The frozen files keep their original paths and bytes.  This module provides
stable scientific aliases without making reusable callers depend on a
conference-named directory layout.
"""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_PATH = Path("metadata/scientific-version-registry.json")


def repository_root(start: Path | None = None) -> Path:
    """Return the nearest repository root containing the version registry."""

    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / REGISTRY_PATH).is_file():
            return parent
    raise FileNotFoundError(f"cannot locate {REGISTRY_PATH}")


def load_version_registry(root: Path | None = None) -> dict[str, object]:
    base = root.resolve() if root else repository_root()
    return json.loads((base / REGISTRY_PATH).read_text(encoding="utf-8"))


def resolve_scientific_path(alias: str, root: Path | None = None) -> Path:
    """Resolve one registered alias to its immutable checked-in source path."""

    base = root.resolve() if root else repository_root()
    registry = load_version_registry(base)
    try:
        record = registry["aliases"][alias]
        relative = record["source_path"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"unknown or pathless scientific alias: {alias}") from exc
    path = base / str(relative)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
