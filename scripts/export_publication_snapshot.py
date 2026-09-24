#!/usr/bin/env python3
"""Create a minimal allowlisted public snapshot without altering the source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "PUBLICATION_FILES.txt"
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def selected_files() -> list[Path]:
    patterns = [
        line.strip() for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected: set[Path] = set()
    missing: list[str] = []
    for pattern in patterns:
        matches = [path for path in ROOT.glob(pattern) if path.is_file()]
        if not matches:
            missing.append(pattern)
        selected.update(matches)
    if missing:
        raise RuntimeError(f"publication allowlist patterns matched nothing: {missing}")
    result = []
    for path in selected:
        relative = path.relative_to(ROOT)
        if FORBIDDEN_PARTS & set(relative.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            continue
        result.append(relative)
    return sorted(result)


def export(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite publication destination: {destination}")
    if ROOT == destination or ROOT in destination.parents:
        raise RuntimeError("publication destination must be outside the research repository")
    destination.mkdir(parents=True)
    records = []
    for relative in selected_files():
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "path": relative.as_posix(),
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    manifest = {
        "schema_version": "icds-2026-publication-snapshot-v1",
        "source_scientific_commit": "2a9883bdb441a83d3887f496f756ad3b9c4a1d1b",
        "source_repository_head": source_commit(),
        "selection_policy": "PUBLICATION_FILES.txt allowlist",
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    (destination / "PUBLICATION_SNAPSHOT.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

