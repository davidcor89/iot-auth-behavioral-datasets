#!/usr/bin/env python3
"""Export the current allowlisted repository artifact to a new directory."""

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


def selected_files() -> list[Path]:
    patterns = [
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected: set[Path] = set()
    for pattern in patterns:
        matches = [path for path in ROOT.glob(pattern) if path.is_file()]
        if not matches:
            raise RuntimeError(f"allowlist pattern matched nothing: {pattern}")
        selected.update(matches)
    return sorted(
        path.relative_to(ROOT)
        for path in selected
        if not (FORBIDDEN_PARTS & set(path.relative_to(ROOT).parts))
        and path.suffix not in FORBIDDEN_SUFFIXES
    )


def export(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite destination: {destination}")
    if ROOT == destination or ROOT in destination.parents:
        raise RuntimeError("destination must be outside the repository")
    destination.mkdir(parents=True)
    records = []
    for relative in selected_files():
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(target),
                "size": target.stat().st_size,
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "file_count": len(records),
        "files": records,
        "schema_version": "repository-artifact-snapshot-v1",
        "selection_policy": "PUBLICATION_FILES.txt allowlist",
        "source_repository_commit": commit,
        "total_bytes": sum(record["size"] for record in records),
    }
    (destination / "REPOSITORY_SNAPSHOT.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
