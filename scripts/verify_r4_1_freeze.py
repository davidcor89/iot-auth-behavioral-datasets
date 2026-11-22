#!/usr/bin/env python3
"""Verify the R4.1 freeze without executing any ML model."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata/r4.1-freeze/freeze-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    inventory = manifest["authoritative_inventory"]
    if inventory != sorted(inventory, key=lambda row: row["path"]):
        raise SystemExit("R4.1 inventory is not canonically path-sorted")
    for record in inventory:
        path = ROOT / record["path"]
        if not path.is_file():
            raise SystemExit(f"missing R4.1 artifact: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise SystemExit(f"size mismatch: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {record['path']}")
    aggregate = canonical_hash(inventory)
    if aggregate != manifest["aggregate_r4_1_fingerprint"]:
        raise SystemExit("aggregate R4.1 fingerprint mismatch")

    output_root = ROOT / manifest["durable_evidence_locators"]["repository_output_root"]
    fingerprints = json.loads((output_root / "fingerprints.json").read_text())
    output_records = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p.name != "fingerprints.json"):
        output_records.append({
            "path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if output_records != fingerprints["files"]:
        raise SystemExit("R4.1 execution-output inventory mismatch")
    if canonical_hash(output_records) != fingerprints["artifact_tree_fingerprint"]:
        raise SystemExit("R4.1 execution-output fingerprint mismatch")

    tags = dict(
        line.split(" ", 1)
        for line in subprocess.run(
            ["git", "tag", "--list", "--format=%(refname:short) %(objectname)"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        ).stdout.splitlines()
        if line
    )
    expected_tags = manifest["git_provenance"]["historical_tags"]
    if any(tags.get(name) != target for name, target in expected_tags.items()):
        raise SystemExit("historical tag target changed")

    print(json.dumps({
        "status": "PASS",
        "inventory_count": len(inventory),
        "aggregate_r4_1_fingerprint": aggregate,
        "execution_output_artifact_count": len(output_records),
        "execution_output_fingerprint": fingerprints["artifact_tree_fingerprint"],
        "historical_tags": expected_tags,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
