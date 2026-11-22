#!/usr/bin/env python3
"""Verify venue-neutral aliases and the unchanged frozen scientific evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "IoT_auth_simulator"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ml.artifact_paths import resolve_scientific_path


class IntegrityError(RuntimeError):
    """Repository alias or immutable evidence verification failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise IntegrityError(f"{label}: expected {expected!r}, found {actual!r}")


def load_legacy_verifier() -> Any:
    path = ROOT / "scripts/verify_publication.py"
    spec = importlib.util.spec_from_file_location("frozen_submission_verifier", path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot load frozen verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inventory() -> dict[str, int]:
    inventory = read_json(ROOT / "metadata/scientific-artifact-inventory.json")
    collection_counts: Counter[str] = Counter()
    for record in inventory["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise IntegrityError(f"missing frozen scientific file: {record['path']}")
        assert_equal(sha256_file(path), record["sha256"], record["path"])
        collection_counts[record["collection"]] += 1
    assert_equal(len(inventory["files"]), inventory["file_count"], "inventory file count")
    return dict(sorted(collection_counts.items()))


def verify_reference_manifests() -> int:
    manifests = [
        ROOT / "datasets/dataset-v1.0/manifest.json",
        ROOT / "validation/dataset-v1.0/manifest.json",
        ROOT / "experiments/eval-v1.0/manifest.json",
        ROOT / "publication-artifacts/eval-v1.0/manifest.json",
    ]
    manifests += sorted((ROOT / "experiments/eval-v1.0").glob("*/manifest.json"))
    manifests += sorted((ROOT / "publication-artifacts/eval-v1.0").glob("*/manifest.json"))
    checked = 0

    def visit(value: Any) -> None:
        nonlocal checked
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = ROOT / value["path"]
                if not path.is_file():
                    raise IntegrityError(f"unresolved reference: {value['path']}")
                assert_equal(sha256_file(path), value["sha256"], value["path"])
                checked += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for manifest in manifests:
        visit(read_json(manifest))
    return checked


def verify_submission_snapshot() -> dict[str, Any]:
    snapshot = read_json(ROOT / "submissions/icds2026/snapshot-manifest.json")
    provenance = snapshot["provenance"]
    tag = provenance["source_tag"]
    import subprocess

    target = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{}}"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert_equal(target, provenance["source_commit"], "historical tag target")
    for record in snapshot["included_files"]:
        content = subprocess.run(
            ["git", "show", f"{tag}^{{}}:{record['source_path']}"], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout
        assert_equal(sha256_bytes(content), record["sha256"], record["source_path"])
        assert_equal(len(content), record["size"], f"size {record['source_path']}")
    return {
        "files": len(snapshot["included_files"]),
        "missing_artifacts_declared": len(snapshot["missing_artifacts"]),
        "source_commit": target,
    }


def verify_dataset_identity() -> dict[str, Any]:
    candidate = read_json(resolve_scientific_path("dataset-v1.0", ROOT))
    experiment = read_json(resolve_scientific_path("eval-v1.0-manifest", ROOT))
    sample_ids = [row["sample_id"] for row in candidate["samples"]]
    assert_equal(len(sample_ids), 336, "candidate sample count")
    assert_equal(len(set(sample_ids)), 336, "candidate unique sample IDs")
    assert_equal(
        sorted(sample_ids), sorted(experiment["candidate_trace_ids"]),
        "candidate/experiment sample memberships",
    )
    roles = Counter(
        row["privileged_metadata"]["role"]["key"] for row in candidate["samples"]
    )
    return {"sample_count": len(sample_ids), "roles": dict(sorted(roles.items()))}


def verify_only() -> dict[str, Any]:
    legacy = load_legacy_verifier()
    return {
        "frozen_evidence": legacy.verify_frozen_artifacts(),
        "scientific_inventory": verify_inventory(),
        "reference_hashes_checked": verify_reference_manifests(),
        "historical_snapshot": verify_submission_snapshot(),
        "dataset_identity": verify_dataset_identity(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reproduce-candidate", action="store_true")
    parser.add_argument("--reproduce-evaluation", action="store_true")
    args = parser.parse_args()
    legacy = load_legacy_verifier()
    result: dict[str, Any] = {"verification": verify_only()}
    if args.reproduce_candidate:
        result["candidate_reproduction"] = legacy.reproduce_candidate()
    if args.reproduce_evaluation:
        result["evaluation_reproduction"] = legacy.reproduce_phase4()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
