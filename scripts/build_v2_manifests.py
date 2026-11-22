#!/usr/bin/env python3
"""Build deterministic V2-01 inventory and historical snapshot manifests."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_TAG = "icds-2026-artifact-v1.0"
VENUE_PATTERN = re.compile(r"icds|codaspy|ifip[\s_-]*sec", re.IGNORECASE)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tracked_files() -> list[Path]:
    return [
        Path(item)
        for item in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        if item
    ]


def classify_venue_path(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("artifacts/") or value in {
        "PUBLICATION_FILES.txt", "PUBLICATION_SNAPSHOT.json"
    }:
        return "frozen scientific evidence"
    if value.startswith("submissions/"):
        return "historical submission content"
    if value.startswith("IoT_auth_simulator/ml/icds/"):
        return "executable code or import"
    if value.startswith("tests/"):
        return "reproducibility test"
    if value.startswith("notebooks/"):
        return "historical publication notebook"
    if value.startswith("scripts/"):
        return "historical publication script"
    if value.startswith("docs/") or value in {"README.md", "CITATION.cff"}:
        return "documentation"
    return "incidental reference"


def build_snapshot() -> dict[str, object]:
    public_snapshot = read_json(ROOT / "PUBLICATION_SNAPSHOT.json")
    files = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "size": record["size"],
            "source_path": record["path"],
            "source_ref": f"{SUBMISSION_TAG}^{{}}",
        }
        for record in public_snapshot["files"]
    ]
    return {
        "included_files": files,
        "missing_artifacts": [
            {
                "artifact": "submitted manuscript source",
                "status": "not present in repository or immutable release",
            },
            {
                "artifact": "exact submitted manuscript PDF",
                "status": "not present in repository or immutable release",
            },
            {
                "artifact": "reviewer reports",
                "status": "not present in repository or immutable release",
            },
            {
                "artifact": "external submission-system metadata",
                "status": "not present in repository or immutable release",
            },
        ],
        "provenance": {
            "publication_snapshot_manifest": "PUBLICATION_SNAPSHOT.json",
            "publication_snapshot_manifest_sha256": sha256_file(
                ROOT / "PUBLICATION_SNAPSHOT.json"
            ),
            "scientific_dataset": "dataset-v1.0",
            "scientific_evaluation": "eval-v1.0",
            "source_commit": git("rev-parse", f"{SUBMISSION_TAG}^{{}}"),
            "source_tag": SUBMISSION_TAG,
            "tag_object": git("rev-parse", SUBMISSION_TAG),
        },
        "schema_version": "historical-submission-snapshot-v1",
        "snapshot_identifier": "submission-icds2026-v1.0",
        "statement": (
            "Historical submission artifact snapshot. Every included file is "
            "bound to the immutable annotated tag and recorded SHA-256; this "
            "manifest does not track future changes to main."
        ),
    }


def build_scientific_inventory() -> dict[str, object]:
    prefixes = {
        "dataset": "artifacts/icds-timing-nonce-candidate/",
        "feature_extraction": "artifacts/icds-ml-phase1/",
        "experiment_definition_and_splits": "artifacts/icds-ml-phase2/",
        "grouped_sensitivity_split": "artifacts/icds-ml-phase3/",
        "predictions_metrics_and_provenance": "artifacts/icds-ml-phase4/",
        "figures_tables_and_source_data": "artifacts/icds-paper-figures/",
    }
    records = []
    for path in sorted(ROOT.joinpath("artifacts").rglob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT).as_posix()
            collection = next(
                name for name, prefix in prefixes.items() if relative.startswith(prefix)
            )
            records.append(
                {
                    "collection": collection,
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return {
        "baseline_commit": "952718ab4faa191c870a045dfa8b7b1c075c0b36",
        "file_count": len(records),
        "files": records,
        "inventory_version": "v2-01-scientific-artifact-inventory-v1",
        "policy": "Hashes captured from the immutable submitted artifact before V2-01 normalization.",
        "source_tag": SUBMISSION_TAG,
    }


def build_venue_inventory() -> dict[str, object]:
    records = []
    for relative in tracked_files():
        if relative.as_posix() in {
            "metadata/scientific-artifact-inventory.json",
            "metadata/venue-reference-inventory.json",
            "submissions/icds2026/snapshot-manifest.json",
        }:
            continue
        path = ROOT / relative
        path_hits = len(VENUE_PATTERN.findall(relative.as_posix()))
        content_hits = 0
        if path.is_file():
            try:
                content_hits = len(
                    VENUE_PATTERN.findall(path.read_text(encoding="utf-8"))
                )
            except (UnicodeDecodeError, OSError):
                pass
        if path_hits or content_hits:
            records.append(
                {
                    "classification": classify_venue_path(relative),
                    "content_occurrences": content_hits,
                    "path": relative.as_posix(),
                    "path_occurrences": path_hits,
                }
            )
    return {
        "inventory_version": "v2-01-venue-reference-inventory-v1",
        "records": records,
        "search_terms": ["ICDS", "CODASPY", "IFIPSEC", "IFIP-SEC", "IFIP_SEC"],
        "total_content_occurrences": sum(r["content_occurrences"] for r in records),
        "total_path_occurrences": sum(r["path_occurrences"] for r in records),
    }


def main() -> None:
    write_json(
        ROOT / "submissions/icds2026/snapshot-manifest.json", build_snapshot()
    )
    write_json(
        ROOT / "metadata/scientific-artifact-inventory.json",
        build_scientific_inventory(),
    )
    write_json(
        ROOT / "metadata/venue-reference-inventory.json", build_venue_inventory()
    )


if __name__ == "__main__":
    main()
