#!/usr/bin/env python3
"""Verify or reproduce the frozen ICDS publication artifact.

The default mode is read-only. Reproduction modes write only to temporary
directories and compare their results with the checked-in frozen evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "IoT_auth_simulator"
PHASE4_ROOT = ROOT / "artifacts/icds-ml-phase4"

EXPECTED_CANDIDATE_FILE_SHA256 = (
    "fa22706b090bedf9b944deee8cc02f7df1bb186661c1cb428d6af82705f08bce"
)
EXPECTED_CANDIDATE_MANIFEST_FILE_SHA256 = (
    "6524346f74331c05aa2aceb157f9e0b17b055b823f4b99475d514b0352e043aa"
)
EXPECTED_PHASE4_TREE = (
    "sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11"
)
EXPECTED_ROLE_COUNTS = {
    "ordinary_normal": 96,
    "hard_negative:small_positive_causal_margin": 48,
    "anomaly:timestamp_inconsistency": 48,
    "hard_negative:legitimate_same_attempt_nonce_recurrence": 48,
    "hard_negative:fresh_nonce_cross_attempt": 48,
    "anomaly:nonce_reuse": 48,
}
EXPECTED_PHASE4_HASHES = {
    "primary/predictions.jsonl": "04fc533592a50e23dab67ba22d3bd7ebf23239702de4a9daa149d586fbdc5539",
    "primary/run-provenance.json": "9a27f9ad469502b4dcd7d0408eff2788a877d51b96dd96751e434fe1343b2594",
    "primary/per-run-metrics.json": "50ee928098644035869dfb48306255b2f749c614111338f8b43445cbb79404aa",
    "primary/aggregate-summary.json": "d33f909862039afd727250b67705c6359d974cde75849455c967efb8b70206ca",
    "grouped/predictions.jsonl": "d76cb118f7d37082d2ee26e9cc90e79bd864875e65fdaca6b8820f04f931fb25",
    "grouped/run-provenance.json": "bd560a4b8eac22e7778f11787449d0c57c32f6af3cf665e93525bf70d7089c13",
    "grouped/per-run-metrics.json": "7568feb63040978278cd202d63e8853197698054060cd25c011e4f780a25a687",
    "grouped/grouped-sensitivity-table.json": "ef69601c1b57492b6fbb1471f1e113a7ddfd1fe2610699707ee091d8a7e4aeb0",
    "claim-audit.json": "b22a62e7eb014c7c492d4031be7d1ec97d5d16b65d55bb96026fa33b7ef33293",
}
SCIENTIFIC_PHASE4_FILES = (
    "primary/predictions.jsonl",
    "primary/per-run-metrics.json",
    "primary/aggregate-summary.json",
    "primary/figure-data-c0-c1-normal-fpr.json",
    "primary/figure-data-c0-c1-normal-fpr.csv",
    "primary/table-a-experimental-outcomes.json",
    "primary/table-a-experimental-outcomes.csv",
    "primary/table-b-normal-subgroup-fpr.json",
    "primary/table-b-normal-subgroup-fpr.csv",
    "grouped/predictions.jsonl",
    "grouped/per-run-metrics.json",
    "grouped/grouped-sensitivity-table.json",
    "grouped/grouped-sensitivity-table.csv",
)


class PublicationVerificationError(RuntimeError):
    """Frozen publication evidence failed verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PublicationVerificationError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def verify_frozen_artifacts() -> dict[str, Any]:
    candidate_path = ROOT / "artifacts/icds-timing-nonce-candidate/scientific-candidate.json"
    manifest_path = ROOT / "artifacts/icds-timing-nonce-candidate/build-manifest.json"
    assert_equal(sha256_file(candidate_path), EXPECTED_CANDIDATE_FILE_SHA256,
                 "candidate file SHA-256")
    assert_equal(sha256_file(manifest_path), EXPECTED_CANDIDATE_MANIFEST_FILE_SHA256,
                 "candidate manifest file SHA-256")

    candidate = read_json(candidate_path)
    manifest = read_json(manifest_path)
    role_counts = Counter(
        row["privileged_metadata"]["role"]["key"] for row in candidate["samples"]
    )
    assert_equal(dict(role_counts), EXPECTED_ROLE_COUNTS, "candidate role counts")
    assert_equal(manifest["counts"]["total"], 336, "candidate sample count")
    assert_equal(manifest["counts"]["truth"], {"normal": 240, "anomaly": 96},
                 "candidate truth counts")
    assert_equal(manifest["counts"]["qualified_hard_negatives"], 144,
                 "qualified hard-negative count")
    assert_equal(manifest["base_pool"]["attempted"], 512, "base attempts")
    assert_equal(manifest["base_pool"]["admitted"], 510, "base admissions")
    assert_equal(manifest["base_pool"]["rejected"], 2, "base rejections")
    assert_equal(manifest["validation"]["all_aggregate_gates_passed"], True,
                 "candidate validation gate")

    phase4_fingerprints = read_json(PHASE4_ROOT / "fingerprints.json")
    assert_equal(phase4_fingerprints["artifact_tree_fingerprint"], EXPECTED_PHASE4_TREE,
                 "frozen Phase-4 tree fingerprint")
    for relative, expected in EXPECTED_PHASE4_HASHES.items():
        assert_equal(sha256_file(PHASE4_ROOT / relative), expected,
                     f"Phase-4 file hash {relative}")
    assert_equal(line_count(PHASE4_ROOT / "primary/predictions.jsonl"), 3780,
                 "primary prediction count")
    assert_equal(line_count(PHASE4_ROOT / "grouped/predictions.jsonl"), 819,
                 "grouped prediction count")
    assert_equal(len(read_json(PHASE4_ROOT / "primary/per-run-metrics.json")), 45,
                 "primary run count")
    assert_equal(len(read_json(PHASE4_ROOT / "grouped/per-run-metrics.json")), 9,
                 "grouped run count")

    paper_manifest_path = ROOT / "artifacts/icds-paper-figures/paper-artifact-manifest.json"
    paper_manifest = read_json(paper_manifest_path)
    assert_equal(paper_manifest["source_phase4_tree_fingerprint"], EXPECTED_PHASE4_TREE,
                 "paper source tree fingerprint")
    for record in paper_manifest["artifacts"]:
        path = paper_manifest_path.parent / record["filename"]
        assert_equal(sha256_file(path), record["sha256"],
                     f"paper artifact hash {record['filename']}")

    # These routines enforce frozen upstream feature, manifest, pipeline, and
    # grouped-cohort fingerprints without creating outputs.
    from ml.icds.phase4_experiment import verify_phase4_inputs
    verify_phase4_inputs(ROOT)

    return {
        "candidate_samples": 336,
        "semantic_normals": 240,
        "anomalies": 96,
        "primary_runs": 45,
        "primary_predictions": 3780,
        "grouped_runs": 9,
        "grouped_predictions": 819,
        "phase4_tree_fingerprint": EXPECTED_PHASE4_TREE,
        "paper_artifacts": len(paper_manifest["artifacts"]),
    }


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + current if current else "")
    return environment


def reproduce_candidate() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="icds-candidate-reproduction-") as directory:
        root = Path(directory)
        subprocess.run(
            [sys.executable, "-m", "simulator.composition.timing_nonce_candidate"],
            cwd=root, env=_environment(), check=True,
        )
        generated = root / "artifacts/icds-timing-nonce-candidate"
        expected = ROOT / "artifacts/icds-timing-nonce-candidate"
        compared = []
        for name in ("scientific-candidate.json", "build-manifest.json",
                     "candidate-characterization.md"):
            assert_equal((generated / name).read_bytes(), (expected / name).read_bytes(),
                         f"reproduced candidate artifact {name}")
            compared.append(name)
    return {"byte_identical": True, "files": compared}


def _without_execution_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_execution_time(item)
            for key, item in value.items()
            if key not in {"created_at_utc", "started_at_utc", "completed_at_utc"}
        }
    if isinstance(value, list):
        return [_without_execution_time(item) for item in value]
    return value


def reproduce_phase4() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="icds-phase4-reproduction-") as directory:
        output = Path(directory) / "phase4"
        subprocess.run(
            [sys.executable, "-m", "ml.icds.phase4_experiment",
             "--repository-root", str(ROOT), "--output", str(output),
             "--authorize-scientific-execution"],
            cwd=ROOT, env=_environment(), check=True,
        )
        for relative in SCIENTIFIC_PHASE4_FILES:
            assert_equal((output / relative).read_bytes(),
                         (PHASE4_ROOT / relative).read_bytes(),
                         f"reproduced scientific Phase-4 artifact {relative}")

        generated_provenance = list(output.glob("primary/**/run-provenance.json"))
        generated_provenance += list(output.glob("grouped/**/run-provenance.json"))
        for generated in generated_provenance:
            relative = generated.relative_to(output)
            frozen = PHASE4_ROOT / relative
            assert_equal(
                _without_execution_time(read_json(generated)),
                _without_execution_time(read_json(frozen)),
                f"normalized Phase-4 provenance {relative}",
            )

        generated_execution = _without_execution_time(read_json(output / "execution-manifest.json"))
        frozen_execution = _without_execution_time(read_json(PHASE4_ROOT / "execution-manifest.json"))
        assert_equal(generated_execution, frozen_execution,
                     "normalized Phase-4 execution manifest")
        rerun_tree = read_json(output / "fingerprints.json")["artifact_tree_fingerprint"]
    return {
        "scientific_files_byte_identical": len(SCIENTIFIC_PHASE4_FILES),
        "normalized_provenance_files_identical": len(generated_provenance),
        "frozen_tree_fingerprint": EXPECTED_PHASE4_TREE,
        "rerun_tree_fingerprint": rerun_tree,
        "tree_difference_reason": "execution timestamps only; claim audit is a frozen post-run artifact",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="verify checked-in frozen artifacts (the default)")
    parser.add_argument("--reproduce-candidate", action="store_true",
                        help="regenerate the candidate in a temporary directory")
    parser.add_argument("--reproduce-phase4", action="store_true",
                        help="rerun the frozen Phase-4 experiment in a temporary directory")
    args = parser.parse_args()

    result: dict[str, Any] = {"verification": verify_frozen_artifacts()}
    if args.reproduce_candidate:
        result["candidate_reproduction"] = reproduce_candidate()
    if args.reproduce_phase4:
        result["phase4_reproduction"] = reproduce_phase4()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
