"""Independent final validation and freeze metadata for the R4.2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.evaluation.metrics import reconstruct_metrics


EXPECTED_HEAD_PARENT = "bd00f098ec86902e5c01e5da1cf1eb4c6e92d77b"
R42_FINGERPRINT = "sha256:c1763f75d1085ccc9c905c5950b201b4dbf7a898b2790a99ffa51181aeb67832"
PREEXECUTION_FINGERPRINT = "sha256:29b2792d1555c51f1d0d5a9aac5ae549ac23c1d6e817023b8acc78e31aab5da1"
R41_AGGREGATE = "sha256:cce473976d8ea079d5ad77031da21a2062a02aaae6121ce451ff0db49eacea34"
R41_TREE = "sha256:ed387b03bcacf12369dbaeb0016ba027db301ffdb7f01ecba7609507906ec284"
PHASE4_TREE = "sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11"
DATASET_SHA256 = "fa22706b090bedf9b944deee8cc02f7df1bb186661c1cb428d6af82705f08bce"
PARTITION_SHA256 = "80c8621860237999b95c2d4e7a0f12a7b56f42ef9c3df3720af5a393bc0131fe"
ACCEPTED_SHA256 = "4b7fd17277a6765bba0e509ee48a39f3c7f59492a4e69d3d238ba2cb91671435"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_inventory(base: Path, records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        path = base / record["path"]
        if not path.is_file():
            raise AssertionError(f"missing artifact: {path}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssertionError(f"artifact changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence_out = (root / args.evidence_out).resolve()
    manifest_out = (root / args.manifest_out).resolve()
    artifact_root = root / "artifacts/r4.2-representation-sensitivity"
    pre_root = root / "metadata/r4.2-preexecution"
    r41_root = root / "artifacts/r4.1-device-disjoint-evaluation"

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(condition), "evidence": evidence})
        if not condition:
            raise AssertionError(f"{name}: {evidence}")

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    check("branch", branch == "research/r4.2-representation-sensitivity", branch)
    check("execution_code_commit", head == EXPECTED_HEAD_PARENT, head)

    fingerprints = read_json(artifact_root / "fingerprints.json")
    check("r4_2_aggregate_fingerprint", fingerprints["artifact_tree_fingerprint"] == R42_FINGERPRINT, fingerprints["artifact_tree_fingerprint"])
    check("r4_2_self_excluding_inventory_count", fingerprints["artifact_count"] == 378, fingerprints["artifact_count"])
    verify_inventory(artifact_root, fingerprints["files"])
    core = {
        "schema_version": fingerprints["schema_version"],
        "artifact_count": fingerprints["artifact_count"],
        "files": fingerprints["files"],
    }
    check("r4_2_fingerprint_reconstruction", canonical_hash(core) == R42_FINGERPRINT, canonical_hash(core))
    all_artifact_files = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    check("r4_2_total_file_count", len(all_artifact_files) == 379, len(all_artifact_files))

    pre_inventory = read_json(pre_root / "durable-artifact-inventory.json")
    pre_core = {key: value for key, value in pre_inventory.items() if key != "inventory_fingerprint"}
    check("preexecution_fingerprint", pre_inventory["inventory_fingerprint"] == PREEXECUTION_FINGERPRINT, pre_inventory["inventory_fingerprint"])
    check("preexecution_fingerprint_reconstruction", canonical_hash(pre_core) == PREEXECUTION_FINGERPRINT, canonical_hash(pre_core))
    verify_inventory(root, pre_inventory["artifacts"])

    contracts = read_json(artifact_root / "feature-contracts.json")
    plans = read_json(artifact_root / "run-matrix.json")
    provenance = read_json(artifact_root / "run-provenance.json")
    metric_rows = read_json(artifact_root / "run-level-metrics.json")
    predictions = read_jsonl(artifact_root / "predictions.jsonl")
    check("new_run_count", len(plans) == len(provenance) == len(metric_rows) == 120, [len(plans), len(provenance), len(metric_rows)])
    check("new_prediction_count", len(predictions) == 10080, len(predictions))

    plan_by_id = {row["planned_run_id"]: row for row in plans}
    provenance_by_id = {row["run_id"]: row for row in provenance}
    metrics_by_id = {row["run_id"]: row["metrics"] for row in metric_rows}
    predictions_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        predictions_by_id[record["run_id"]].append(record)
    expected_ids = set(plan_by_id)
    check("unique_complete_run_id_set", len(expected_ids) == 120 and expected_ids == set(provenance_by_id) == set(metrics_by_id) == set(predictions_by_id), len(expected_ids))
    check("predictions_per_run", all(len(rows) == 84 and len({row["trace_id"] for row in rows}) == 84 for rows in predictions_by_id.values()), "84 unique per run")

    r41_provenance = read_json(r41_root / "run-provenance.json")
    c1_by_key = {(row["partition_id"], row["model"]): row for row in r41_provenance if row["condition"] == "C1"}
    feature_names = {
        condition: [entry["feature_name"] for entry in contracts["conditions"][condition]["ordered_features"]]
        for condition in ("C", "D")
    }
    check("C_feature_count", len(feature_names["C"]) == 89, len(feature_names["C"]))
    check("D_feature_count", len(feature_names["D"]) == 82, len(feature_names["D"]))
    check("C_feature_fingerprint", contracts["conditions"]["C"]["ordered_list_fingerprint"] == "sha256:3f74f94a69f1f25fecf76172821e67bc3c89111279dd1e089363ecfbf37d1af6", contracts["conditions"]["C"]["ordered_list_fingerprint"])
    check("D_feature_fingerprint", contracts["conditions"]["D"]["ordered_list_fingerprint"] == "sha256:37ee3873ad93b9feeea2221edabf4c06b7eec8614b54331956407d7e2f98b3a3", contracts["conditions"]["D"]["ordered_list_fingerprint"])
    check("relation_count_retained", "semantic_timing_relation_count" in feature_names["C"] and "semantic_timing_relation_count" in feature_names["D"], True)

    for run_id, plan in plan_by_id.items():
        rows = predictions_by_id[run_id]
        run_provenance = provenance_by_id[run_id]
        anchor = c1_by_key[(plan["partition_id"], plan["model"])]
        check(f"training_membership:{run_id}", plan["training_trace_ids"] == anchor["training_trace_ids"] == run_provenance["training_trace_ids"], "exact C1 cohort")
        check(f"test_membership:{run_id}", plan["test_trace_ids"] == anchor["test_trace_ids"] == run_provenance["test_trace_ids"] and {row["trace_id"] for row in rows} == set(plan["test_trace_ids"]), "exact frozen test")
        check(f"device_disjoint:{run_id}", not set(run_provenance["training_device_keys"]) & set(run_provenance["test_device_keys"]), "zero overlap")
        check(f"feature_contract:{run_id}", run_provenance["ordered_feature_names"] == feature_names[plan["condition"]] and run_provenance["feature_contract_fingerprint"] == contracts["conditions"][plan["condition"]]["ordered_list_fingerprint"], plan["condition"])
        check(f"metric_reconstruction:{run_id}", all(metrics_by_id[run_id][key] == value for key, value in reconstruct_metrics(rows).items()), "exact")

    r41_fingerprints = read_json(r41_root / "fingerprints.json")
    check("r4_1_artifact_tree", r41_fingerprints["artifact_tree_fingerprint"] == R41_TREE, r41_fingerprints["artifact_tree_fingerprint"])
    verify_inventory(r41_root, r41_fingerprints["files"])
    freeze = read_json(root / "metadata/r4.1-freeze/freeze-manifest.json")
    check("r4_1_aggregate", freeze["aggregate_r4_1_fingerprint"] == R41_AGGREGATE, freeze["aggregate_r4_1_fingerprint"])
    verify_inventory(root, freeze["authoritative_inventory"])

    dataset_manifest = read_json(root / "datasets/dataset-v1.0/manifest.json")
    dataset_path = root / dataset_manifest["candidate"]["path"]
    check("dataset_hash", sha256(dataset_path) == DATASET_SHA256, sha256(dataset_path))
    phase4 = read_json(root / "artifacts/icds-ml-phase4/fingerprints.json")
    check("historical_phase4", phase4["artifact_tree_fingerprint"] == PHASE4_TREE, phase4["artifact_tree_fingerprint"])
    check("partition_manifest_hash", sha256(root / "metadata/r4.1-freeze/inputs/partition-manifest.json") == PARTITION_SHA256, PARTITION_SHA256)
    check("accepted_partitions_hash", sha256(root / "metadata/r4.1-freeze/inputs/accepted-partitions.csv") == ACCEPTED_SHA256, ACCEPTED_SHA256)

    paired = read_json(artifact_root / "paired-comparisons.json")
    check("paired_comparison_count", len(paired) == 240, len(paired))
    check("paired_contrast_counts", Counter(row["contrast"] for row in paired) == {"A_to_C": 60, "C_to_D": 60, "A_to_B": 60, "C_to_B": 60}, dict(Counter(row["contrast"] for row in paired)))
    independent = read_json(artifact_root / "independent-validation-report.json")
    check("execution_independent_validation", independent["status"] == "PASS", independent["status"])

    validation = {
        "schema_version": "r4.2-final-freeze-validation-v1",
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator_path": "metadata/r4.2-freeze/validate_r4_2_freeze.py",
        "validator_sha256": sha256(Path(__file__)),
        "repository_branch": branch,
        "execution_code_commit": head,
        "check_count": len(checks),
        "new_run_count": 120,
        "new_prediction_count": 10080,
        "r4_2_total_artifact_files": 379,
        "r4_2_inventory_files_excluding_fingerprint_manifest": 378,
        "r4_2_aggregate_fingerprint": R42_FINGERPRINT,
        "r4_1_files_verified": len(r41_fingerprints["files"]),
        "r4_1_freeze_files_verified": len(freeze["authoritative_inventory"]),
        "checks": checks,
    }
    write_json(evidence_out, validation)

    complete_inventory = list(fingerprints["files"])
    fingerprint_path = artifact_root / "fingerprints.json"
    complete_inventory.append({
        "path": "fingerprints.json",
        "bytes": fingerprint_path.stat().st_size,
        "sha256": sha256(fingerprint_path),
        "inventory_role": "self-excluded fingerprint manifest",
    })
    complete_inventory.sort(key=lambda record: record["path"])
    execution_manifest = read_json(artifact_root / "execution-manifest.json")
    pre_verification = read_json(artifact_root / "pre-execution-verification.json")
    freeze_manifest = {
        "schema_version": "r4.2-scientific-freeze-manifest-v1",
        "status": "FROZEN",
        "archive_scope": "R4.2 representation-sensitivity scientific execution",
        "repository_branch": branch,
        "execution_code_commit": EXPECTED_HEAD_PARENT,
        "final_archival_commit": "excluded to avoid a self-referential commit identifier; record externally after commit",
        "artifact_namespace": "artifacts/r4.2-representation-sensitivity",
        "artifact_inventory_count": len(complete_inventory),
        "scientific_artifact_count_excluding_fingerprint_manifest": fingerprints["artifact_count"],
        "artifact_inventory": complete_inventory,
        "final_aggregate_r4_2_fingerprint": R42_FINGERPRINT,
        "preexecution_fingerprint": PREEXECUTION_FINGERPRINT,
        "upstream": {
            "r4_1_execution_commit": "97003b1d9448caa105b6e206ceb5deb18c02a207",
            "r4_1_archival_checkpoint": "7ad3284f4c45f793c3f7ee10a678c3f73109bf53",
            "r4_1_aggregate_fingerprint": R41_AGGREGATE,
            "r4_1_artifact_tree_fingerprint": R41_TREE,
            "dataset_file_sha256": DATASET_SHA256,
            "historical_phase4_fingerprint": PHASE4_TREE,
            "partition_manifest_sha256": PARTITION_SHA256,
            "accepted_partitions_sha256": ACCEPTED_SHA256,
        },
        "feature_contract_fingerprints": {
            condition: {
                "feature_count": contracts["conditions"][condition]["feature_count"],
                "ordered_list_fingerprint": contracts["conditions"][condition]["ordered_list_fingerprint"],
                "contract_fingerprint": contracts["conditions"][condition]["contract_fingerprint"],
            }
            for condition in ("A", "B", "C", "D")
        },
        "execution": {
            "new_conditions": ["C", "D"],
            "immutable_anchor_conditions": ["A", "B"],
            "new_run_count": 120,
            "new_prediction_count": 10080,
            "combined_evaluation_count": 240,
            "paired_comparison_count": 240,
            "execution_implementation_path": "artifacts/r4.2-representation-sensitivity/execution-implementation/r4_2_execution.executed.py",
            "execution_implementation_sha256": execution_manifest["execution_implementation_sha256"],
            "software_versions": pre_verification["software_versions"],
        },
        "validation": {
            "status": "PASS",
            "evidence_path": "metadata/r4.2-freeze/validation-evidence.json",
            "evidence_sha256": sha256(evidence_out),
            "execution_independent_validation_path": "artifacts/r4.2-representation-sensitivity/independent-validation-report.json",
            "execution_independent_validation_sha256": sha256(artifact_root / "independent-validation-report.json"),
        },
        "tag_policy": "No tag was created or moved by the freeze task.",
    }
    write_json(manifest_out, freeze_manifest)
    print(json.dumps({
        "status": validation["status"],
        "checks": len(checks),
        "artifact_inventory_count": len(complete_inventory),
        "r4_2_fingerprint": R42_FINGERPRINT,
        "r4_1_fingerprint": R41_AGGREGATE,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
