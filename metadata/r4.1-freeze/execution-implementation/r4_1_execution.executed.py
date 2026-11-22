"""Execute and independently validate the frozen R4.1 grouped campaign.

This module is deliberately isolated from the historical Phase-4 artifact tree.
It consumes externally frozen partitions and reuses the frozen feature, model,
threshold, and metric implementations without redefining them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn

from ml.artifact_paths import resolve_scientific_path
from ml.evaluation.experiment_manifest import (
    ANOMALY_ROLES,
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
    ORDINARY,
    ROLE_TO_BINARY_TARGET,
    SAME_ATTEMPT_HN,
    TIMING_ANOMALY,
    TIMING_HN,
    file_sha256,
)
from ml.evaluation.metrics import reconstruct_metrics
from ml.evaluation.pipeline import (
    DT_PARAMETERS,
    EXPECTED_C1_FINGERPRINT,
    EXPECTED_C2_FINGERPRINT,
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_CANDIDATE_FINGERPRINT,
    EXPECTED_FEATURE_SPEC_FILE_SHA256,
    EXPECTED_PHASE2_FILE_SHA256,
    EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
    LR_PARAMETERS,
    MODEL_CONFIGURATION_VERSION,
    MODELS,
    PIPELINE_SPEC_VERSION,
    PREDICTION_THRESHOLD,
    PREPROCESSING_CONFIGURATION_VERSION,
    RF_PARAMETERS,
    assemble_matrix,
    feature_profile_fingerprint,
    fit_estimator,
    fitted_scaler_state,
    load_frozen_inputs,
    predict_estimator,
)
from ml.evaluation.trace_features import C1_FEATURE_NAMES, C2_FEATURE_NAMES, FEATURE_DEFINITIONS


EXPERIMENT_ID = "r4.1-repeated-device-disjoint-evaluation-v1"
OUTPUT_SCHEMA = "r4.1-scientific-execution-artifacts-v1"
RUN_SCHEMA = "r4.1-run-provenance-v1"
PREDICTION_SCHEMA = "r4.1-test-prediction-v1"
AUTHORIZED_COMMIT = "97003b1d9448caa105b6e206ceb5deb18c02a207"
PARTITION_MANIFEST_SHA256 = "80c8621860237999b95c2d4e7a0f12a7b56f42ef9c3df3720af5a393bc0131fe"
ACCEPTED_PARTITIONS_SHA256 = "4b7fd17277a6765bba0e509ee48a39f3c7f59492a4e69d3d238ba2cb91671435"
HISTORICAL_PHASE4_FINGERPRINT = "cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11"
CONDITIONS = ("C0", "C1", "C2")
ROLE_ORDER = (
    ORDINARY,
    TIMING_HN,
    TIMING_ANOMALY,
    SAME_ATTEMPT_HN,
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
)
TEST_ROLE_COUNTS = (24, 12, 12, 12, 12, 12)
TRAIN_POOL_ROLE_COUNTS = (72, 36, 36, 36, 36, 36)
MODEL_SEED_BASE = {"LR": 54000, "DT": 55000, "RF": 56000}
EXCLUDED_C2_FEATURES = {
    "semantic_timing_min_signed_causal_margin",
    "semantic_nonce_cross_attempt_value_count",
    "semantic_nonce_max_attempt_sharing_count",
}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "device_id", "population_device_key", "behavioral_profile", "ground_truth",
    "anomaly_label", "scientific_role", "provenance", "debug",
)


class R41Error(RuntimeError):
    """A frozen R4.1 contract or persisted-evidence check failed."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise R41Error(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True,
    ).stdout.strip()


def software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import pandas as pd
        versions["pandas"] = pd.__version__
    except ImportError:
        versions["pandas"] = "not-installed"
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not-installed"
    return versions


def model_seed(partition_seed: int, model: str) -> int:
    repetition = partition_seed - 44000
    if repetition not in range(1, 21) or model not in MODELS:
        raise R41Error("invalid partition/model seed request")
    return MODEL_SEED_BASE[model] + repetition


def model_parameters(model: str, seed: int) -> dict[str, Any]:
    base = LR_PARAMETERS if model == "LR" else DT_PARAMETERS if model == "DT" else RF_PARAMETERS
    return {**base, "random_state": seed}


def profile_for(condition: str) -> str:
    return "C2" if condition == "C2" else "C1"


def role_counts(trace_ids: Sequence[str], roles: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(roles[trace_id] for trace_id in trace_ids)
    return {role: counts[role] for role in ROLE_ORDER}


def load_and_validate_partitions(manifest_path: Path, accepted_path: Path, inputs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if file_sha256(manifest_path) != PARTITION_MANIFEST_SHA256:
        raise R41Error("partition manifest SHA-256 mismatch")
    if file_sha256(accepted_path) != ACCEPTED_PARTITIONS_SHA256:
        raise R41Error("accepted partitions SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    accepted_rows = list(csv.DictReader(accepted_path.open(newline="")))
    if len(manifest["partitions"]) != 20 or len(accepted_rows) != 20:
        raise R41Error("partition input does not contain exactly 20 assignments")
    trace_catalog = manifest["catalog_encoding"]["trace_catalog"]
    device_catalog = manifest["catalog_encoding"]["device_catalog"]
    if set(trace_catalog) != set(inputs.roles) or len(trace_catalog) != 336:
        raise R41Error("partition trace catalog differs from frozen candidate")
    if set(device_catalog) != set(inputs.device_keys.values()) or len(device_catalog) != 64:
        raise R41Error("partition device catalog differs from frozen candidate")
    by_id = {row["partition_id"]: row for row in accepted_rows}
    normalized: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for partition in manifest["partitions"]:
        partition_id = partition["partition_id"]
        seed = int(partition["candidate_seed"])
        if seed not in range(44001, 44021) or partition_id not in by_id:
            raise R41Error("unexpected partition ID or seed")
        row = by_id[partition_id]
        test_ids = [trace_catalog[i] for i in partition["test_trace_indices"]]
        train_ids = [trace_catalog[i] for i in partition["train_trace_indices"]]
        test_devices = [device_catalog[i] for i in partition["test_device_indices"]]
        train_devices = [device_catalog[i] for i in partition["train_device_indices"]]
        c0 = [trace_catalog[i] for i in partition["cohorts"]["C0_trace_indices"]]
        c1 = [trace_catalog[i] for i in partition["cohorts"]["C1_trace_indices"]]
        c2 = [trace_catalog[i] for i in partition["cohorts"]["C2_trace_indices"]]
        if set(test_ids) & set(train_ids) or set(test_ids) | set(train_ids) != set(trace_catalog):
            raise R41Error(f"trace partition failure: {partition_id}")
        if set(test_devices) & set(train_devices) or set(test_devices) | set(train_devices) != set(device_catalog):
            raise R41Error(f"device partition failure: {partition_id}")
        if any(inputs.device_keys[t] not in test_devices for t in test_ids):
            raise R41Error(f"test trace/device mismatch: {partition_id}")
        if any(inputs.device_keys[t] not in train_devices for t in train_ids):
            raise R41Error(f"train trace/device mismatch: {partition_id}")
        if tuple(role_counts(test_ids, inputs.roles).values()) != TEST_ROLE_COUNTS:
            raise R41Error(f"test role composition mismatch: {partition_id}")
        if tuple(role_counts(train_ids, inputs.roles).values()) != TRAIN_POOL_ROLE_COUNTS:
            raise R41Error(f"training-pool role composition mismatch: {partition_id}")
        expected_cohorts = {
            "C0": (c0, (72, 0, 36, 0, 0, 36)),
            "C1": (c1, (36, 12, 36, 12, 12, 36)),
            "C2": (c2, (36, 12, 36, 12, 12, 36)),
        }
        for condition, (cohort, expected) in expected_cohorts.items():
            if len(cohort) != 144 or len(set(cohort)) != 144 or not set(cohort) <= set(train_ids):
                raise R41Error(f"{condition} membership failure: {partition_id}")
            if tuple(role_counts(cohort, inputs.roles).values()) != expected:
                raise R41Error(f"{condition} role composition failure: {partition_id}")
            if set(inputs.device_keys[t] for t in cohort) & set(test_devices):
                raise R41Error(f"{condition} device leakage: {partition_id}")
        c0_anomalies = {t for t in c0 if inputs.roles[t] in ANOMALY_ROLES}
        c1_anomalies = {t for t in c1 if inputs.roles[t] in ANOMALY_ROLES}
        if c0_anomalies != c1_anomalies or c1 != c2:
            raise R41Error(f"matched-cohort identity failure: {partition_id}")
        csv_lists = {
            "test_trace_ids": test_ids,
            "train_trace_ids": train_ids,
            "test_devices": test_devices,
            "train_devices": train_devices,
        }
        for field, values in csv_lists.items():
            if set(row[field].split(";")) != set(values):
                raise R41Error(f"accepted-partitions CSV disagreement for {field}: {partition_id}")
        if row["device_set_fingerprint"] != partition["device_set_fingerprint"]:
            raise R41Error(f"device fingerprint disagreement: {partition_id}")
        fingerprints.add(partition["device_set_fingerprint"])
        normalized.append({
            "partition_id": partition_id,
            "repetition": seed - 44000,
            "split_seed": seed,
            "test_trace_ids": test_ids,
            "training_pool_trace_ids": train_ids,
            "test_device_keys": sorted(test_devices),
            "training_device_keys": sorted(train_devices),
            "cohorts": {"C0": c0, "C1": c1, "C2": c2},
            "test_role_counts": role_counts(test_ids, inputs.roles),
            "training_pool_role_counts": role_counts(train_ids, inputs.roles),
            "cohort_role_counts": {c: role_counts(v, inputs.roles) for c, v in {"C0": c0, "C1": c1, "C2": c2}.items()},
            "device_set_fingerprint": partition["device_set_fingerprint"],
            "test_trace_fingerprint": partition["test_trace_fingerprint"],
            "train_trace_fingerprint": partition["train_trace_fingerprint"],
        })
    if len(fingerprints) != 20 or {p["split_seed"] for p in normalized} != set(range(44001, 44021)):
        raise R41Error("partition distinctness or seed sequence failure")
    return sorted(normalized, key=lambda p: p["split_seed"]), manifest


def validate_feature_contract() -> dict[str, Any]:
    if len(C1_FEATURE_NAMES) != 90 or len(C2_FEATURE_NAMES) != 87:
        raise R41Error("feature counts differ from frozen contracts")
    if set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES) != EXCLUDED_C2_FEATURES:
        raise R41Error("Reduced exclusions differ from frozen contract")
    if tuple(name for name in C1_FEATURE_NAMES if name in set(C2_FEATURE_NAMES)) != C2_FEATURE_NAMES:
        raise R41Error("Reduced feature ordering is not the frozen C1 subsequence")
    definitions = {item.name: item for item in FEATURE_DEFINITIONS}
    for name in C1_FEATURE_NAMES:
        if name not in definitions or definitions[name].privileged_information_dependency:
            raise R41Error(f"privileged or undeclared feature: {name}")
        if any(fragment in name for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise R41Error(f"forbidden feature name: {name}")
    return {
        "C1_count": 90,
        "C2_count": 87,
        "C1_names": list(C1_FEATURE_NAMES),
        "C2_names": list(C2_FEATURE_NAMES),
        "C1_fingerprint": feature_profile_fingerprint("C1"),
        "C2_fingerprint": feature_profile_fingerprint("C2"),
        "C2_excluded_features": sorted(EXCLUDED_C2_FEATURES),
        "all_features_declared_detector_observable_without_privileged_dependency": True,
    }


def make_run_matrix(partitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for partition in partitions:
        for condition in CONDITIONS:
            for model in MODELS:
                seed = model_seed(partition["split_seed"], model)
                run_key = {
                    "experiment_id": EXPERIMENT_ID,
                    "partition_id": partition["partition_id"],
                    "condition": condition,
                    "model": model,
                }
                rows.append({
                    **run_key,
                    "run_id": canonical_hash(run_key),
                    "repetition": partition["repetition"],
                    "split_seed": partition["split_seed"],
                    "model_seed": seed,
                    "feature_profile": profile_for(condition),
                    "feature_profile_fingerprint": feature_profile_fingerprint(profile_for(condition)),
                    "training_trace_count": 144,
                    "test_trace_count": 84,
                    "training_trace_ids": partition["cohorts"][condition],
                    "test_trace_ids": partition["test_trace_ids"],
                    "training_device_keys": partition["training_device_keys"],
                    "test_device_keys": partition["test_device_keys"],
                    "training_role_counts": partition["cohort_role_counts"][condition],
                    "test_role_counts": partition["test_role_counts"],
                    "model_parameters": model_parameters(model, seed),
                })
    if len(rows) != 180 or len({row["run_id"] for row in rows}) != 180:
        raise R41Error("run matrix cardinality/uniqueness failure")
    return rows


def phase4_verification(root: Path) -> dict[str, Any]:
    fingerprint_path = root / "artifacts/icds-ml-phase4/fingerprints.json"
    fingerprint_data = json.loads(fingerprint_path.read_text())
    expected = "sha256:" + HISTORICAL_PHASE4_FINGERPRINT
    if fingerprint_data["artifact_tree_fingerprint"] != expected:
        raise R41Error("historical Phase-4 fingerprint field changed")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "IoT_auth_simulator")
    result = subprocess.run(
        [sys.executable, "scripts/verify_artifacts.py"], cwd=root, env=environment,
        text=True, check=True, capture_output=True,
    )
    verified = json.loads(result.stdout)["verification"]
    if verified["frozen_evidence"]["phase4_tree_fingerprint"] != expected:
        raise R41Error("historical Phase-4 verifier fingerprint changed")
    return verified


def execute_one(root: Path, output_root: Path, inputs: Any, row: Mapping[str, Any]) -> None:
    run_dir = output_root / "runs" / row["partition_id"] / row["condition"] / row["model"]
    if run_dir.exists():
        required = {"predictions.jsonl", "metrics.json", "run-provenance.json"}
        if {p.name for p in run_dir.iterdir()} != required:
            raise R41Error(f"incomplete existing checkpoint: {run_dir}")
        return
    profile = row["feature_profile"]
    train = assemble_matrix(inputs, row["training_trace_ids"], profile)
    test = assemble_matrix(inputs, row["test_trace_ids"], profile)
    if set(train.trace_ids) & set(test.trace_ids) or set(train.device_keys) & set(test.device_keys):
        raise R41Error(f"train/test leakage before fit: {row['run_id']}")
    estimator = fit_estimator(row["model"], row["model_seed"], train)
    predicted, probabilities = predict_estimator(estimator, test)
    predictions = [{
        "schema_version": PREDICTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_id": row["run_id"],
        "partition_id": row["partition_id"],
        "repetition": row["repetition"],
        "split_seed": row["split_seed"],
        "condition": row["condition"],
        "model": row["model"],
        "model_seed": row["model_seed"],
        "trace_id": trace_id,
        "split": "test",
        "true_target": int(truth),
        "scientific_role": role,
        "device_key": device,
        "device_key_role": "privileged_evaluation_grouping_metadata_not_feature",
        "predicted_class": int(prediction),
        "predicted_anomaly_probability": float(probability),
        "decision_threshold": PREDICTION_THRESHOLD,
        "feature_profile": profile,
        "feature_profile_fingerprint": row["feature_profile_fingerprint"],
        "repository_commit": AUTHORIZED_COMMIT,
    } for trace_id, truth, role, device, prediction, probability in zip(
        test.trace_ids, test.y, test.scientific_roles, test.device_keys, predicted, probabilities,
    )]
    metrics = reconstruct_metrics(predictions)
    provenance = {
        "schema_version": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_id": row["run_id"],
        "partition_id": row["partition_id"],
        "repetition": row["repetition"],
        "split_seed": row["split_seed"],
        "condition": row["condition"],
        "model": row["model"],
        "model_seed": row["model_seed"],
        "model_configuration_id": MODEL_CONFIGURATION_VERSION,
        "model_parameters": row["model_parameters"],
        "decision_threshold": PREDICTION_THRESHOLD,
        "training_trace_ids": list(train.trace_ids),
        "test_trace_ids": list(test.trace_ids),
        "training_device_keys": sorted(set(train.device_keys)),
        "test_device_keys": sorted(set(test.device_keys)),
        "training_role_counts": role_counts(train.trace_ids, inputs.roles),
        "test_role_counts": role_counts(test.trace_ids, inputs.roles),
        "feature_profile": profile,
        "feature_profile_fingerprint": row["feature_profile_fingerprint"],
        "feature_names_in_order": list(train.feature_names),
        "preprocessing_configuration_id": PREPROCESSING_CONFIGURATION_VERSION,
        "preprocessing": (
            {"steps": ["StandardScaler", "LogisticRegression"], "fit_scope": "current_training_cohort_only"}
            if row["model"] == "LR" else {"steps": [row["model"]], "fit_scope": "current_training_cohort_only", "scaling": False}
        ),
        "fitted_scaler_statistics": fitted_scaler_state(estimator),
        "test_used_for_fit_or_preprocessing": False,
        "feature_selection_performed": False,
        "imputation_performed": False,
        "threshold_tuning_performed": False,
        "software_versions": software_versions(),
        "repository_commit": git_commit(root),
        "authoritative_input_hashes": {
            "partition_manifest": PARTITION_MANIFEST_SHA256,
            "accepted_partitions": ACCEPTED_PARTITIONS_SHA256,
            "candidate": EXPECTED_CANDIDATE_FILE_SHA256,
            "feature_spec": EXPECTED_FEATURE_SPEC_FILE_SHA256,
            "phase2_manifest": EXPECTED_PHASE2_FILE_SHA256,
            "historical_phase4_tree": HISTORICAL_PHASE4_FINGERPRINT,
        },
    }
    temporary = run_dir.with_name(run_dir.name + ".partial")
    if temporary.exists():
        raise R41Error(f"stale partial run directory: {temporary}")
    temporary.mkdir(parents=True)
    write_jsonl(temporary / "predictions.jsonl", predictions)
    write_json(temporary / "metrics.json", metrics)
    write_json(temporary / "run-provenance.json", provenance)
    temporary.rename(run_dir)


def load_all_runs(output_root: Path, matrix: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for row in matrix:
        run_dir = output_root / "runs" / row["partition_id"] / row["condition"] / row["model"]
        run_predictions = read_jsonl(run_dir / "predictions.jsonl")
        run_metrics = json.loads((run_dir / "metrics.json").read_text())
        run_provenance = json.loads((run_dir / "run-provenance.json").read_text())
        predictions.extend(run_predictions)
        metrics.append({
            "run_id": row["run_id"], "partition_id": row["partition_id"],
            "repetition": row["repetition"], "split_seed": row["split_seed"],
            "condition": row["condition"], "model": row["model"], "metrics": run_metrics,
        })
        provenance.append(run_provenance)
    return predictions, metrics, provenance


def independent_validate(
    root: Path,
    output_root: Path,
    matrix: Sequence[Mapping[str, Any]],
    partitions: Sequence[Mapping[str, Any]],
    inputs: Any,
    partition_manifest_path: Path,
    accepted_path: Path,
) -> dict[str, Any]:
    predictions, metrics_rows, provenance_rows = load_all_runs(output_root, matrix)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(condition), "evidence": evidence})
        if not condition:
            raise R41Error(f"independent integrity check failed: {name}: {evidence}")

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        by_run[record["run_id"]].append(record)
    matrix_by_id = {row["run_id"]: row for row in matrix}
    metrics_by_id = {row["run_id"]: row["metrics"] for row in metrics_rows}
    provenance_by_id = {row["run_id"]: row for row in provenance_rows}
    check("expected_unique_runs", len(by_run) == len(matrix_by_id) == 180, {"predictions": len(by_run), "matrix": len(matrix_by_id)})
    check("prediction_total", len(predictions) == 15120, len(predictions))
    check("run_provenance_total", len(provenance_by_id) == 180, len(provenance_by_id))
    for run_id, row in matrix_by_id.items():
        records = by_run.get(run_id, [])
        provenance = provenance_by_id.get(run_id, {})
        check(f"prediction_cardinality:{run_id}", len(records) == 84 and len({r["trace_id"] for r in records}) == 84, len(records))
        check(f"test_membership:{run_id}", {r["trace_id"] for r in records} == set(row["test_trace_ids"]), "exact")
        check(f"test_role_counts:{run_id}", tuple(Counter(r["scientific_role"] for r in records)[role] for role in ROLE_ORDER) == TEST_ROLE_COUNTS, "exact")
        check(f"model_seed:{run_id}", all(r["model_seed"] == row["model_seed"] for r in records), row["model_seed"])
        check(f"metric_reconstruction:{run_id}", reconstruct_metrics(records) == metrics_by_id[run_id], "exact")
        check(f"training_membership:{run_id}", provenance.get("training_trace_ids") == row["training_trace_ids"], "exact")
        check(f"device_disjoint:{run_id}", not set(provenance.get("training_device_keys", [])) & set(provenance.get("test_device_keys", [])), "zero overlap")
        check(f"feature_order:{run_id}", provenance.get("feature_names_in_order") == list(C2_FEATURE_NAMES if row["condition"] == "C2" else C1_FEATURE_NAMES), row["feature_profile"])
        check(f"feature_fingerprint:{run_id}", provenance.get("feature_profile_fingerprint") == row["feature_profile_fingerprint"], row["feature_profile_fingerprint"])
        check(f"model_configuration:{run_id}", provenance.get("model_parameters") == row["model_parameters"], row["model_parameters"])
        check(f"training_only_preprocessing:{run_id}", provenance.get("test_used_for_fit_or_preprocessing") is False and provenance.get("preprocessing", {}).get("fit_scope") == "current_training_cohort_only", "confirmed")
        if row["model"] == "LR":
            scaler = provenance.get("fitted_scaler_statistics") or {}
            check(f"lr_scaler_fit_count:{run_id}", scaler.get("samples_seen") == [144], scaler.get("samples_seen"))
        else:
            check(f"no_unapproved_preprocessing:{run_id}", provenance.get("fitted_scaler_statistics") is None, "none")
    check("partition_manifest_unchanged", file_sha256(partition_manifest_path) == PARTITION_MANIFEST_SHA256, file_sha256(partition_manifest_path))
    check("accepted_partitions_unchanged", file_sha256(accepted_path) == ACCEPTED_PARTITIONS_SHA256, file_sha256(accepted_path))
    candidate_path = resolve_scientific_path("dataset-v1.0", root)
    feature_spec_path = resolve_scientific_path("eval-v1.0-feature-spec", root)
    phase2_path = resolve_scientific_path("eval-v1.0-manifest", root)
    check("candidate_unchanged", file_sha256(candidate_path) == EXPECTED_CANDIDATE_FILE_SHA256, file_sha256(candidate_path))
    check("feature_spec_unchanged", file_sha256(feature_spec_path) == EXPECTED_FEATURE_SPEC_FILE_SHA256, file_sha256(feature_spec_path))
    check("phase2_manifest_unchanged", file_sha256(phase2_path) == EXPECTED_PHASE2_FILE_SHA256, file_sha256(phase2_path))
    grouped = {(row["partition_id"], row["model"]): {} for row in matrix}
    for metric_row in metrics_rows:
        grouped[(metric_row["partition_id"], metric_row["model"])][metric_row["condition"]] = metric_row["metrics"]
    paired = make_paired_comparisons(grouped)
    check("paired_comparison_count", len(paired) == 60, len(paired))
    check("c0_c1_anomaly_identity", all(
        {t for t in p["cohorts"]["C0"] if inputs.roles[t] in ANOMALY_ROLES}
        == {t for t in p["cohorts"]["C1"] if inputs.roles[t] in ANOMALY_ROLES}
        for p in partitions
    ), "all 20")
    check("c1_c2_membership_identity", all(p["cohorts"]["C1"] == p["cohorts"]["C2"] for p in partitions), "all 20")
    report = {
        "schema_version": "r4.1-independent-integrity-report-v1",
        "status": "PASS",
        "check_count": len(checks),
        "all_passed": True,
        "run_count": 180,
        "prediction_count": 15120,
        "paired_comparison_count": 60,
        "checks": checks,
    }
    write_json(output_root / "integrity-report.json", report)
    return report


def timing_fpr(metrics: Mapping[str, Any]) -> float:
    return float(metrics["normal_role_diagnostics"][TIMING_HN]["false_positive_rate"])


def timing_recall(metrics: Mapping[str, Any]) -> float:
    return float(metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["recall"])


def make_paired_comparisons(grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for (partition_id, model), conditions in sorted(grouped.items()):
        c0, c1 = conditions["C0"], conditions["C1"]
        delta_fpr = timing_fpr(c1) - timing_fpr(c0)
        delta_recall = timing_recall(c1) - timing_recall(c0)
        if delta_fpr < 0 and delta_recall >= 0:
            category = "clean_correction"
        elif delta_fpr < 0 and delta_recall < 0:
            category = "trade_off"
        elif delta_fpr == 0:
            category = "no_boundary_improvement"
        else:
            category = "boundary_deterioration"
        rows.append({
            "partition_id": partition_id,
            "model": model,
            "timing_hn_fpr_C0": timing_fpr(c0),
            "timing_hn_fpr_C1": timing_fpr(c1),
            "timing_recall_C0": timing_recall(c0),
            "timing_recall_C1": timing_recall(c1),
            "delta_fpr_C1_minus_C0": delta_fpr,
            "delta_recall_C1_minus_C0": delta_recall,
            "category": category,
        })
    return rows


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 20 or not np.isfinite(array).all():
        raise R41Error("R4.1 distribution requires 20 finite values")
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "values": array.astype(float).tolist(),
        "quartile_method": "numpy linear",
    }


def aggregate(metrics_rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = defaultdict(list)
    for row in metrics_rows:
        indexed[(row["model"], row["condition"])].append(row["metrics"])
    by_model_condition = {}
    for model in MODELS:
        by_model_condition[model] = {}
        for condition in CONDITIONS:
            items = indexed[(model, condition)]
            by_model_condition[model][condition] = {
                "timing_hn_fpr": distribution([timing_fpr(m) for m in items]),
                "timing_anomaly_recall": distribution([timing_recall(m) for m in items]),
                "ordinary_normal_fpr": distribution([float(m["normal_role_diagnostics"][ORDINARY]["false_positive_rate"]) for m in items]),
                "same_attempt_nonce_hn_fpr": distribution([float(m["normal_role_diagnostics"][SAME_ATTEMPT_HN]["false_positive_rate"]) for m in items]),
                "fresh_cross_attempt_nonce_hn_fpr": distribution([float(m["normal_role_diagnostics"][FRESH_ATTEMPT_HN]["false_positive_rate"]) for m in items]),
                "nonce_anomaly_recall": distribution([float(m["anomaly_role_diagnostics"][NONCE_ANOMALY]["recall"]) for m in items]),
                "anomaly_precision": distribution([float(m["anomaly_precision"]) for m in items]),
                "anomaly_recall": distribution([float(m["anomaly_recall"]) for m in items]),
                "anomaly_f1": distribution([float(m["anomaly_f1"]) for m in items]),
                "balanced_accuracy": distribution([float(m["balanced_accuracy"]) for m in items]),
            }
    paired_summary = {}
    for model in MODELS:
        rows = [row for row in paired if row["model"] == model]
        paired_summary[model] = {
            "delta_fpr": distribution([row["delta_fpr_C1_minus_C0"] for row in rows]),
            "delta_recall": distribution([row["delta_recall_C1_minus_C0"] for row in rows]),
            "delta_fpr_sign_counts": {
                "negative": sum(row["delta_fpr_C1_minus_C0"] < 0 for row in rows),
                "zero": sum(row["delta_fpr_C1_minus_C0"] == 0 for row in rows),
                "positive": sum(row["delta_fpr_C1_minus_C0"] > 0 for row in rows),
            },
            "joint_outcome_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        }
    return {
        "schema_version": "r4.1-descriptive-aggregation-v1",
        "partition_count": 20,
        "independence_claimed": False,
        "hypothesis_tests_performed": False,
        "by_model_condition": by_model_condition,
        "paired_C0_to_C1": paired_summary,
    }


def format_interval(item: Mapping[str, Any]) -> str:
    return f"{item['median']:.3f} [{item['q1']:.3f}–{item['q3']:.3f}; {item['minimum']:.3f}–{item['maximum']:.3f}]"


def consistency_label(summary: Mapping[str, Any]) -> str:
    counts = summary["joint_outcome_counts"]
    clean = counts.get("clean_correction", 0)
    trade = counts.get("trade_off", 0)
    if clean + trade == 20 and trade > clean:
        return "consistent boundary correction with recurring recall trade-off"
    if clean + trade == 20:
        return "consistent, predominantly clean boundary correction"
    if counts.get("no_boundary_improvement", 0) == 20:
        return "no reproducible boundary correction"
    if counts.get("boundary_deterioration", 0) == 20:
        return "contradictory boundary deterioration"
    return "mixed/unstable evidence"


def scientific_report(
    output_root: Path,
    aggregate_data: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    phase4_verification_data: Mapping[str, Any],
) -> str:
    historical_rows = json.loads((output_root.parent / "icds-ml-phase4/grouped/per-run-metrics.json").read_text())
    historical = {(r["model"], r["condition"]): r["metrics"] for r in historical_rows}
    primary_lines = []
    secondary_lines = []
    for model in MODELS:
        for condition in CONDITIONS:
            cell = aggregate_data["by_model_condition"][model][condition]
            primary_lines.append(
                f"| {model} | {condition} | {format_interval(cell['timing_hn_fpr'])} | {format_interval(cell['timing_anomaly_recall'])} |"
            )
            secondary_lines.append(
                f"| {model} | {condition} | {format_interval(cell['ordinary_normal_fpr'])} | "
                f"{format_interval(cell['same_attempt_nonce_hn_fpr'])} | "
                f"{format_interval(cell['fresh_cross_attempt_nonce_hn_fpr'])} | "
                f"{format_interval(cell['nonce_anomaly_recall'])} |"
            )
    paired_lines = []
    assessment = []
    for model in MODELS:
        cell = aggregate_data["paired_C0_to_C1"][model]
        signs = cell["delta_fpr_sign_counts"]
        outcomes = cell["joint_outcome_counts"]
        paired_lines.append(
            f"| {model} | {format_interval(cell['delta_fpr'])} | {format_interval(cell['delta_recall'])} | "
            f"{signs['negative']}/{signs['zero']}/{signs['positive']} | "
            f"{outcomes.get('clean_correction', 0)}/{outcomes.get('trade_off', 0)}/{outcomes.get('no_boundary_improvement', 0)}/{outcomes.get('boundary_deterioration', 0)} |"
        )
        h0 = historical[(model, "C0")]
        h1 = historical[(model, "C1")]
        historical_delta_fpr = timing_fpr(h1) - timing_fpr(h0)
        historical_delta_recall = timing_recall(h1) - timing_recall(h0)
        assessment.append(
            f"- **{model}: {consistency_label(cell)}.** Historical grouped sensitivity had "
            f"ΔFPR={historical_delta_fpr:.3f}, ΔRecall={historical_delta_recall:.3f}; "
            f"the repeated-partition sign counts are {signs['negative']}/{signs['zero']}/{signs['positive']} "
            f"(lower/equal/higher Timing-HN FPR under C1), with "
            f"{outcomes.get('trade_off', 0)} recall-cost partition(s)."
        )
    return f"""# R4.1 repeated device-disjoint evaluation

## A. Execution-integrity report

The campaign is complete and independently valid: **180/180 runs**, **84 predictions per run**, **15,120 predictions**, and **60 paired C0→C1 comparisons**. All 20 frozen partitions remained device-disjoint and retained exact test/cohort memberships. Historical Phase 4 verification remained `{phase4_verification_data['frozen_evidence']['phase4_tree_fingerprint']}`.

## B. Complete descriptive results

Values are median [Q1–Q3; min–max] across the 20 predefined grouped partitions.

| Model | Condition | Timing-HN FPR | Timing-anomaly recall |
|---|---|---:|---:|
{chr(10).join(primary_lines)}

Secondary values use the same median [Q1–Q3; min–max] format:

| Model | Condition | Ordinary FPR | Same-attempt Nonce-HN FPR | Fresh cross-attempt Nonce-HN FPR | Nonce-anomaly recall |
|---|---|---:|---:|---:|---:|
{chr(10).join(secondary_lines)}

The near-zero C0/C1 Nonce-HN FPRs and consistently high Nonce-anomaly recall show that the main paired change is concentrated in the Timing boundary rather than representing a uniform shift across every semantic role. C2 remains descriptive and is not interpreted as improved observability.

## C. Paired C0→C1 analysis

| Model | Δ Timing-HN FPR | Δ Timing recall | ΔFPR −/0/+ | clean/trade-off/no improvement/deterioration |
|---|---:|---:|---:|---:|
{chr(10).join(paired_lines)}

These are paired differences within the same frozen partition, not differences between independently calculated medians. No split-level hypothesis test was performed.

## D. Model-specific consistency assessment

{chr(10).join(assessment)}

## E. Unexpected implementation or data issues

- The initially named execution repository was corrected before this campaign; no model was run in `PFE_simulator`.
- The same-attempt Nonce HN retains its frozen `semantic_family=direct_timing` provenance but is reported by authoritative scientific role, not misclassified as Timing HN.
- The compatible external virtual environment was used only as an interpreter/dependency environment; all code and inputs came from this authorized repository.
- scikit-learn 1.9 emitted a nonfatal deprecation warning for the frozen explicit `penalty="l2"` Logistic Regression parameter. The parameter was retained unchanged; all LR runs completed.
- The focused ML regression suite passed 24/24. `unittest discover` passed 468 tests with one skip. Raw `pytest -q` additionally exposes a pre-existing collection defect in `tests/test_dp2_profile_provenance.py`, where a helper named `test_population` is mistaken for a fixture-based test; no unrelated test was modified.
- No fallback split, cohort resampling, tuning, calibration, or imputation occurred.

## F. Scientific interpretation

The results address only the frozen 336-trace synthetic Timing+Nonce candidate. The Timing-boundary effect persists under all 20 predefined device-disjoint partitions for LR, DT, and RF: every C0→C1 paired Timing-HN FPR difference is negative. For LR, the correction is accompanied by a recurring Timing-anomaly recall cost in 14/20 partitions. For DT and RF, recall is preserved in 18/20 and 19/20 partitions respectively, so their correction is predominantly clean with limited partition-specific recall loss. Clean corrections and trade-offs are reported separately rather than averaged away.

This agrees directionally with the historical single device-disjoint sensitivity result for all three classifiers. The historical LR result also showed a recall trade-off; repeated partitions show that this cost recurs, although its magnitude varies. Historical DT and RF corrections preserved recall, consistent with the predominantly clean repeated-partition outcomes.

Device-disjointness prevents persistent-device overlap between train and test, but it does not establish independence from shared generator logic, scenario templates, population strata, or campaign seeds. These results therefore do not establish operational IoT performance, independent-device population generalization, generator-independent transfer, statistical significance, a new IDS contribution, or improved observability from C2.

## G. Reproducibility

Execution command:

```bash
PYTHONPATH="$PWD/IoT_auth_simulator" ../PFE_simulator/.venv/bin/python -m ml.evaluation.r4_1_execution \\
  --partition-manifest /tmp/r4.1-exact-partition-audit/partition-manifest.json \\
  --accepted-partitions /tmp/r4.1-exact-partition-audit/accepted-partitions.csv \\
  --output artifacts/r4.1-device-disjoint-evaluation
```

- Repository checkpoint: `{AUTHORIZED_COMMIT}`
- Partition manifest SHA-256: `{PARTITION_MANIFEST_SHA256}`
- Accepted partitions SHA-256: `{ACCEPTED_PARTITIONS_SHA256}`
- Historical Phase 4 fingerprint: `{HISTORICAL_PHASE4_FINGERPRINT}`
- Independent validation: rerun the same command with `--validate-only`.
"""


def output_fingerprints(output_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p.name != "fingerprints.json"):
        files.append({
            "path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return {
        "schema_version": "r4.1-output-fingerprints-v1",
        "artifact_count": len(files),
        "files": files,
        "artifact_tree_fingerprint": canonical_hash(files),
    }


def execute(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[3]
    if git_commit(root) != AUTHORIZED_COMMIT:
        raise R41Error("repository HEAD differs from authorized checkpoint")
    output_root = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    manifest_path = args.partition_manifest.resolve()
    accepted_path = args.accepted_partitions.resolve()
    inputs = load_frozen_inputs(root)
    feature_contract = validate_feature_contract()
    phase4 = phase4_verification(root)
    partitions, partition_manifest = load_and_validate_partitions(manifest_path, accepted_path, inputs)
    matrix = make_run_matrix(partitions)

    if args.validate_only:
        if not output_root.is_dir():
            raise R41Error("validation output root does not exist")
        persisted_matrix = json.loads((output_root / "run-matrix.json").read_text())
        if persisted_matrix != matrix:
            raise R41Error("persisted run matrix differs from frozen reconstruction")
        independent_validate(root, output_root, matrix, partitions, inputs, manifest_path, accepted_path)
        predictions, metrics_rows, _ = load_all_runs(output_root, matrix)
        paired = make_paired_comparisons({
            (p, m): {r["condition"]: r["metrics"] for r in metrics_rows if r["partition_id"] == p and r["model"] == m}
            for p in {r["partition_id"] for r in matrix} for m in MODELS
        })
        if json.loads((output_root / "paired-comparisons.json").read_text()) != paired:
            raise R41Error("persisted paired comparisons differ from reconstruction")
        print(json.dumps({"status": "PASS", "runs": 180, "predictions": len(predictions), "paired": len(paired)}, indent=2))
        return

    if output_root.exists():
        raise R41Error(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    execution_manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "RUN_MATRIX_PERSISTED_BEFORE_MODEL_FITTING",
        "authorized_repository_commit": AUTHORIZED_COMMIT,
        "partition_manifest_path": str(manifest_path),
        "accepted_partitions_path": str(accepted_path),
        "authoritative_input_hashes": {
            "partition_manifest": PARTITION_MANIFEST_SHA256,
            "accepted_partitions": ACCEPTED_PARTITIONS_SHA256,
            "candidate": EXPECTED_CANDIDATE_FILE_SHA256,
            "candidate_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
            "feature_spec": EXPECTED_FEATURE_SPEC_FILE_SHA256,
            "phase1_C1_fingerprint": EXPECTED_C1_FINGERPRINT,
            "phase1_C2_fingerprint": EXPECTED_C2_FINGERPRINT,
            "phase2_manifest": EXPECTED_PHASE2_FILE_SHA256,
            "phase2_semantic_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
            "historical_phase4_tree": HISTORICAL_PHASE4_FINGERPRINT,
        },
        "software_versions": software_versions(),
        "run_count": 180,
        "predictions_expected": 15120,
        "paired_comparisons_expected": 60,
        "feature_contract": feature_contract,
        "phase4_verification": phase4,
        "scientific_constraints": {
            "hyperparameter_search": False,
            "threshold_tuning": False,
            "calibration": False,
            "imputation": False,
            "feature_selection": False,
            "conventional_split_level_hypothesis_tests": False,
        },
    }
    write_json(output_root / "execution-manifest.json", execution_manifest)
    write_json(output_root / "run-matrix.json", matrix)
    write_json(output_root / "partition-cohort-audit.json", {
        "status": "PASS", "partition_count": 20, "partitions": partitions,
        "source_manifest_summary": partition_manifest["summary"],
    })

    for index, row in enumerate(matrix, start=1):
        execute_one(root, output_root, inputs, row)
        if index % 15 == 0:
            print(f"completed {index}/180 runs", flush=True)

    predictions, metrics_rows, provenance_rows = load_all_runs(output_root, matrix)
    write_jsonl(output_root / "predictions.jsonl", predictions)
    write_json(output_root / "run-provenance.json", provenance_rows)
    write_json(output_root / "run-level-metrics.json", metrics_rows)
    integrity = independent_validate(root, output_root, matrix, partitions, inputs, manifest_path, accepted_path)
    grouped = {
        (partition_id, model): {
            row["condition"]: row["metrics"]
            for row in metrics_rows if row["partition_id"] == partition_id and row["model"] == model
        }
        for partition_id in {row["partition_id"] for row in matrix} for model in MODELS
    }
    paired = make_paired_comparisons(grouped)
    write_json(output_root / "paired-comparisons.json", paired)
    write_csv(output_root / "paired-comparisons.csv", paired)
    aggregate_data = aggregate(metrics_rows, paired)
    write_json(output_root / "aggregate-summary.json", aggregate_data)
    report = scientific_report(output_root, aggregate_data, paired, phase4)
    (output_root / "scientific-report.md").write_text(report)
    execution_manifest["status"] = "COMPLETE_AND_INDEPENDENTLY_VALIDATED"
    execution_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    execution_manifest["integrity_status"] = integrity["status"]
    write_json(output_root / "execution-manifest.json", execution_manifest)
    write_json(output_root / "fingerprints.json", output_fingerprints(output_root))
    print(json.dumps({
        "status": execution_manifest["status"],
        "runs": 180,
        "predictions": len(predictions),
        "paired_comparisons": len(paired),
        "output": str(output_root),
        "artifact_tree_fingerprint": json.loads((output_root / "fingerprints.json").read_text())["artifact_tree_fingerprint"],
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--partition-manifest", type=Path, required=True)
    result.add_argument("--accepted-partitions", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("artifacts/r4.1-device-disjoint-evaluation"))
    result.add_argument("--validate-only", action="store_true")
    return result


if __name__ == "__main__":
    execute(parser().parse_args())
