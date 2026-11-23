from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
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
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
    ORDINARY,
    SAME_ATTEMPT_HN,
    TIMING_ANOMALY,
    TIMING_HN,
)
from ml.evaluation.metrics import METRIC_CONTRACT_VERSION, reconstruct_metrics
from ml.evaluation.pipeline import (
    MatrixBundle,
    MODEL_CONFIGURATION_VERSION,
    PREPROCESSING_CONFIGURATION_VERSION,
    PREDICTION_THRESHOLD,
    _model_parameters,
    fit_estimator,
    fitted_scaler_state,
    load_frozen_inputs,
    predict_estimator,
)

SCHEMA = "r4.2-representation-sensitivity-execution-v1"
RUN_SCHEMA = "r4.2-run-provenance-v1"
PREDICTION_SCHEMA = "r4.2-test-prediction-v1"
EXPERIMENT_ID = "r4.2-representation-sensitivity-v1"
AUTHORIZED_HEAD = "bd00f098ec86902e5c01e5da1cf1eb4c6e92d77b"
PREEXEC_FINGERPRINT = "sha256:29b2792d1555c51f1d0d5a9aac5ae549ac23c1d6e817023b8acc78e31aab5da1"
R41_TREE_FINGERPRINT = "sha256:ed387b03bcacf12369dbaeb0016ba027db301ffdb7f01ecba7609507906ec284"
R41_AGGREGATE = "sha256:cce473976d8ea079d5ad77031da21a2062a02aaae6121ce451ff0db49eacea34"
HISTORICAL_PHASE4 = "cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11"
PARTITION_SHA = "80c8621860237999b95c2d4e7a0f12a7b56f42ef9c3df3720af5a393bc0131fe"
ACCEPTED_SHA = "4b7fd17277a6765bba0e509ee48a39f3c7f59492a4e69d3d238ba2cb91671435"
MODELS = ("LR", "DT", "RF")
CONDITIONS = ("A", "B", "C", "D")
CONTRASTS = (("A", "C"), ("C", "D"), ("A", "B"), ("C", "B"))
ROLE_ORDER = (ORDINARY, TIMING_HN, TIMING_ANOMALY, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN, NONCE_ANOMALY)


class R42Error(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise R42Error("refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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


def verify_tree(root: Path, manifest: Mapping[str, Any], base: Path) -> int:
    for rec in manifest["files"]:
        path = base / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or file_sha(path) != rec["sha256"]:
            raise R42Error(f"immutable artifact mismatch: {path}")
    return len(manifest["files"])


def preflight(root: Path, output: Path, script_path: Path) -> dict[str, Any]:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty_before = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if branch != "research/r4.2-representation-sensitivity" or head != AUTHORIZED_HEAD or dirty_before:
        raise R42Error(f"repository gate failed: branch={branch}, head={head}, status={dirty_before!r}")
    if output.exists():
        raise R42Error(f"output already exists; explicit resume mode is required: {output}")

    pre = root / "metadata/r4.2-preexecution"
    inv = json.loads((pre / "durable-artifact-inventory.json").read_text())
    inv_core = {k: v for k, v in inv.items() if k != "inventory_fingerprint"}
    if canonical_hash(inv_core) != PREEXEC_FINGERPRINT or inv["inventory_fingerprint"] != PREEXEC_FINGERPRINT:
        raise R42Error("pre-execution fingerprint mismatch")
    for rec in inv["artifacts"]:
        path = root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or file_sha(path) != rec["sha256"]:
            raise R42Error(f"pre-execution artifact mismatch: {rec['path']}")

    freeze = json.loads((root / "metadata/r4.1-freeze/freeze-manifest.json").read_text())
    if freeze["aggregate_r4_1_fingerprint"] != R41_AGGREGATE:
        raise R42Error("R4.1 aggregate mismatch")
    for rec in freeze["authoritative_inventory"]:
        path = root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or file_sha(path) != rec["sha256"]:
            raise R42Error(f"R4.1 freeze inventory mismatch: {rec['path']}")

    r41_root = root / "artifacts/r4.1-device-disjoint-evaluation"
    r41_fp = json.loads((r41_root / "fingerprints.json").read_text())
    if r41_fp["artifact_tree_fingerprint"] != R41_TREE_FINGERPRINT:
        raise R42Error("R4.1 artifact tree fingerprint mismatch")
    verified_r41 = verify_tree(root, r41_fp, r41_root)
    phase4 = json.loads((root / "artifacts/icds-ml-phase4/fingerprints.json").read_text())
    if phase4["artifact_tree_fingerprint"] != "sha256:" + HISTORICAL_PHASE4:
        raise R42Error("historical Phase 4 fingerprint mismatch")

    contracts = json.loads((pre / "feature-contracts.json").read_text())
    plans = json.loads((pre / "execution-manifest-cd.json").read_text())
    anchors = json.loads((pre / "frozen-anchor-runs-ab.json").read_text())
    f2 = json.loads((pre / "f2-generic-temporal-features.json").read_text())
    if plans["run_count"] != 120 or len(plans["plans"]) != 120 or len({r["planned_run_id"] for r in plans["plans"]}) != 120:
        raise R42Error("planned run matrix mismatch")
    if anchors["anchor_count"] != 120:
        raise R42Error("anchor count mismatch")
    if [f["feature_name"] for f in f2["features"]] != f2["ordered_feature_names"] or len(f2["features"]) != 7:
        raise R42Error("F2 manifest mismatch")
    expected_counts = {"A": 90, "B": 87, "C": 89, "D": 82}
    names = {}
    for condition, count in expected_counts.items():
        record = contracts["conditions"][condition]
        names[condition] = [item["feature_name"] for item in record["ordered_features"]]
        if len(names[condition]) != count or len(set(names[condition])) != count:
            raise R42Error(f"{condition} feature contract mismatch")
    if "semantic_timing_relation_count" not in names["C"] or "semantic_timing_relation_count" not in names["D"]:
        raise R42Error("required F3 relation count missing")
    if names["D"] != [name for name in names["C"] if name not in set(f2["ordered_feature_names"])]:
        raise R42Error("D is not the approved C-minus-F2 subsequence")

    inputs = load_frozen_inputs(root)
    for plan in plans["plans"]:
        condition = plan["condition"]
        if condition not in {"C", "D"} or plan["feature_count"] != len(names[condition]):
            raise R42Error("plan feature condition mismatch")
        if plan["ordered_list_fingerprint"] != contracts["conditions"][condition]["ordered_list_fingerprint"]:
            raise R42Error("plan feature fingerprint mismatch")
        if len(plan["training_trace_ids"]) != 144 or len(plan["test_trace_ids"]) != 84:
            raise R42Error("plan membership cardinality mismatch")
        if set(plan["training_trace_ids"]) & set(plan["test_trace_ids"]):
            raise R42Error("plan train/test trace overlap")
        train_devices = {inputs.device_keys[t] for t in plan["training_trace_ids"]}
        test_devices = {inputs.device_keys[t] for t in plan["test_trace_ids"]}
        if train_devices & test_devices:
            raise R42Error("plan train/test device overlap")
        if plan["model_parameters"] != _model_parameters(plan["model"], plan["model_seed"]):
            raise R42Error("model configuration mismatch")
        if plan["model_configuration_id"] != MODEL_CONFIGURATION_VERSION or plan["preprocessing_configuration_id"] != PREPROCESSING_CONFIGURATION_VERSION:
            raise R42Error("frozen model/preprocessing version mismatch")

    return {
        "schema_version": "r4.2-pre-execution-verification-v1",
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "repository_checkpoint": head,
        "python_executable": sys.executable,
        "software_versions": software_versions(),
        "execution_implementation_path": "execution-implementation/r4_2_execution.executed.py",
        "execution_implementation_sha256": file_sha(script_path),
        "preexecution_fingerprint": PREEXEC_FINGERPRINT,
        "r4_1_aggregate_fingerprint": R41_AGGREGATE,
        "r4_1_artifact_tree_fingerprint": R41_TREE_FINGERPRINT,
        "r4_1_artifact_files_verified": verified_r41,
        "r4_1_freeze_files_verified": len(freeze["authoritative_inventory"]),
        "historical_phase4_fingerprint": HISTORICAL_PHASE4,
        "partition_manifest_sha256": PARTITION_SHA,
        "accepted_partitions_sha256": ACCEPTED_SHA,
        "dataset_file_sha256": file_sha(resolve_scientific_path("dataset-v1.0", root)),
        "feature_contract_counts": expected_counts,
        "feature_contract_fingerprints": {c: contracts["conditions"][c]["ordered_list_fingerprint"] for c in CONDITIONS},
        "f2_features": f2["ordered_feature_names"],
        "planned_new_runs": 120,
        "planned_new_predictions": 10080,
        "immutable_anchor_runs": 120,
        "model_fit_or_prediction_performed_during_gate": False,
    }


def matrix_for(inputs: Any, trace_ids: Sequence[str], names: Sequence[str]) -> MatrixBundle:
    ids = tuple(trace_ids)
    if len(ids) != len(set(ids)):
        raise R42Error("duplicate trace ID in matrix membership")
    values = [[inputs.features_c1[t][name] for name in names] for t in ids]
    if any(value is None for row in values for value in row):
        raise R42Error("missing feature value")
    X = np.asarray(values, dtype=np.float64)
    if X.shape != (len(ids), len(names)) or not np.isfinite(X).all():
        raise R42Error("invalid matrix")
    roles = tuple(inputs.roles[t] for t in ids)
    targets = np.asarray([0 if role in {ORDINARY, TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN} else 1 for role in roles], dtype=np.int64)
    devices = tuple(inputs.device_keys[t] for t in ids)
    return MatrixBundle(ids, tuple(names), X, targets, roles, devices)


def subgroup_confusions(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for role in ROLE_ORDER:
        subset = [r for r in records if r["scientific_role"] == role]
        result[role] = {
            "count": len(subset),
            "tn": sum(r["true_target"] == 0 and r["predicted_class"] == 0 for r in subset),
            "fp": sum(r["true_target"] == 0 and r["predicted_class"] == 1 for r in subset),
            "fn": sum(r["true_target"] == 1 and r["predicted_class"] == 0 for r in subset),
            "tp": sum(r["true_target"] == 1 and r["predicted_class"] == 1 for r in subset),
        }
    return result


def execute_one(root: Path, output: Path, plan: Mapping[str, Any], inputs: Any, names: Sequence[str], preflight_data: Mapping[str, Any]) -> None:
    run_dir = output / "runs" / plan["partition_id"] / plan["condition"] / plan["model"]
    if run_dir.exists():
        required = [run_dir / name for name in ("predictions.jsonl", "metrics.json", "run-provenance.json")]
        if all(path.is_file() for path in required):
            return
        raise R42Error(f"partial run directory requires manual review: {run_dir}")
    train = matrix_for(inputs, plan["training_trace_ids"], names)
    test = matrix_for(inputs, plan["test_trace_ids"], names)
    if set(train.device_keys) & set(test.device_keys):
        raise R42Error("device overlap at execution")
    estimator = fit_estimator(plan["model"], plan["model_seed"], train)
    predicted, probabilities = predict_estimator(estimator, test)
    predictions = []
    for trace_id, truth, role, device, prediction, score in zip(
        test.trace_ids, test.y, test.scientific_roles, test.device_keys, predicted, probabilities,
    ):
        predictions.append({
            "schema_version": PREDICTION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "run_id": plan["planned_run_id"],
            "partition_id": plan["partition_id"],
            "repetition": plan["repetition"],
            "split_seed": plan["split_seed"],
            "condition": plan["condition"],
            "model": plan["model"],
            "model_seed": plan["model_seed"],
            "trace_id": trace_id,
            "split": "test",
            "true_target": int(truth),
            "scientific_role": role,
            "device_key": device,
            "device_key_role": "privileged_evaluation_grouping_metadata_not_feature",
            "predicted_class": int(prediction),
            "predicted_anomaly_probability": float(score),
            "decision_threshold": PREDICTION_THRESHOLD,
            "feature_contract": plan["condition"],
            "feature_contract_fingerprint": plan["ordered_list_fingerprint"],
            "repository_checkpoint": AUTHORIZED_HEAD,
            "preexecution_fingerprint": PREEXEC_FINGERPRINT,
        })
    metrics = reconstruct_metrics(predictions)
    metrics["subgroup_confusion_counts"] = subgroup_confusions(predictions)
    provenance = {
        "schema_version": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_id": plan["planned_run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition_id": plan["partition_id"],
        "repetition": plan["repetition"],
        "split_seed": plan["split_seed"],
        "condition": plan["condition"],
        "model": plan["model"],
        "model_seed": plan["model_seed"],
        "decision_threshold": PREDICTION_THRESHOLD,
        "ordered_feature_names": list(names),
        "feature_count": len(names),
        "feature_contract_fingerprint": plan["ordered_list_fingerprint"],
        "training_trace_ids": list(train.trace_ids),
        "test_trace_ids": list(test.trace_ids),
        "training_device_keys": list(plan["training_device_keys"]),
        "test_device_keys": list(plan["test_device_keys"]),
        "training_role_counts": plan["training_role_counts"],
        "test_role_counts": plan["test_role_counts"],
        "model_configuration_id": plan["model_configuration_id"],
        "model_parameters": plan["model_parameters"],
        "preprocessing_configuration_id": plan["preprocessing_configuration_id"],
        "preprocessing": plan["preprocessing_contract"],
        "fitted_scaler_statistics": fitted_scaler_state(estimator),
        "imputation_performed": False,
        "feature_selection_performed": False,
        "threshold_tuning_performed": False,
        "test_used_for_fit_or_preprocessing": False,
        "frozen_anchor_A_run_id": plan["frozen_anchor_A_run_id"],
        "frozen_anchor_B_run_id": plan["frozen_anchor_B_run_id"],
        "repository_checkpoint": AUTHORIZED_HEAD,
        "execution_implementation_sha256": preflight_data["execution_implementation_sha256"],
        "preexecution_fingerprint": PREEXEC_FINGERPRINT,
        "r4_1_aggregate_fingerprint": R41_AGGREGATE,
        "r4_1_artifact_tree_fingerprint": R41_TREE_FINGERPRINT,
        "historical_phase4_fingerprint": HISTORICAL_PHASE4,
        "partition_manifest_sha256": PARTITION_SHA,
        "accepted_partitions_sha256": ACCEPTED_SHA,
        "dataset_file_sha256": preflight_data["dataset_file_sha256"],
        "software_versions": software_versions(),
    }
    temporary = run_dir.with_name(run_dir.name + ".tmp")
    if temporary.exists():
        raise R42Error(f"stale temporary run directory: {temporary}")
    temporary.mkdir(parents=True)
    write_jsonl(temporary / "predictions.jsonl", predictions)
    write_json(temporary / "metrics.json", metrics)
    write_json(temporary / "run-provenance.json", provenance)
    temporary.rename(run_dir)


def load_new_runs(output: Path, plans: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, metrics_rows, provenance_rows = [], [], []
    for plan in plans:
        run_dir = output / "runs" / plan["partition_id"] / plan["condition"] / plan["model"]
        pred = read_jsonl(run_dir / "predictions.jsonl")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        provenance = json.loads((run_dir / "run-provenance.json").read_text())
        predictions.extend(pred)
        metrics_rows.append({
            "run_id": plan["planned_run_id"], "partition_id": plan["partition_id"],
            "repetition": plan["repetition"], "condition": plan["condition"],
            "model": plan["model"], "metrics": metrics,
        })
        provenance_rows.append(provenance)
    return predictions, metrics_rows, provenance_rows


def load_anchor_metrics(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], Mapping[str, Any]]]:
    r41 = root / "artifacts/r4.1-device-disjoint-evaluation"
    matrix = json.loads((r41 / "run-matrix.json").read_text())
    rows = []
    indexed = {}
    for row in matrix:
        if row["condition"] not in {"C1", "C2"}:
            continue
        condition = "A" if row["condition"] == "C1" else "B"
        metrics_path = r41 / "runs" / row["partition_id"] / row["condition"] / row["model"] / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        item = {
            "run_id": row["run_id"], "partition_id": row["partition_id"],
            "repetition": row["repetition"], "condition": condition,
            "model": row["model"], "metrics": metrics,
            "source": "immutable R4.1 anchor",
        }
        rows.append(item)
        indexed[(row["partition_id"], row["model"], condition)] = metrics
    if len(rows) != 120:
        raise R42Error("anchor metric count mismatch")
    return rows, indexed


def timing_values(metrics: Mapping[str, Any]) -> tuple[int, int, int, float, float]:
    fp = int(metrics["normal_role_diagnostics"][TIMING_HN]["false_positive_count"])
    tp = int(metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["true_positive_count"])
    fn = 12 - tp
    return fp, tp, fn, fp / 12, tp / 12


def direction(value: float) -> str:
    return "negative" if value < 0 else "positive" if value > 0 else "zero"


def joint_category(delta_fpr: float, delta_recall: float) -> str:
    if delta_fpr < 0 and delta_recall >= 0:
        return "fpr_improved_recall_preserved_or_improved"
    if delta_fpr < 0 and delta_recall < 0:
        return "fpr_improved_recall_degraded"
    if delta_fpr == 0 and delta_recall == 0:
        return "no_change"
    if delta_fpr == 0 and delta_recall > 0:
        return "fpr_unchanged_recall_improved"
    if delta_fpr == 0 and delta_recall < 0:
        return "fpr_unchanged_recall_degraded"
    if delta_fpr > 0 and delta_recall >= 0:
        return "fpr_worsened_recall_preserved_or_improved"
    return "both_fpr_and_recall_worsened"


def make_pairs(all_metrics: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    partition_models = sorted({(p, m) for p, m, _ in all_metrics})
    if len(partition_models) != 60:
        raise R42Error("partition/model coverage mismatch")
    for source, target in CONTRASTS:
        contrast = f"{source}_to_{target}"
        for partition, model in partition_models:
            before, after = all_metrics[(partition, model, source)], all_metrics[(partition, model, target)]
            bfp, btp, bfn, bfpr, brecall = timing_values(before)
            afp, atp, afn, afpr, arecall = timing_values(after)
            dfpr, drecall = afpr - bfpr, arecall - brecall
            rows.append({
                "schema_version": "r4.2-paired-comparison-v1",
                "contrast": contrast,
                "source_condition": source,
                "target_condition": target,
                "partition_id": partition,
                "model": model,
                "source_timing_hn_false_positive_count": bfp,
                "target_timing_hn_false_positive_count": afp,
                "delta_timing_hn_false_positive_count": afp - bfp,
                "source_timing_hn_fpr": bfpr,
                "target_timing_hn_fpr": afpr,
                "delta_fpr": dfpr,
                "delta_fpr_direction": direction(dfpr),
                "source_timing_anomaly_true_positive_count": btp,
                "target_timing_anomaly_true_positive_count": atp,
                "delta_timing_anomaly_true_positive_count": atp - btp,
                "source_timing_anomaly_false_negative_count": bfn,
                "target_timing_anomaly_false_negative_count": afn,
                "delta_timing_anomaly_false_negative_count": afn - bfn,
                "source_timing_anomaly_recall": brecall,
                "target_timing_anomaly_recall": arecall,
                "delta_recall": drecall,
                "delta_recall_direction": direction(drecall),
                "joint_outcome_category": joint_category(dfpr, drecall),
            })
    if len(rows) != 240 or Counter(r["contrast"] for r in rows) != {f"{a}_to_{b}": 60 for a, b in CONTRASTS}:
        raise R42Error("paired comparison cardinality mismatch")
    return rows


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 20 or not np.isfinite(array).all():
        raise R42Error("distribution requires exactly 20 finite partition values")
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "values": array.astype(float).tolist(),
        "quartile_method": "numpy linear",
    }


def aggregate(all_rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_key[(row["model"], row["condition"])].append(row["metrics"])
    by_model_condition = {}
    for model in MODELS:
        by_model_condition[model] = {}
        for condition in CONDITIONS:
            items = by_key[(model, condition)]
            if len(items) != 20:
                raise R42Error("aggregate cell requires 20 partitions")
            by_model_condition[model][condition] = {
                "timing_hn_fpr": distribution([timing_values(m)[3] for m in items]),
                "timing_anomaly_recall": distribution([timing_values(m)[4] for m in items]),
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
        paired_summary[model] = {}
        for source, target in CONTRASTS:
            contrast = f"{source}_to_{target}"
            rows = [r for r in pairs if r["model"] == model and r["contrast"] == contrast]
            paired_summary[model][contrast] = {
                "delta_fpr": distribution([float(r["delta_fpr"]) for r in rows]),
                "delta_recall": distribution([float(r["delta_recall"]) for r in rows]),
                "delta_false_positive_count": distribution([float(r["delta_timing_hn_false_positive_count"]) for r in rows]),
                "delta_true_positive_count": distribution([float(r["delta_timing_anomaly_true_positive_count"]) for r in rows]),
                "delta_false_negative_count": distribution([float(r["delta_timing_anomaly_false_negative_count"]) for r in rows]),
                "delta_fpr_direction_counts": dict(sorted(Counter(r["delta_fpr_direction"] for r in rows).items())),
                "delta_recall_direction_counts": dict(sorted(Counter(r["delta_recall_direction"] for r in rows).items())),
                "joint_outcome_counts": dict(sorted(Counter(r["joint_outcome_category"] for r in rows).items())),
            }
    return {
        "schema_version": "r4.2-descriptive-aggregation-v1",
        "partition_count": 20,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "pooled_prediction_independence_claimed": False,
        "hypothesis_tests_performed": False,
        "by_model_condition": by_model_condition,
        "paired_comparisons": paired_summary,
    }


def validate_persisted(root: Path, output: Path, plans: Sequence[Mapping[str, Any]], contracts: Mapping[str, Any], inputs: Any) -> dict[str, Any]:
    checks = []
    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise R42Error(f"independent validation failed: {name}: {evidence}")

    predictions, metric_rows, provenance_rows = load_new_runs(output, plans)
    check("expected_new_runs", len(metric_rows) == len(provenance_rows) == 120, len(metric_rows))
    check("unique_new_run_ids", len({r["run_id"] for r in provenance_rows}) == 120, len({r["run_id"] for r in provenance_rows}))
    check("prediction_total", len(predictions) == 10080, len(predictions))
    pred_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        pred_by_run[record["run_id"]].append(record)
    plan_by_id = {p["planned_run_id"]: p for p in plans}
    metric_by_id = {r["run_id"]: r["metrics"] for r in metric_rows}
    provenance_by_id = {r["run_id"]: r for r in provenance_rows}
    check("run_id_set_exact", set(pred_by_run) == set(plan_by_id) == set(metric_by_id) == set(provenance_by_id), len(pred_by_run))
    names_by_condition = {c: [item["feature_name"] for item in contracts["conditions"][c]["ordered_features"]] for c in ("C", "D")}
    for run_id, plan in plan_by_id.items():
        records = pred_by_run[run_id]
        provenance = provenance_by_id[run_id]
        persisted_metrics = metric_by_id[run_id]
        reconstructed = reconstruct_metrics(records)
        check(f"prediction_cardinality:{run_id}", len(records) == 84 and len({r["trace_id"] for r in records}) == 84, len(records))
        check(f"test_membership:{run_id}", {r["trace_id"] for r in records} == set(plan["test_trace_ids"]), "exact")
        check(f"role_counts:{run_id}", Counter(r["scientific_role"] for r in records) == Counter(plan["test_role_counts"]), dict(Counter(r["scientific_role"] for r in records)))
        check(f"metric_reconstruction:{run_id}", all(persisted_metrics[k] == v for k, v in reconstructed.items()), "exact")
        check(f"subgroup_confusions:{run_id}", persisted_metrics["subgroup_confusion_counts"] == subgroup_confusions(records), "exact")
        check(f"feature_order:{run_id}", provenance["ordered_feature_names"] == names_by_condition[plan["condition"]], plan["condition"])
        check(f"feature_count:{run_id}", provenance["feature_count"] == (89 if plan["condition"] == "C" else 82), provenance["feature_count"])
        check(f"feature_fingerprint:{run_id}", provenance["feature_contract_fingerprint"] == contracts["conditions"][plan["condition"]]["ordered_list_fingerprint"], provenance["feature_contract_fingerprint"])
        check(f"relation_count_retained:{run_id}", "semantic_timing_relation_count" in provenance["ordered_feature_names"], True)
        check(f"training_membership:{run_id}", provenance["training_trace_ids"] == plan["training_trace_ids"], "exact")
        check(f"test_provenance_membership:{run_id}", provenance["test_trace_ids"] == plan["test_trace_ids"], "exact")
        check(f"device_disjoint:{run_id}", not set(provenance["training_device_keys"]) & set(provenance["test_device_keys"]), "zero overlap")
        check(f"model_configuration:{run_id}", provenance["model_parameters"] == plan["model_parameters"] and provenance["model_seed"] == plan["model_seed"], plan["model_seed"])
        check(f"training_only_preprocessing:{run_id}", provenance["test_used_for_fit_or_preprocessing"] is False and provenance["preprocessing"] == plan["preprocessing_contract"], "confirmed")

    r41 = root / "artifacts/r4.1-device-disjoint-evaluation"
    r41_fp = json.loads((r41 / "fingerprints.json").read_text())
    check("r4_1_tree_fingerprint", r41_fp["artifact_tree_fingerprint"] == R41_TREE_FINGERPRINT, r41_fp["artifact_tree_fingerprint"])
    check("r4_1_artifacts_immutable", verify_tree(root, r41_fp, r41) == 551, 551)
    check("historical_phase4_unchanged", json.loads((root / "artifacts/icds-ml-phase4/fingerprints.json").read_text())["artifact_tree_fingerprint"] == "sha256:" + HISTORICAL_PHASE4, HISTORICAL_PHASE4)
    report = {
        "schema_version": "r4.2-independent-validation-v1",
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "run_count": 120,
        "prediction_count": 10080,
        "checks": checks,
    }
    write_json(output / "independent-validation-report.json", report)
    return report


def interval_text(item: Mapping[str, Any]) -> str:
    return f"{item['median']:.3f} [{item['q1']:.3f}–{item['q3']:.3f}; {item['minimum']:.3f}–{item['maximum']:.3f}]"


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# R4.2 representation-sensitivity campaign", "",
        "The frozen campaign completed and passed independent validation: **120 new C/D runs**, **10,080 new predictions**, and **120 immutable A/B anchors referenced without retraining**. Results are descriptive across 20 predefined device-disjoint partitions; no split-level hypothesis tests were performed.", "",
        "## Timing-boundary results", "",
        "Values are median [Q1–Q3; min–max] across the 20 partitions.", "",
        "| Model | Representation | Timing-HN FPR | Timing-anomaly recall |", "|---|---|---|---|",
    ]
    for model in MODELS:
        for condition in CONDITIONS:
            cell = summary["by_model_condition"][model][condition]
            lines.append(f"| {model} | {condition} | {interval_text(cell['timing_hn_fpr'])} | {interval_text(cell['timing_anomaly_recall'])} |")
    lines.extend(["", "## Paired representation changes", "", "Deltas are target minus source within the same frozen partition and model.", "", "| Model | Contrast | ΔFPR | ΔRecall | FPR directions (−/0/+) | Recall directions (−/0/+) | Joint outcomes |", "|---|---|---|---|---|---|---|"])
    for model in MODELS:
        for source, target in CONTRASTS:
            contrast = f"{source}_to_{target}"
            cell = summary["paired_comparisons"][model][contrast]
            fdirs, rdirs = cell["delta_fpr_direction_counts"], cell["delta_recall_direction_counts"]
            lines.append(
                f"| {model} | {source}→{target} | {interval_text(cell['delta_fpr'])} | {interval_text(cell['delta_recall'])} | "
                f"{fdirs.get('negative',0)}/{fdirs.get('zero',0)}/{fdirs.get('positive',0)} | "
                f"{rdirs.get('negative',0)}/{rdirs.get('zero',0)}/{rdirs.get('positive',0)} | "
                f"{json.dumps(cell['joint_outcome_counts'], sort_keys=True)} |"
            )
    lines.extend([
        "", "## Interpretation", "",
        "A→C isolates the empirical sensitivity to removing the explicit signed Timing-margin summary while retaining generic temporal evidence and `semantic_timing_relation_count`. C→D measures the additional empirical change after removing the seven frozen generic-temporal summaries. A→B is historical continuity rather than a Timing-only ablation because B also removes two direct Nonce summaries. C→B is the additional joint removal of those two Nonce summaries.", "",
        "The model-specific direction and magnitude are reported above rather than collapsed into a single claim. Any stability means only that the frozen classifier/partition combination retained similar behavior under the specified representation; any change means sensitivity to that controlled representation change. Neither outcome establishes unique feature-family necessity, sufficiency, formal observability, operational detectability, or causal attribution.", "",
        "These conclusions are restricted to the frozen 336-trace synthetic Timing+Nonce candidate. Device-disjoint evaluation prevents persistent-device overlap within each split but does not establish independence from shared generator logic, scenario templates, population construction, or repeated test-trace reuse across partitions.", "",
        "## Reproducibility", "",
        f"- Repository checkpoint: `{AUTHORIZED_HEAD}`.",
        f"- Pre-execution fingerprint: `{PREEXEC_FINGERPRINT}`.",
        f"- Frozen R4.1 artifact tree: `{R41_TREE_FINGERPRINT}`.",
        "- Executed implementation: `execution-implementation/r4_2_execution.executed.py` (hash recorded in run provenance).",
        "- Command: `PYTHONPATH=$PWD/IoT_auth_simulator:$PWD ../PFE_simulator/.venv/bin/python artifacts/r4.2-representation-sensitivity/execution-implementation/r4_2_execution.executed.py --root . --output artifacts/r4.2-representation-sensitivity`.",
        "- The artifact fingerprint manifest records every output byte except itself.", "",
    ])
    return "\n".join(lines)


def fingerprint_outputs(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "fingerprints.json"):
        files.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha(path),
        })
    core = {"schema_version": "r4.2-artifact-fingerprints-v1", "artifact_count": len(files), "files": files}
    return {**core, "artifact_tree_fingerprint": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    script_path = Path(__file__).resolve()
    gate = preflight(root, output, script_path)
    pre = root / "metadata/r4.2-preexecution"
    contracts = json.loads((pre / "feature-contracts.json").read_text())
    source_manifest = json.loads((pre / "execution-manifest-cd.json").read_text())
    plans = source_manifest["plans"]
    inputs = load_frozen_inputs(root)

    # Persist all frozen execution inputs and expected matrix before fitting.
    output.mkdir(parents=True)
    implementation_dir = output / "execution-implementation"
    implementation_dir.mkdir()
    shutil.copyfile(script_path, implementation_dir / "r4_2_execution.executed.py")
    shutil.copyfile(pre / "feature-contracts.json", output / "feature-contracts.json")
    shutil.copyfile(pre / "f2-generic-temporal-features.json", output / "f2-generic-temporal-features.json")
    shutil.copyfile(pre / "frozen-anchor-runs-ab.json", output / "anchor-references.json")
    write_json(output / "pre-execution-verification.json", gate)
    write_json(output / "run-matrix.json", plans)
    write_json(output / "execution-manifest.json", {
        "schema_version": SCHEMA,
        "status": "EXECUTION_AUTHORIZED_BY_USER_TASK",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_checkpoint": AUTHORIZED_HEAD,
        "preexecution_fingerprint": PREEXEC_FINGERPRINT,
        "source_preexecution_manifest_path": "metadata/r4.2-preexecution/execution-manifest-cd.json",
        "source_preexecution_manifest_sha256": file_sha(pre / "execution-manifest-cd.json"),
        "planned_new_runs": 120,
        "planned_new_predictions": 10080,
        "immutable_anchor_runs": 120,
        "conditions_executed": ["C", "D"],
        "conditions_reused_by_reference": ["A", "B"],
        "models": list(MODELS),
        "partitions": 20,
        "execution_implementation_sha256": gate["execution_implementation_sha256"],
        "command": "PYTHONPATH=$PWD/IoT_auth_simulator:$PWD ../PFE_simulator/.venv/bin/python /tmp/r4_2_execute.py --root . --output artifacts/r4.2-representation-sensitivity",
    })
    audit = json.loads((pre / "durable-artifact-inventory.json").read_text())
    write_json(output / "feature-family-audit-reference.json", {
        "schema_version": "r4.2-feature-family-audit-reference-v1",
        "preexecution_fingerprint": PREEXEC_FINGERPRINT,
        "durable_audit_inventory_fingerprint": audit["inventory_fingerprint"],
        "audit_files": [rec for rec in audit["artifacts"] if "/audit/" in rec["path"]],
    })

    names_by_condition = {
        c: [item["feature_name"] for item in contracts["conditions"][c]["ordered_features"]]
        for c in ("C", "D")
    }
    for index, plan in enumerate(plans, start=1):
        execute_one(root, output, plan, inputs, names_by_condition[plan["condition"]], gate)
        if index % 10 == 0:
            print(f"completed {index}/120 runs", flush=True)

    predictions, new_metrics, new_provenance = load_new_runs(output, plans)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "run-level-metrics.json", new_metrics)
    write_json(output / "run-provenance.json", new_provenance)
    validation = validate_persisted(root, output, plans, contracts, inputs)

    anchor_rows, anchor_index = load_anchor_metrics(root)
    new_index = {(r["partition_id"], r["model"], r["condition"]): r["metrics"] for r in new_metrics}
    all_index = {**anchor_index, **new_index}
    if len(all_index) != 240:
        raise R42Error("combined evaluation index mismatch")
    pairs = make_pairs(all_index)
    write_json(output / "paired-comparisons.json", pairs)
    write_csv(output / "paired-comparisons.csv", [{k: v for k, v in row.items() if k != "schema_version"} for row in pairs])
    combined_rows = anchor_rows + new_metrics
    write_json(output / "combined-evaluation-index.json", combined_rows)
    summary = aggregate(combined_rows, pairs)
    write_json(output / "aggregate-summary.json", summary)
    (output / "scientific-report.md").write_text(report_markdown(summary))

    # Final immutable-input verification after execution.
    r41_root = root / "artifacts/r4.1-device-disjoint-evaluation"
    r41_fp = json.loads((r41_root / "fingerprints.json").read_text())
    if verify_tree(root, r41_fp, r41_root) != 551:
        raise R42Error("post-execution R4.1 immutability failure")
    if validation["status"] != "PASS":
        raise R42Error("independent validation did not pass")
    write_json(output / "fingerprints.json", fingerprint_outputs(output))
    print(json.dumps({
        "status": "PASS",
        "new_runs": 120,
        "new_predictions": 10080,
        "paired_comparisons": 240,
        "artifact_tree_fingerprint": json.loads((output / "fingerprints.json").read_text())["artifact_tree_fingerprint"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
