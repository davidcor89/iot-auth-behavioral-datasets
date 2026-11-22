"""Execute the frozen scientific evaluation campaign exactly once.

This orchestration layer consumes the frozen Phase-1/2/3 artifacts.  It does
not define features, cohorts, preprocessing, models, thresholds, or metrics.
The output root must not exist: scientific evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
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
    file_sha256,
)
from ml.evaluation.grouped_cohorts import GROUPED_COHORT_VERSION
from ml.evaluation.metrics import reconstruct_metrics
from ml.evaluation.pipeline import (
    EXPECTED_CANDIDATE_FINGERPRINT,
    EXPECTED_FEATURE_SPEC_FILE_SHA256,
    EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
    MODEL_CONFIGURATION_VERSION,
    MODELS,
    PIPELINE_SPEC_VERSION,
    PREDICTION_RECORD_VERSION,
    PREDICTION_THRESHOLD,
    PREPROCESSING_CONFIGURATION_VERSION,
    RUN_PROVENANCE_VERSION,
    FrozenInputs,
    assemble_matrix,
    execute_run,
    feature_profile_fingerprint,
    fit_estimator,
    fitted_scaler_state,
    load_frozen_inputs,
    model_seed,
    predict_estimator,
)


PHASE4_VERSION = "icds-frozen-scientific-experiment-v1"
EXPECTED_PIPELINE_SPEC_FILE_SHA256 = (
    "e37b12ef4cb98d214c25b7ed952dc03767da86139d0f92fbce1e5c471fab929b"
)
EXPECTED_PIPELINE_SPEC_FINGERPRINT = (
    "sha256:e130ddb1af2b394409b9e1f058a4874d26a809b400d344585bc8ba0b9df60e74"
)
EXPECTED_GROUPED_COHORT_FINGERPRINT = (
    "sha256:b111c2710848bda6e3e1cfec2b9a018b65e95b7276116490a2e41599446bdf53"
)
EXPECTED_GROUPED_COHORT_FILE_SHA256 = (
    "6adf193dda8a73ea3b50bc2de677a57a914abed594f7c351a199ecc8f4d1a47b"
)
CONDITIONS = ("C0", "C1", "C2")
PRIMARY_REPETITIONS = tuple(range(1, 6))
GROUPED_MODEL_SEED_INDEX = 1
NORMAL_ROLES = (ORDINARY, TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN)


class Phase4ExecutionError(RuntimeError):
    """Frozen scientific execution or evidence validation failed."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_phase4_inputs(root: Path) -> tuple[FrozenInputs, Mapping[str, Any], Mapping[str, Any]]:
    inputs = load_frozen_inputs(root)
    pipeline_spec_path = resolve_scientific_path("eval-v1.0-pipeline-spec", root)
    grouped_path = resolve_scientific_path("eval-v1.0-grouped-split", root)
    if file_sha256(pipeline_spec_path) != EXPECTED_PIPELINE_SPEC_FILE_SHA256:
        raise Phase4ExecutionError("frozen Phase-3 pipeline-spec SHA-256 changed")
    if file_sha256(grouped_path) != EXPECTED_GROUPED_COHORT_FILE_SHA256:
        raise Phase4ExecutionError("frozen grouped-cohort SHA-256 changed")
    pipeline_spec = json.loads(pipeline_spec_path.read_text(encoding="utf-8"))
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    if pipeline_spec["specification_fingerprint"] != EXPECTED_PIPELINE_SPEC_FINGERPRINT:
        raise Phase4ExecutionError("frozen pipeline semantic fingerprint changed")
    if grouped["artifact_fingerprint"] != EXPECTED_GROUPED_COHORT_FINGERPRINT:
        raise Phase4ExecutionError("frozen grouped-cohort semantic fingerprint changed")
    if grouped["schema_version"] != GROUPED_COHORT_VERSION:
        raise Phase4ExecutionError("unexpected grouped-cohort schema")
    return inputs, pipeline_spec, grouped


def primary_run_plan() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (repetition, condition, model)
        for repetition in PRIMARY_REPETITIONS
        for condition in CONDITIONS
        for model in MODELS
    )


def grouped_run_plan() -> tuple[tuple[str, str], ...]:
    return tuple((condition, model) for condition in CONDITIONS for model in MODELS)


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 5 or any(not math.isfinite(float(value)) for value in values):
        raise Phase4ExecutionError("primary summary requires five finite repetition values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "values_by_repetition": array.astype(float).tolist(),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "iqr": float(np.quantile(array, 0.75, method="linear")
                     - np.quantile(array, 0.25, method="linear")),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "summary_policy": "median; linear Q1/Q3 and IQR; min/max; all five raw values",
    }


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    if metric in {"anomaly_f1", "balanced_accuracy", "anomaly_precision", "anomaly_recall"}:
        return float(metrics[metric])
    if metric.startswith("normal_fpr:"):
        role = metric.split(":", 1)[1]
        return float(metrics["normal_role_diagnostics"][role]["false_positive_rate"])
    if metric.startswith("anomaly_recall:"):
        role = metric.split(":", 1)[1]
        return float(metrics["anomaly_role_diagnostics"][role]["recall"])
    raise KeyError(metric)


SUMMARY_METRICS = (
    "anomaly_f1",
    "balanced_accuracy",
    f"normal_fpr:{ORDINARY}",
    f"normal_fpr:{TIMING_HN}",
    f"normal_fpr:{SAME_ATTEMPT_HN}",
    f"normal_fpr:{FRESH_ATTEMPT_HN}",
    f"anomaly_recall:{TIMING_ANOMALY}",
    f"anomaly_recall:{NONCE_ANOMALY}",
)


def summarize_primary(per_run: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (row["repetition"], row["condition"], row["model"]): row["metrics"]
        for row in per_run
    }
    if len(indexed) != 45:
        raise Phase4ExecutionError("primary summary requires exactly 45 unique runs")
    by_model_condition: dict[str, Any] = {}
    for model in MODELS:
        by_model_condition[model] = {}
        for condition in CONDITIONS:
            by_model_condition[model][condition] = {
                metric: distribution_summary([
                    _metric_value(indexed[(repetition, condition, model)], metric)
                    for repetition in PRIMARY_REPETITIONS
                ]) for metric in SUMMARY_METRICS
            }

    comparisons: dict[str, Any] = {}
    for model in MODELS:
        comparisons[model] = {}
        for name, source, target in (("C0_to_C1", "C0", "C1"), ("C1_to_C2", "C1", "C2")):
            comparisons[model][name] = {
                metric: distribution_summary([
                    _metric_value(indexed[(repetition, target, model)], metric)
                    - _metric_value(indexed[(repetition, source, model)], metric)
                    for repetition in PRIMARY_REPETITIONS
                ]) for metric in SUMMARY_METRICS
            }
    return {
        "schema_version": "icds-primary-summary-v1",
        "repetition_count": 5,
        "quartile_method": "numpy linear quantiles",
        "by_model_condition": by_model_condition,
        "paired_differences_target_minus_source": comparisons,
    }


def validate_prediction_campaign(
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_run_count: int,
    expected_predictions_per_run: int,
) -> None:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in predictions:
        key = (
            record.get("scientific_scope", "primary"), record.get("repetition"),
            record["condition"], record["model"],
        )
        grouped[key].append(record)
    if len(grouped) != expected_run_count:
        raise Phase4ExecutionError("scientific prediction run count changed")
    for key, records in grouped.items():
        trace_ids = [record["trace_id"] for record in records]
        if len(records) != expected_predictions_per_run or len(trace_ids) != len(set(trace_ids)):
            raise Phase4ExecutionError(f"prediction cardinality/uniqueness failure for {key}")
        if any(record["split"] != "test" for record in records):
            raise Phase4ExecutionError(f"non-test prediction in scientific evidence for {key}")
        reconstructed = reconstruct_metrics(records)
        if reconstructed["prediction_count"] != expected_predictions_per_run:
            raise Phase4ExecutionError(f"metric reconstruction failure for {key}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _flatten_summary_rows(summary: Mapping[str, Any], metrics: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for condition in CONDITIONS:
            row: dict[str, Any] = {"model": model, "condition": condition}
            for metric in metrics:
                distribution = summary["by_model_condition"][model][condition][metric]
                for statistic in ("median", "q1", "q3", "iqr", "minimum", "maximum"):
                    row[f"{metric}__{statistic}"] = distribution[statistic]
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise Phase4ExecutionError("cannot write an empty scientific table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _primary_tables(
    per_run: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output: Path,
) -> None:
    table_a_metrics = (
        "anomaly_f1", "balanced_accuracy",
        f"anomaly_recall:{TIMING_ANOMALY}", f"anomaly_recall:{NONCE_ANOMALY}",
    )
    table_b_metrics = tuple(f"normal_fpr:{role}" for role in NORMAL_ROLES)
    table_a = _flatten_summary_rows(summary, table_a_metrics)
    table_b = _flatten_summary_rows(summary, table_b_metrics)
    _write_json(output / "table-a-experimental-outcomes.json", table_a)
    _write_csv(output / "table-a-experimental-outcomes.csv", table_a)
    _write_json(output / "table-b-normal-subgroup-fpr.json", table_b)
    _write_csv(output / "table-b-normal-subgroup-fpr.csv", table_b)

    figure_rows: list[dict[str, Any]] = []
    for row in per_run:
        if row["condition"] not in {"C0", "C1"}:
            continue
        for role in NORMAL_ROLES:
            diagnostic = row["metrics"]["normal_role_diagnostics"][role]
            figure_rows.append({
                "repetition": row["repetition"],
                "model": row["model"],
                "condition": row["condition"],
                "normal_role": role,
                "false_positive_count": diagnostic["false_positive_count"],
                "normal_count": diagnostic["count"],
                "false_positive_rate": diagnostic["false_positive_rate"],
            })
    _write_json(output / "figure-data-c0-c1-normal-fpr.json", figure_rows)
    _write_csv(output / "figure-data-c0-c1-normal-fpr.csv", figure_rows)


def execute_primary(inputs: FrozenInputs, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    run_root = output_root / "runs"
    for repetition, condition, model in primary_run_plan():
        result = execute_run(
            repetition, condition, model,
            inputs=inputs,
            output_root=run_root,
            execution_mode="SCIENTIFIC_EXECUTION",
            scientific_execution_authorized=True,
        )
        predictions.extend(result.predictions)
        provenance.append(result.provenance)
        per_run.append({
            "repetition": repetition,
            "condition": condition,
            "model": model,
            "run_id": result.provenance["run_id"],
            "metrics": result.metrics,
        })
    validate_prediction_campaign(
        predictions, expected_run_count=45, expected_predictions_per_run=84,
    )
    _write_jsonl(output_root / "predictions.jsonl", predictions)
    _write_json(output_root / "run-provenance.json", provenance)
    _write_json(output_root / "per-run-metrics.json", per_run)
    return predictions, provenance, per_run


def _grouped_model_parameters(model: str, seed: int, estimator: Any) -> dict[str, Any]:
    if model == "LR":
        params = estimator.named_steps["model"].get_params()
    else:
        params = estimator.get_params()
    names = {
        "LR": ("penalty", "C", "solver", "max_iter", "class_weight", "random_state"),
        "DT": ("criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf",
               "max_features", "class_weight", "random_state"),
        "RF": ("n_estimators", "criterion", "max_depth", "min_samples_split",
               "min_samples_leaf", "max_features", "bootstrap", "class_weight", "n_jobs",
               "random_state"),
    }[model]
    return {name: params[name] for name in names}


def _execute_grouped_run(
    inputs: FrozenInputs,
    grouped: Mapping[str, Any],
    condition: str,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    membership = grouped["C1"] if condition == "C2" else grouped[condition]
    profile = "C2" if condition == "C2" else "C1"
    train = assemble_matrix(inputs, membership["training_trace_ids"], profile)
    test = assemble_matrix(inputs, grouped["common_test_trace_ids"], profile)
    if set(train.trace_ids) & set(test.trace_ids):
        raise Phase4ExecutionError("grouped training and test trace membership overlaps")
    train_devices, test_devices = set(train.device_keys), set(test.device_keys)
    if train_devices & test_devices or len(train_devices) != 48 or len(test_devices) != 16:
        raise Phase4ExecutionError("grouped device-disjoint contract failed")

    seed = model_seed(GROUPED_MODEL_SEED_INDEX, model)
    estimator = fit_estimator(model, seed, train)
    predicted, probabilities = predict_estimator(estimator, test)
    run_id = _canonical_hash({
        "phase4": PHASE4_VERSION,
        "scope": "device_grouped_sensitivity",
        "grouped_cohort": grouped["artifact_fingerprint"],
        "condition": condition,
        "model": model,
        "model_seed": seed,
        "feature_profile": profile,
    })
    commit = _git_commit(inputs.root)
    predictions = [{
        "schema_version": PREDICTION_RECORD_VERSION,
        "experiment_version": PIPELINE_SPEC_VERSION,
        "phase4_version": PHASE4_VERSION,
        "scientific_scope": "device_grouped_sensitivity",
        "execution_mode": "SCIENTIFIC_EXECUTION",
        "run_id": run_id,
        "dataset_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
        "feature_contract_file_sha256": EXPECTED_FEATURE_SPEC_FILE_SHA256,
        "phase2_manifest_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        "grouped_cohort_fingerprint": grouped["artifact_fingerprint"],
        "repetition": "grouped-seed-43001",
        "condition": condition,
        "model": model,
        "model_seed": seed,
        "trace_id": trace_id,
        "split": "test",
        "true_target": int(truth),
        "scientific_role": role,
        "device_key": device,
        "device_key_role": "privileged_evaluation_metadata_not_feature",
        "predicted_class": int(prediction),
        "predicted_anomaly_probability": float(probability),
        "decision_threshold": PREDICTION_THRESHOLD,
        "feature_profile": profile,
        "feature_profile_fingerprint": feature_profile_fingerprint(profile),
        "model_configuration_id": MODEL_CONFIGURATION_VERSION,
        "preprocessing_configuration_id": PREPROCESSING_CONFIGURATION_VERSION,
        "repository_commit": commit,
    } for trace_id, truth, role, device, prediction, probability in zip(
        test.trace_ids, test.y, test.scientific_roles, test.device_keys,
        predicted, probabilities,
    )]
    metrics = reconstruct_metrics(predictions)
    provenance = {
        "schema_version": RUN_PROVENANCE_VERSION,
        "phase4_version": PHASE4_VERSION,
        "scientific_scope": "device_grouped_sensitivity",
        "execution_mode": "SCIENTIFIC_EXECUTION",
        "scientific_result": True,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "model": model,
        "model_seed": seed,
        "grouped_model_seed_policy": (
            "single grouped split uses Phase-3 repetition-index-1 family seed"
        ),
        "decision_threshold": PREDICTION_THRESHOLD,
        "training_trace_ids": list(train.trace_ids),
        "test_trace_ids": list(test.trace_ids),
        "training_device_ids": sorted(train_devices),
        "test_device_ids": sorted(test_devices),
        "device_overlap_count": 0,
        "feature_profile": profile,
        "feature_profile_fingerprint": feature_profile_fingerprint(profile),
        "feature_names_in_order": list(train.feature_names),
        "training_target_counts": {
            "normal": int((train.y == 0).sum()), "anomaly": int((train.y == 1).sum()),
        },
        "training_scientific_role_counts": dict(sorted(Counter(train.scientific_roles).items())),
        "test_target_counts": {
            "normal": int((test.y == 0).sum()), "anomaly": int((test.y == 1).sum()),
        },
        "test_scientific_role_counts": dict(sorted(Counter(test.scientific_roles).items())),
        "model_configuration_id": MODEL_CONFIGURATION_VERSION,
        "model_parameters": _grouped_model_parameters(model, seed, estimator),
        "preprocessing_configuration_id": PREPROCESSING_CONFIGURATION_VERSION,
        "fitted_scaler_statistics": fitted_scaler_state(estimator),
        "upstream": {
            "candidate_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
            "phase2_manifest_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
            "pipeline_spec_fingerprint": EXPECTED_PIPELINE_SPEC_FINGERPRINT,
            "grouped_cohort_fingerprint": grouped["artifact_fingerprint"],
        },
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "repository_commit": commit,
        "test_used_for_fit_or_preprocessing": False,
        "feature_selection_performed": False,
        "imputation_performed": False,
        "threshold_tuning_performed": False,
    }
    return predictions, provenance, metrics


def execute_grouped(
    inputs: FrozenInputs, grouped: Mapping[str, Any], output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for condition, model in grouped_run_plan():
        run_predictions, run_provenance, metrics = _execute_grouped_run(
            inputs, grouped, condition, model,
        )
        predictions.extend(run_predictions)
        provenance.append(run_provenance)
        per_run.append({
            "scope": "device_grouped_sensitivity",
            "condition": condition,
            "model": model,
            "run_id": run_provenance["run_id"],
            "metrics": metrics,
        })
        run_dir = output_root / "runs" / condition / model
        _write_jsonl(run_dir / "predictions.jsonl", run_predictions)
        _write_json(run_dir / "run-provenance.json", run_provenance)
        _write_json(run_dir / "metrics.json", metrics)
    validate_prediction_campaign(
        predictions, expected_run_count=9, expected_predictions_per_run=91,
    )
    _write_jsonl(output_root / "predictions.jsonl", predictions)
    _write_json(output_root / "run-provenance.json", provenance)
    _write_json(output_root / "per-run-metrics.json", per_run)
    table_rows: list[dict[str, Any]] = []
    for row in per_run:
        metrics = row["metrics"]
        table_row: dict[str, Any] = {
            "model": row["model"], "condition": row["condition"],
            "anomaly_f1": metrics["anomaly_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "timing_anomaly_recall": metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["recall"],
            "timing_anomaly_tp": metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["true_positive_count"],
            "timing_anomaly_count": metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["count"],
            "nonce_anomaly_recall": metrics["anomaly_role_diagnostics"][NONCE_ANOMALY]["recall"],
            "nonce_anomaly_tp": metrics["anomaly_role_diagnostics"][NONCE_ANOMALY]["true_positive_count"],
            "nonce_anomaly_count": metrics["anomaly_role_diagnostics"][NONCE_ANOMALY]["count"],
        }
        for role in NORMAL_ROLES:
            diagnostic = metrics["normal_role_diagnostics"][role]
            prefix = {
                ORDINARY: "ordinary", TIMING_HN: "timing_hn",
                SAME_ATTEMPT_HN: "same_attempt_nonce_hn",
                FRESH_ATTEMPT_HN: "fresh_attempt_nonce_hn",
            }[role]
            table_row[f"{prefix}_fp"] = diagnostic["false_positive_count"]
            table_row[f"{prefix}_count"] = diagnostic["count"]
            table_row[f"{prefix}_fpr"] = diagnostic["false_positive_rate"]
        table_rows.append(table_row)
    _write_json(output_root / "grouped-sensitivity-table.json", table_rows)
    _write_csv(output_root / "grouped-sensitivity-table.csv", table_rows)
    return predictions, provenance, per_run


def _artifact_hashes(output_root: Path) -> dict[str, Any]:
    paths = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "fingerprints.json"
    )
    records = [{
        "path": path.relative_to(output_root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    } for path in paths]
    return {
        "schema_version": "icds-phase4-artifact-fingerprints-v1",
        "phase4_version": PHASE4_VERSION,
        "files": records,
        "artifact_tree_fingerprint": _canonical_hash(records),
    }


def execute_phase4(root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise Phase4ExecutionError(f"refusing to overwrite Phase-4 output: {output_root}")
    inputs, pipeline_spec, grouped = verify_phase4_inputs(root)
    output_root.mkdir(parents=True)
    execution_manifest = {
        "schema_version": "icds-phase4-execution-manifest-v1",
        "phase4_version": PHASE4_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(root),
        "primary_run_plan": [
            {"repetition": repetition, "condition": condition, "model": model}
            for repetition, condition, model in primary_run_plan()
        ],
        "grouped_run_plan": [
            {"condition": condition, "model": model}
            for condition, model in grouped_run_plan()
        ],
        "grouped_C2_basis": "C1 membership with frozen C2 feature profile",
        "grouped_model_seed_index": GROUPED_MODEL_SEED_INDEX,
        "upstream": {
            "candidate_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
            "phase2_manifest_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
            "pipeline_spec_fingerprint": pipeline_spec["specification_fingerprint"],
            "grouped_cohort_fingerprint": grouped["artifact_fingerprint"],
        },
        "post_result_adaptation_permitted": False,
    }
    _write_json(output_root / "execution-manifest.json", execution_manifest)

    primary_predictions, primary_provenance, primary_per_run = execute_primary(
        inputs, output_root / "primary",
    )
    primary_summary = summarize_primary(primary_per_run)
    _write_json(output_root / "primary/aggregate-summary.json", primary_summary)
    _primary_tables(primary_per_run, primary_summary, output_root / "primary")

    grouped_predictions, grouped_provenance, grouped_per_run = execute_grouped(
        inputs, grouped, output_root / "grouped",
    )
    completed = {
        **execution_manifest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary": {
            "run_count": len(primary_provenance),
            "prediction_count": len(primary_predictions),
            "per_run_metric_count": len(primary_per_run),
        },
        "grouped": {
            "run_count": len(grouped_provenance),
            "prediction_count": len(grouped_predictions),
            "per_run_metric_count": len(grouped_per_run),
        },
    }
    _write_json(output_root / "execution-manifest.json", completed)
    fingerprints = _artifact_hashes(output_root)
    _write_json(output_root / "fingerprints.json", fingerprints)
    return {"execution": completed, "fingerprints": fingerprints}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-scientific-execution", action="store_true")
    args = parser.parse_args()
    if not args.authorize_scientific_execution:
        parser.error("explicit --authorize-scientific-execution is required")
    result = execute_phase4(args.repository_root.resolve(), args.output.resolve())
    print(json.dumps({
        "primary_runs": result["execution"]["primary"]["run_count"],
        "primary_predictions": result["execution"]["primary"]["prediction_count"],
        "grouped_runs": result["execution"]["grouped"]["run_count"],
        "grouped_predictions": result["execution"]["grouped"]["prediction_count"],
        "artifact_tree_fingerprint": result["fingerprints"]["artifact_tree_fingerprint"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
