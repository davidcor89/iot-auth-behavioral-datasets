"""Leakage-safe execution machinery for the frozen baseline experiment.

Importing this module performs no extraction, fitting, preprocessing, or model
execution.  Every learned object is created afresh inside one requested run and
is fitted only with that run's persisted training membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.artifact_paths import resolve_scientific_path

from ml.evaluation.experiment_manifest import (
    ANOMALY_ROLES,
    EXPECTED_C1_FINGERPRINT,
    EXPECTED_C2_FINGERPRINT,
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_CANDIDATE_FINGERPRINT,
    EXPECTED_FEATURE_SPEC_FILE_SHA256,
    FRESH_ATTEMPT_HN,
    HN_ROLES,
    NONCE_ANOMALY,
    ORDINARY,
    ROLE_TO_BINARY_TARGET,
    SAME_ATTEMPT_HN,
    TIMING_ANOMALY,
    TIMING_HN,
    candidate_control_records,
    file_sha256,
    validate_manifest,
)
from ml.evaluation.metrics import METRIC_CONTRACT_VERSION, reconstruct_metrics
from ml.evaluation.trace_features import (
    C1_FEATURE_NAMES,
    C2_FEATURE_NAMES,
    FEATURE_DEFINITIONS,
    FEATURE_SPEC_VERSION,
    extract_candidate_feature_rows,
)


PIPELINE_SPEC_VERSION = "icds-leakage-safe-baseline-pipeline-v1"
RUN_PROVENANCE_VERSION = "icds-ml-run-provenance-v1"
PREDICTION_RECORD_VERSION = "icds-test-prediction-record-v1"
MODEL_CONFIGURATION_VERSION = "icds-fixed-baseline-models-v1"
PREPROCESSING_CONFIGURATION_VERSION = "icds-train-only-preprocessing-v1"
EXPECTED_PHASE2_MANIFEST_FINGERPRINT = (
    "sha256:8e11d3a31dec7d9ad57fe59a6a658f58bfb7919e7bd744fb1600b594bcf0b8ed"
)
EXPECTED_PHASE2_FILE_SHA256 = "f7b851266e720a25db45e5430cf7b09aa501fd84fb7a057aad2779d6d1d8aa64"
PREDICTION_THRESHOLD = 0.5
CONDITIONS = ("C0", "C1", "C2")
MODELS = ("LR", "DT", "RF")
MODEL_SEED_BASE = {"LR": 51000, "DT": 52000, "RF": 53000}

LR_PARAMETERS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "class_weight": None,
}
DT_PARAMETERS = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "class_weight": None,
}
RF_PARAMETERS = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": 1,
}


class PipelineContractError(ValueError):
    """A frozen input or requested execution violates the Phase-3 contract."""


@dataclass(frozen=True)
class MatrixBundle:
    trace_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    scientific_roles: tuple[str, ...]
    device_keys: tuple[str, ...]


@dataclass(frozen=True)
class FrozenInputs:
    root: Path
    candidate: Mapping[str, Any]
    experiment_manifest: Mapping[str, Any]
    phase1_audit: Mapping[str, Any]
    features_c1: Mapping[str, Mapping[str, int | float | None]]
    features_c2: Mapping[str, Mapping[str, int | float | None]]
    roles: Mapping[str, str]
    device_keys: Mapping[str, str]


@dataclass(frozen=True)
class RunOutput:
    predictions: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    metrics: dict[str, Any]
    estimator: BaseEstimator


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_seed(repetition: int, model: str) -> int:
    if repetition not in range(1, 6):
        raise PipelineContractError("repetition must be in 1..5")
    if model not in MODELS:
        raise PipelineContractError(f"model must be one of {MODELS}")
    return MODEL_SEED_BASE[model] + repetition


def feature_profile_for_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise PipelineContractError(f"condition must be one of {CONDITIONS}")
    return "C2" if condition == "C2" else "C1"


def feature_names_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "C1":
        return C1_FEATURE_NAMES
    if profile == "C2":
        return C2_FEATURE_NAMES
    raise PipelineContractError("feature profile must be C1 or C2")


def feature_profile_fingerprint(profile: str) -> str:
    return _canonical_hash({
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "profile": profile,
        "ordered_feature_names": list(feature_names_for_profile(profile)),
    })


def _feature_rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_hash(rows)


def load_frozen_inputs(root: Path | None = None) -> FrozenInputs:
    root = root or repository_root()
    candidate_path = resolve_scientific_path("dataset-v1.0", root)
    feature_spec_path = resolve_scientific_path("eval-v1.0-feature-spec", root)
    phase1_audit_path = resolve_scientific_path("eval-v1.0-phase1-audit", root)
    phase2_path = resolve_scientific_path("eval-v1.0-manifest", root)

    if file_sha256(candidate_path) != EXPECTED_CANDIDATE_FILE_SHA256:
        raise PipelineContractError("frozen candidate file SHA-256 changed")
    if file_sha256(feature_spec_path) != EXPECTED_FEATURE_SPEC_FILE_SHA256:
        raise PipelineContractError("frozen Phase-1 feature-spec SHA-256 changed")
    if file_sha256(phase2_path) != EXPECTED_PHASE2_FILE_SHA256:
        raise PipelineContractError("frozen Phase-2 serialized manifest SHA-256 changed")

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    phase1_audit = json.loads(phase1_audit_path.read_text(encoding="utf-8"))
    experiment_manifest = json.loads(phase2_path.read_text(encoding="utf-8"))
    if candidate.get("candidate_fingerprint") != EXPECTED_CANDIDATE_FINGERPRINT:
        raise PipelineContractError("frozen candidate semantic fingerprint changed")
    if experiment_manifest.get("manifest_fingerprint") != EXPECTED_PHASE2_MANIFEST_FINGERPRINT:
        raise PipelineContractError("frozen Phase-2 semantic manifest fingerprint changed")
    validate_manifest(experiment_manifest)

    rows_c1 = extract_candidate_feature_rows(candidate, profile="C1")
    rows_c2 = extract_candidate_feature_rows(candidate, profile="C2")
    if _feature_rows_fingerprint(rows_c1) != EXPECTED_C1_FINGERPRINT:
        raise PipelineContractError("frozen Phase-1 C1 feature rows changed")
    if _feature_rows_fingerprint(rows_c2) != EXPECTED_C2_FINGERPRINT:
        raise PipelineContractError("frozen Phase-1 C2 feature rows changed")
    if phase1_audit["missing_or_non_applicable_counts_c1"]:
        raise PipelineContractError("unexpected missing feature values require researcher review")

    controls = candidate_control_records(candidate)
    roles = {record.sample_id: record.role for record in controls}
    device_keys = {record.sample_id: record.device_key for record in controls}
    if roles != experiment_manifest["trace_role_index"]:
        raise PipelineContractError("candidate roles and Phase-2 role index differ")
    return FrozenInputs(
        root=root,
        candidate=candidate,
        experiment_manifest=experiment_manifest,
        phase1_audit=phase1_audit,
        features_c1={row["sample_reference"]: row["features"] for row in rows_c1},
        features_c2={row["sample_reference"]: row["features"] for row in rows_c2},
        roles=roles,
        device_keys=device_keys,
    )


def feature_type_inventory(inputs: FrozenInputs) -> dict[str, Any]:
    declarations = Counter(definition.output_type for definition in FEATURE_DEFINITIONS)
    all_values = [
        value for features in inputs.features_c1.values() for value in features.values()
    ]
    missing = sum(value is None for value in all_values)
    non_numeric = sum(
        value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
        for value in all_values
    )
    non_finite = sum(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and not math.isfinite(float(value))
        for value in all_values
    )
    return {
        "feature_count_C1": len(C1_FEATURE_NAMES),
        "feature_count_C2": len(C2_FEATURE_NAMES),
        "declared_output_types_C1": dict(sorted(declarations.items())),
        "categorical_features": 0,
        "binary_typed_features": 0,
        "candidate_missing_values": missing,
        "candidate_non_numeric_values": non_numeric,
        "candidate_non_finite_values": non_finite,
        "imputation_required": False,
        "categorical_encoding_required": False,
    }


def _repetition(manifest: Mapping[str, Any], repetition: int) -> Mapping[str, Any]:
    matches = [item for item in manifest["repetitions"] if item["repetition"] == repetition]
    if len(matches) != 1:
        raise PipelineContractError("requested repetition is absent or duplicated")
    return matches[0]


def validate_comparison_contract(manifest: Mapping[str, Any], repetition: int) -> None:
    item = _repetition(manifest, repetition)
    c0, c1, c2 = item["C0"], item["C1"], item["C2"]
    c0_ids, c1_ids = set(c0["training_trace_ids"]), set(c1["training_trace_ids"])
    role_index = manifest["trace_role_index"]
    c0_anomalies = {sample_id for sample_id in c0_ids if role_index[sample_id] in ANOMALY_ROLES}
    c1_anomalies = {sample_id for sample_id in c1_ids if role_index[sample_id] in ANOMALY_ROLES}
    if len(c0_ids) != len(c1_ids) or len(c0_ids) != 144:
        raise PipelineContractError("C0/C1 training sizes differ")
    if c0["training_counts"]["binary"] != c1["training_counts"]["binary"]:
        raise PipelineContractError("C0/C1 binary prevalence differs")
    if c0_anomalies != c1_anomalies:
        raise PipelineContractError("C0/C1 anomaly membership differs")
    if Counter(role_index[sample_id] for sample_id in c1_ids if role_index[sample_id] in HN_ROLES) != Counter({
        TIMING_HN: 12, SAME_ATTEMPT_HN: 12, FRESH_ATTEMPT_HN: 12,
    }):
        raise PipelineContractError("C1 hard-negative composition changed")
    if c0["common_test_trace_ids"] != c1["common_test_trace_ids"]:
        raise PipelineContractError("C0/C1 common test membership differs")
    if c0["feature_profile"] != c1["feature_profile"] or c0["feature_profile"] != "C1":
        raise PipelineContractError("C0/C1 feature profile differs")
    if c2["training_trace_ids"] != c1["training_trace_ids"]:
        raise PipelineContractError("C1/C2 training membership differs")
    if c2["common_test_trace_ids"] != c1["common_test_trace_ids"]:
        raise PipelineContractError("C1/C2 test membership differs")
    if set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES) != {
        "semantic_timing_min_signed_causal_margin",
        "semantic_nonce_cross_attempt_value_count",
        "semantic_nonce_max_attempt_sharing_count",
    } or not set(C2_FEATURE_NAMES) < set(C1_FEATURE_NAMES):
        raise PipelineContractError("C1/C2 feature-profile difference changed")


def assemble_matrix(
    inputs: FrozenInputs, trace_ids: Sequence[str], profile: str,
) -> MatrixBundle:
    names = feature_names_for_profile(profile)
    feature_index = inputs.features_c1 if profile == "C1" else inputs.features_c2
    ordered_ids = tuple(str(sample_id) for sample_id in trace_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise PipelineContractError("matrix membership contains duplicate trace IDs")
    try:
        values = [[feature_index[sample_id][name] for name in names] for sample_id in ordered_ids]
        roles = tuple(inputs.roles[sample_id] for sample_id in ordered_ids)
        devices = tuple(inputs.device_keys[sample_id] for sample_id in ordered_ids)
    except KeyError as error:
        raise PipelineContractError(f"unknown trace or feature in matrix: {error}") from error
    if any(value is None for row in values for value in row):
        raise PipelineContractError("missing feature values require researcher review")
    X = np.asarray(values, dtype=np.float64)
    if X.shape != (len(ordered_ids), len(names)) or not np.isfinite(X).all():
        raise PipelineContractError("feature matrix shape or finiteness check failed")
    y = np.asarray([ROLE_TO_BINARY_TARGET[role] for role in roles], dtype=np.int64)
    return MatrixBundle(ordered_ids, names, X, y, roles, devices)


def build_estimator(model: str, seed: int) -> BaseEstimator:
    """Return a fresh, unfitted estimator for exactly one run."""
    if model == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=seed, **LR_PARAMETERS)),
        ])
    if model == "DT":
        return DecisionTreeClassifier(random_state=seed, **DT_PARAMETERS)
    if model == "RF":
        return RandomForestClassifier(random_state=seed, **RF_PARAMETERS)
    raise PipelineContractError(f"model must be one of {MODELS}")


def fit_estimator(model: str, seed: int, training: MatrixBundle) -> BaseEstimator:
    estimator = build_estimator(model, seed)
    estimator.fit(training.X, training.y)
    return estimator


def predict_estimator(estimator: BaseEstimator, test: MatrixBundle) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(estimator.predict_proba(test.X)[:, 1], dtype=np.float64)
    if probabilities.shape != (len(test.trace_ids),) or not np.isfinite(probabilities).all():
        raise PipelineContractError("model emitted invalid test probabilities")
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(np.int64)
    return predictions, probabilities


def fitted_scaler_state(estimator: BaseEstimator) -> dict[str, list[float]] | None:
    if not isinstance(estimator, Pipeline):
        return None
    scaler = estimator.named_steps["scaler"]
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "variance": scaler.var_.astype(float).tolist(),
        "samples_seen": np.atleast_1d(scaler.n_samples_seen_).astype(int).tolist(),
    }


def _model_parameters(model: str, seed: int) -> dict[str, Any]:
    base = LR_PARAMETERS if model == "LR" else DT_PARAMETERS if model == "DT" else RF_PARAMETERS
    return {**base, "random_state": seed}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }
    try:
        import pandas as pd
        versions["pandas"] = pd.__version__
    except ImportError:
        versions["pandas"] = "not-installed"
    return versions


def execute_run(
    repetition: int,
    condition: str,
    model: str,
    *,
    inputs: FrozenInputs | None = None,
    output_root: Path | None = None,
    execution_mode: str = "ENGINEERING_SMOKE_TEST",
    scientific_execution_authorized: bool = False,
) -> RunOutput:
    """Fit one frozen run; full scientific execution requires explicit authorization."""
    if execution_mode not in {"ENGINEERING_SMOKE_TEST", "SCIENTIFIC_EXECUTION"}:
        raise PipelineContractError("unknown execution mode")
    if execution_mode == "SCIENTIFIC_EXECUTION" and not scientific_execution_authorized:
        raise PipelineContractError("scientific execution requires explicit authorization")
    if condition not in CONDITIONS or model not in MODELS:
        raise PipelineContractError("unknown condition or model")
    inputs = inputs or load_frozen_inputs()
    validate_comparison_contract(inputs.experiment_manifest, repetition)
    item = _repetition(inputs.experiment_manifest, repetition)
    condition_record = item[condition]
    profile = feature_profile_for_condition(condition)
    train = assemble_matrix(inputs, condition_record["training_trace_ids"], profile)
    test = assemble_matrix(inputs, condition_record["common_test_trace_ids"], profile)
    if set(train.trace_ids) & set(test.trace_ids):
        raise PipelineContractError("test trace appears in training membership")

    seed = model_seed(repetition, model)
    estimator = fit_estimator(model, seed, train)
    predicted, probabilities = predict_estimator(estimator, test)
    run_id = _canonical_hash({
        "pipeline": PIPELINE_SPEC_VERSION,
        "manifest": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        "repetition": repetition,
        "condition": condition,
        "model": model,
        "model_seed": seed,
    })
    repository_commit = _git_commit(inputs.root)
    predictions = tuple({
        "schema_version": PREDICTION_RECORD_VERSION,
        "experiment_version": PIPELINE_SPEC_VERSION,
        "execution_mode": execution_mode,
        "run_id": run_id,
        "dataset_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
        "feature_contract_file_sha256": EXPECTED_FEATURE_SPEC_FILE_SHA256,
        "phase2_manifest_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        "repetition": repetition,
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
        "repository_commit": repository_commit,
    } for trace_id, truth, role, device, prediction, probability in zip(
        test.trace_ids, test.y, test.scientific_roles, test.device_keys,
        predicted, probabilities,
    ))
    metrics = reconstruct_metrics(predictions)
    provenance = {
        "schema_version": RUN_PROVENANCE_VERSION,
        "execution_mode": execution_mode,
        "scientific_result": execution_mode == "SCIENTIFIC_EXECUTION",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repetition": repetition,
        "condition": condition,
        "model": model,
        "model_seed": seed,
        "decision_threshold": PREDICTION_THRESHOLD,
        "training_trace_ids": list(train.trace_ids),
        "test_trace_ids": list(test.trace_ids),
        "feature_profile": profile,
        "feature_profile_fingerprint": feature_profile_fingerprint(profile),
        "feature_names_in_order": list(train.feature_names),
        "training_target_counts": {
            "normal": int((train.y == 0).sum()),
            "anomaly": int((train.y == 1).sum()),
        },
        "training_scientific_role_counts": dict(sorted(Counter(train.scientific_roles).items())),
        "test_target_counts": {
            "normal": int((test.y == 0).sum()),
            "anomaly": int((test.y == 1).sum()),
        },
        "model_configuration_id": MODEL_CONFIGURATION_VERSION,
        "model_parameters": _model_parameters(model, seed),
        "preprocessing_configuration_id": PREPROCESSING_CONFIGURATION_VERSION,
        "preprocessing": (
            {"steps": ["StandardScaler", "LogisticRegression"], "fit_scope": "training_only"}
            if model == "LR" else {"steps": [model], "fit_scope": "training_only", "scaling": False}
        ),
        "fitted_scaler_statistics": fitted_scaler_state(estimator),
        "upstream": {
            "candidate_file_sha256": EXPECTED_CANDIDATE_FILE_SHA256,
            "candidate_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
            "feature_spec_file_sha256": EXPECTED_FEATURE_SPEC_FILE_SHA256,
            "phase1_feature_rows_fingerprint": (
                EXPECTED_C1_FINGERPRINT if profile == "C1" else EXPECTED_C2_FINGERPRINT
            ),
            "phase2_file_sha256": EXPECTED_PHASE2_FILE_SHA256,
            "phase2_manifest_fingerprint": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        },
        "software_versions": _software_versions(),
        "repository_commit": repository_commit,
        "test_used_for_fit_or_preprocessing": False,
        "feature_selection_performed": False,
        "imputation_performed": False,
        "threshold_tuning_performed": False,
    }
    output = RunOutput(predictions, provenance, metrics, estimator)
    if output_root is not None:
        persist_run(output_root, output)
    return output


def persist_run(output_root: Path, output: RunOutput) -> Path:
    provenance = output.provenance
    run_dir = (
        output_root / PIPELINE_SPEC_VERSION
        / f"repetition-{provenance['repetition']}"
        / provenance["condition"] / provenance["model"]
    )
    if run_dir.exists():
        raise PipelineContractError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                for record in output.predictions),
        encoding="utf-8",
    )
    (run_dir / "run-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(output.metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


def pipeline_specification() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": PIPELINE_SPEC_VERSION,
        "scientific_role": "evaluation instrument for the ICDS dataset methodology",
        "prediction_unit": "one complete canonical scientific trace",
        "prediction_horizon": "post_trace_offline",
        "feature_profiles": {
            "C0": {"profile": "C1", "feature_count": 90},
            "C1": {"profile": "C1", "feature_count": 90},
            "C2": {
                "profile": "C2",
                "feature_count": 87,
                "excluded_features": sorted(set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES)),
            },
        },
        "preprocessing": {
            "LR": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "scaler_fit_scope": "current_training_cohort_only",
                "imputation": None,
                "feature_filtering": None,
            },
            "DT": {"pipeline": ["DecisionTreeClassifier"], "scaling": None},
            "RF": {"pipeline": ["RandomForestClassifier"], "scaling": None},
        },
        "models": {
            "LR": LR_PARAMETERS,
            "DT": DT_PARAMETERS,
            "RF": RF_PARAMETERS,
        },
        "model_seed_policy": {
            "formula": "family_base + repetition; identical across C0/C1/C2",
            "family_bases": MODEL_SEED_BASE,
            "repetitions": {
                str(repetition): {
                    model: model_seed(repetition, model) for model in MODELS
                } for repetition in range(1, 6)
            },
        },
        "threshold_policy": {
            "positive_class": "anomaly",
            "predicted_anomaly_probability_greater_than_or_equal": PREDICTION_THRESHOLD,
            "tuned": False,
        },
        "target_mapping": {"normal": 0, "anomaly": 1},
        "metrics": {
            "contract_version": METRIC_CONTRACT_VERSION,
            "aggregate": [
                "anomaly_precision", "anomaly_recall", "anomaly_f1",
                "balanced_accuracy", "confusion_matrix",
            ],
            "normal_subgroups": [ORDINARY, TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN],
            "anomaly_subgroups": [TIMING_ANOMALY, NONCE_ANOMALY],
        },
        "prediction_schema_version": PREDICTION_RECORD_VERSION,
        "run_provenance_schema_version": RUN_PROVENANCE_VERSION,
        "upstream_fingerprints": {
            "candidate_file_sha256": EXPECTED_CANDIDATE_FILE_SHA256,
            "candidate_fingerprint": EXPECTED_CANDIDATE_FINGERPRINT,
            "feature_spec_file_sha256": EXPECTED_FEATURE_SPEC_FILE_SHA256,
            "phase1_C1": EXPECTED_C1_FINGERPRINT,
            "phase1_C2": EXPECTED_C2_FINGERPRINT,
            "phase2_file_sha256": EXPECTED_PHASE2_FILE_SHA256,
            "phase2_manifest": EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        },
        "forbidden_operations": [
            "dataset_wide_preprocessing", "test_dependent_preprocessing",
            "imputation", "constant_feature_removal", "feature_selection",
            "hyperparameter_search", "threshold_tuning", "model_ranking",
        ],
        "full_scientific_execution_requires_explicit_authorization": True,
    }
    spec["specification_fingerprint"] = _canonical_hash(spec)
    return spec


def write_pipeline_spec(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pipeline_specification(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-spec", type=Path)
    args = parser.parse_args()
    if args.write_spec is None:
        parser.error("Phase 3 does not auto-execute models; use --write-spec only")
    write_pipeline_spec(args.write_spec)
    print(json.dumps({
        "specification_fingerprint": pipeline_specification()["specification_fingerprint"],
        "models_executed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
