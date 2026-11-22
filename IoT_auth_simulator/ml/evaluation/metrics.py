"""Reconstruct the frozen evaluation metric set from persisted prediction records."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from ml.evaluation.experiment_manifest import (
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
    ORDINARY,
    SAME_ATTEMPT_HN,
    TIMING_ANOMALY,
    TIMING_HN,
)


METRIC_CONTRACT_VERSION = "icds-binary-metric-contract-v1"
NORMAL_ROLES = (ORDINARY, TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN)
ANOMALY_ROLES = (TIMING_ANOMALY, NONCE_ANOMALY)


class PredictionContractError(ValueError):
    """Persisted prediction records do not form one valid evaluation run."""


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def reconstruct_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute only the approved aggregate and role-specific diagnostics."""
    if not records:
        raise PredictionContractError("at least one prediction record is required")
    trace_ids = [str(record["trace_id"]) for record in records]
    if len(trace_ids) != len(set(trace_ids)):
        raise PredictionContractError("duplicate trace prediction within one run")
    run_keys = {
        (record["repetition"], record["condition"], record["model"])
        for record in records
    }
    if len(run_keys) != 1:
        raise PredictionContractError("records must belong to exactly one run")

    pairs: list[tuple[int, int]] = []
    roles: Counter[str] = Counter()
    for record in records:
        truth, prediction = record["true_target"], record["predicted_class"]
        score = record["predicted_anomaly_probability"]
        role = str(record["scientific_role"])
        if truth not in (0, 1) or prediction not in (0, 1):
            raise PredictionContractError("targets and predictions must be binary")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise PredictionContractError("prediction score must be finite")
        if not 0.0 <= float(score) <= 1.0:
            raise PredictionContractError("predicted probability must be in [0, 1]")
        if role not in (*NORMAL_ROLES, *ANOMALY_ROLES):
            raise PredictionContractError(f"unknown scientific role: {role}")
        pairs.append((int(truth), int(prediction)))
        roles[role] += 1

    tn = sum(truth == 0 and prediction == 0 for truth, prediction in pairs)
    fp = sum(truth == 0 and prediction == 1 for truth, prediction in pairs)
    fn = sum(truth == 1 and prediction == 0 for truth, prediction in pairs)
    tp = sum(truth == 1 and prediction == 1 for truth, prediction in pairs)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)

    normal_diagnostics: dict[str, Any] = {}
    for role in NORMAL_ROLES:
        role_records = [record for record in records if record["scientific_role"] == role]
        false_positives = sum(record["predicted_class"] == 1 for record in role_records)
        normal_diagnostics[role] = {
            "count": len(role_records),
            "false_positive_count": false_positives,
            "false_positive_rate": _safe_ratio(false_positives, len(role_records)),
        }

    anomaly_diagnostics: dict[str, Any] = {}
    for role in ANOMALY_ROLES:
        role_records = [record for record in records if record["scientific_role"] == role]
        true_positives = sum(record["predicted_class"] == 1 for record in role_records)
        anomaly_diagnostics[role] = {
            "count": len(role_records),
            "true_positive_count": true_positives,
            "recall": _safe_ratio(true_positives, len(role_records)),
        }

    return {
        "schema_version": METRIC_CONTRACT_VERSION,
        "prediction_count": len(records),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": _safe_ratio(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
        "normal_role_diagnostics": normal_diagnostics,
        "anomaly_role_diagnostics": anomaly_diagnostics,
        "scientific_role_counts": dict(sorted(roles.items())),
    }
