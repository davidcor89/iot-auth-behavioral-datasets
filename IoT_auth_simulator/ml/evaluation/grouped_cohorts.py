"""Construct matched C0/C1 cohorts inside the frozen grouped train partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.evaluation.experiment_manifest import (
    FRESH_ATTEMPT_HN,
    HN_ROLES,
    NONCE_ANOMALY,
    ORDINARY,
    SAME_ATTEMPT_HN,
    SampleControlRecord,
    TIMING_ANOMALY,
    TIMING_HN,
    candidate_control_records,
    device_aware_select,
)


GROUPED_COHORT_VERSION = "icds-device-grouped-matched-cohorts-v1"
GROUPED_COHORT_SEED = 43001
EXPECTED_GROUPED_FINGERPRINT = (
    "sha256:3d652875b33d562b9fce5b007f01a7d99c26716d9cf5f3950b2c89de21f0579c"
)


class GroupedCohortError(ValueError):
    """The frozen grouped partition cannot support the declared construction."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select(
    records: Sequence[SampleControlRecord], count: int, namespace: str,
) -> tuple[SampleControlRecord, ...]:
    return device_aware_select(
        records, count, seed=GROUPED_COHORT_SEED, namespace=f"grouped/{namespace}",
    )


def _ids(records: Sequence[SampleControlRecord]) -> list[str]:
    return sorted(record.sample_id for record in records)


def _counts(records: Sequence[SampleControlRecord]) -> dict[str, Any]:
    roles = Counter(record.role for record in records)
    targets = Counter(record.binary_target for record in records)
    return {
        "total": len(records),
        "binary": {"normal": targets[0], "anomaly": targets[1]},
        "roles": dict(sorted(roles.items())),
        "distinct_devices": len({record.device_key for record in records}),
    }


def build_grouped_cohorts(
    candidate: Mapping[str, Any], experiment_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the simplest balanced 71/71 C0/C1 comparison without moving devices."""
    grouped = experiment_manifest["device_grouped_sensitivity"]
    split = grouped["accepted_split"]
    if (
        grouped["status"] != "STRUCTURALLY_ADEQUATE"
        or split is None
        or split["seed"] != GROUPED_COHORT_SEED
        or split["fingerprint"] != EXPECTED_GROUPED_FINGERPRINT
    ):
        raise GroupedCohortError("frozen grouped split identity changed")

    records_by_id = {
        record.sample_id: record for record in candidate_control_records(candidate)
    }
    train = tuple(records_by_id[sample_id] for sample_id in split["train_trace_ids"])
    test = tuple(records_by_id[sample_id] for sample_id in split["test_trace_ids"])
    by_role = {
        role: tuple(record for record in train if record.role == role)
        for role in (ORDINARY, TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN,
                     TIMING_ANOMALY, NONCE_ANOMALY)
    }

    # There are 71 grouped-training ordinary normals.  Matching binary
    # prevalence therefore admits 71 anomalies: remove one deterministically
    # from each available anomaly family (35 timing + 36 nonce).
    timing_anomalies = _select(by_role[TIMING_ANOMALY], 35, "anomaly/timing")
    nonce_anomalies = _select(by_role[NONCE_ANOMALY], 36, "anomaly/nonce")
    anomalies = tuple(sorted((*timing_anomalies, *nonce_anomalies), key=lambda row: row.sample_id))

    ordinary_c0 = tuple(sorted(by_role[ORDINARY], key=lambda row: row.sample_id))
    retained_ordinary = _select(ordinary_c0, 36, "c1/ordinary-retained")
    hns_by_role = {
        TIMING_HN: _select(by_role[TIMING_HN], 12, "c1/hn/timing"),
        SAME_ATTEMPT_HN: _select(by_role[SAME_ATTEMPT_HN], 12, "c1/hn/same-attempt"),
        FRESH_ATTEMPT_HN: _select(by_role[FRESH_ATTEMPT_HN], 11, "c1/hn/fresh-attempt"),
    }
    hns = tuple(sorted(
        (record for role in HN_ROLES for record in hns_by_role[role]),
        key=lambda row: row.sample_id,
    ))
    c0 = tuple(sorted((*ordinary_c0, *anomalies), key=lambda row: row.sample_id))
    c1 = tuple(sorted((*retained_ordinary, *hns, *anomalies), key=lambda row: row.sample_id))

    c0_ids, c1_ids = set(_ids(c0)), set(_ids(c1))
    test_ids = set(_ids(test))
    anomaly_ids = set(_ids(anomalies))
    if len(c0) != 142 or len(c1) != 142 or c0_ids & test_ids or c1_ids & test_ids:
        raise GroupedCohortError("matched grouped cohorts violate size or test isolation")
    if anomaly_ids - c0_ids or anomaly_ids - c1_ids:
        raise GroupedCohortError("grouped C0/C1 anomaly membership differs")
    if _counts(c0)["binary"] != {"normal": 71, "anomaly": 71}:
        raise GroupedCohortError("grouped C0 is not prevalence matched")
    if _counts(c1)["binary"] != {"normal": 71, "anomaly": 71}:
        raise GroupedCohortError("grouped C1 is not prevalence matched")

    artifact: dict[str, Any] = {
        "schema_version": GROUPED_COHORT_VERSION,
        "source_grouped_split_fingerprint": split["fingerprint"],
        "source_grouped_seed": split["seed"],
        "selection_seed": GROUPED_COHORT_SEED,
        "construction": {
            "interpretation": "secondary device-disjoint descriptive sensitivity only",
            "train_devices_unchanged": True,
            "test_devices_unchanged": True,
            "normal_target": 71,
            "anomaly_target": 71,
            "C0_normal_allocation": {ORDINARY: 71},
            "C1_normal_allocation": {
                ORDINARY: 36,
                TIMING_HN: 12,
                SAME_ATTEMPT_HN: 12,
                FRESH_ATTEMPT_HN: 11,
            },
            "common_anomaly_allocation": {TIMING_ANOMALY: 35, NONCE_ANOMALY: 36},
            "allocation_reason": (
                "largest balanced comparison supported by 71 ordinary normals; "
                "odd normal total gives 36 ordinary and 35 HN traces"
            ),
        },
        "train_device_ids": split["train_device_ids"],
        "test_device_ids": split["test_device_ids"],
        "common_test_trace_ids": _ids(test),
        "C0": {"training_trace_ids": _ids(c0), "counts": _counts(c0)},
        "C1": {
            "training_trace_ids": _ids(c1),
            "counts": _counts(c1),
            "retained_ordinary_trace_ids": _ids(retained_ordinary),
            "introduced_hn_trace_ids": _ids(hns),
            "introduced_hn_trace_ids_by_role": {
                role: _ids(hns_by_role[role]) for role in HN_ROLES
            },
        },
        "common_anomaly_trace_ids": _ids(anomalies),
        "models_executed": False,
        "features_inspected_for_selection": False,
        "device_profiles_inspected_for_selection": False,
    }
    artifact["artifact_fingerprint"] = _canonical_hash(artifact)
    return artifact


def write_grouped_cohorts(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    manifest = json.loads(args.experiment_manifest.read_text(encoding="utf-8"))
    artifact = build_grouped_cohorts(candidate, manifest)
    write_grouped_cohorts(args.output, artifact)
    print(json.dumps({
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "C0_training_count": artifact["C0"]["counts"]["total"],
        "C1_training_count": artifact["C1"]["counts"]["total"],
        "models_executed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
