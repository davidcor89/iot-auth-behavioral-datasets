"""Freeze ICDS experimental cohorts before any preprocessing or model fitting.

The builder consumes only privileged experimental-control metadata: stable
sample identity, scientific role, binary truth for integrity checks, and stable
population-device identity.  It never reads O03 observations, Device Profile
parameters, validator margins, or model-derived information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ml.icds.trace_features import (
    C1_FEATURE_NAMES,
    C2_FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
)
from simulator.data.dataset_contract import DATASET_VERSION, FIELD_MANIFEST_VERSION
from simulator.provenance import (
    BEHAVIOR_MODEL_VERSION,
    EVENT_SCHEMA_VERSION,
    GENERATOR_SOFTWARE_VERSION,
)


EXPERIMENT_MANIFEST_VERSION = "icds-experiment-manifest-v1"
PRIMARY_SPLIT_VERSION = "icds-role-stratified-trace-split-v1"
DEVICE_AWARE_SELECTION_VERSION = "icds-device-round-robin-selection-v1"
GROUPED_SENSITIVITY_VERSION = "icds-device-grouped-sensitivity-search-v1"

CANDIDATE_PATH = "artifacts/icds-timing-nonce-candidate/scientific-candidate.json"
CANDIDATE_BUILD_MANIFEST_PATH = "artifacts/icds-timing-nonce-candidate/build-manifest.json"
FEATURE_SPEC_PATH = "IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json"
PHASE1_AUDIT_PATH = "artifacts/icds-ml-phase1/structural-extraction-audit.json"

EXPECTED_CANDIDATE_FILE_SHA256 = "fa22706b090bedf9b944deee8cc02f7df1bb186661c1cb428d6af82705f08bce"
EXPECTED_FEATURE_SPEC_FILE_SHA256 = "f425fcc081a12f20335e21692b8156260793e02829159e54ec1f4f426c8f54bb"
EXPECTED_C1_FINGERPRINT = "sha256:4f3510f849c978dc30d89e4b661446133669503f51abff9e6517521dd747008f"
EXPECTED_C2_FINGERPRINT = "sha256:39c7a57dd4c989bfc70c88379235467254760255318b06b90ff5713de2a3368e"
EXPECTED_CANDIDATE_FINGERPRINT = "sha256:be34fcbce532d4791cf3f27f056146215fdd255d1e28415e00e3e77f248d3f5c"
EXPECTED_CANDIDATE_COMMIT = "2a9883bdb441a83d3887f496f756ad3b9c4a1d1b"

ORDINARY = "ordinary_normal"
TIMING_HN = "hard_negative:small_positive_causal_margin"
TIMING_ANOMALY = "anomaly:timestamp_inconsistency"
SAME_ATTEMPT_HN = "hard_negative:legitimate_same_attempt_nonce_recurrence"
FRESH_ATTEMPT_HN = "hard_negative:fresh_nonce_cross_attempt"
NONCE_ANOMALY = "anomaly:nonce_reuse"
ROLES = (
    ORDINARY,
    TIMING_HN,
    TIMING_ANOMALY,
    SAME_ATTEMPT_HN,
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
)
HN_ROLES = (TIMING_HN, SAME_ATTEMPT_HN, FRESH_ATTEMPT_HN)
ANOMALY_ROLES = (TIMING_ANOMALY, NONCE_ANOMALY)
ROLE_TO_BINARY_TARGET = {
    ORDINARY: 0,
    TIMING_HN: 0,
    SAME_ATTEMPT_HN: 0,
    FRESH_ATTEMPT_HN: 0,
    TIMING_ANOMALY: 1,
    NONCE_ANOMALY: 1,
}
EXPECTED_ROLE_COUNTS = {
    ORDINARY: 96,
    TIMING_HN: 48,
    TIMING_ANOMALY: 48,
    SAME_ATTEMPT_HN: 48,
    FRESH_ATTEMPT_HN: 48,
    NONCE_ANOMALY: 48,
}
TEST_ROLE_COUNTS = {
    ORDINARY: 24,
    TIMING_HN: 12,
    TIMING_ANOMALY: 12,
    SAME_ATTEMPT_HN: 12,
    FRESH_ATTEMPT_HN: 12,
    NONCE_ANOMALY: 12,
}
SPLIT_SEEDS = (41001, 41002, 41003, 41004, 41005)
COHORT_SEEDS = (42001, 42002, 42003, 42004, 42005)
GROUPED_SEARCH_SEEDS = tuple(range(43001, 43101))


class ExperimentManifestError(ValueError):
    """Frozen candidate or cohort construction violates a Phase-2 invariant."""


@dataclass(frozen=True)
class SampleControlRecord:
    sample_id: str
    device_key: str
    role: str
    binary_target: int


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_seed(root_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{EXPERIMENT_MANIFEST_VERSION}|{root_seed}|{namespace}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def candidate_control_records(candidate: Mapping[str, Any]) -> tuple[SampleControlRecord, ...]:
    """Read only metadata permitted for Phase-2 experimental control."""
    if candidate.get("candidate_fingerprint") != EXPECTED_CANDIDATE_FINGERPRINT:
        raise ExperimentManifestError("candidate semantic fingerprint changed")
    samples = candidate.get("samples")
    if not isinstance(samples, list) or len(samples) != 336 or candidate.get("sample_count") != 336:
        raise ExperimentManifestError("candidate must contain exactly 336 samples")
    records: list[SampleControlRecord] = []
    for sample in samples:
        metadata = sample["privileged_metadata"]
        role = metadata["role"]["key"]
        if role not in ROLE_TO_BINARY_TARGET:
            raise ExperimentManifestError(f"unexpected scientific role: {role}")
        declared_truth = metadata["detector_truth"]
        expected_truth = "anomaly" if ROLE_TO_BINARY_TARGET[role] else "normal"
        if declared_truth != expected_truth:
            raise ExperimentManifestError(f"role/truth disagreement for {sample['sample_id']}")
        records.append(SampleControlRecord(
            sample_id=str(sample["sample_id"]),
            device_key=str(metadata["device"]["population_device_key"]),
            role=role,
            binary_target=ROLE_TO_BINARY_TARGET[role],
        ))
    if len({record.sample_id for record in records}) != 336:
        raise ExperimentManifestError("candidate sample IDs are not unique")
    if len({record.device_key for record in records}) != 64:
        raise ExperimentManifestError("candidate must expose 64 stable device keys")
    if Counter(record.role for record in records) != Counter(EXPECTED_ROLE_COUNTS):
        raise ExperimentManifestError("candidate scientific-role counts changed")
    return tuple(sorted(records, key=lambda item: item.sample_id))


def _records_by_role(records: Iterable[SampleControlRecord]) -> dict[str, list[SampleControlRecord]]:
    grouped: dict[str, list[SampleControlRecord]] = {role: [] for role in ROLES}
    for record in records:
        grouped[record.role].append(record)
    return {role: sorted(values, key=lambda item: item.sample_id) for role, values in grouped.items()}


def _role_counts(records: Iterable[SampleControlRecord]) -> dict[str, int]:
    counts = Counter(record.role for record in records)
    return {role: counts[role] for role in ROLES}


def _binary_counts(records: Iterable[SampleControlRecord]) -> dict[str, int]:
    counts = Counter(record.binary_target for record in records)
    return {"normal": counts[0], "anomaly": counts[1], "total": counts[0] + counts[1]}


def _ids(records: Iterable[SampleControlRecord]) -> list[str]:
    return sorted(record.sample_id for record in records)


def _devices(records: Iterable[SampleControlRecord]) -> list[str]:
    return sorted({record.device_key for record in records})


def _device_coverage(records: Iterable[SampleControlRecord]) -> dict[str, Any]:
    values = tuple(records)
    per_device = Counter(record.device_key for record in values)
    return {
        "distinct_devices": len(per_device),
        "maximum_traces_per_device": max(per_device.values(), default=0),
        "per_role_distinct_devices": {
            role: len({record.device_key for record in values if record.role == role})
            for role in ROLES
        },
    }


def primary_role_stratified_split(
    records: Sequence[SampleControlRecord], split_seed: int,
) -> tuple[tuple[SampleControlRecord, ...], tuple[SampleControlRecord, ...]]:
    """Create an exact 75/25 role-stratified split with isolated role streams."""
    train: list[SampleControlRecord] = []
    test: list[SampleControlRecord] = []
    for role, values in _records_by_role(records).items():
        shuffled = list(values)
        random.Random(_derived_seed(split_seed, f"primary-split/{role}")).shuffle(shuffled)
        test_count = TEST_ROLE_COUNTS[role]
        test.extend(shuffled[:test_count])
        train.extend(shuffled[test_count:])
    train.sort(key=lambda item: item.sample_id)
    test.sort(key=lambda item: item.sample_id)
    return tuple(train), tuple(test)


def device_aware_select(
    records: Sequence[SampleControlRecord], count: int, *, seed: int, namespace: str,
) -> tuple[SampleControlRecord, ...]:
    """Seeded device-first round robin without profile or feature inspection."""
    if count < 0 or count > len(records):
        raise ValueError("selection count is outside the eligible cohort")
    grouped: dict[str, list[SampleControlRecord]] = defaultdict(list)
    for record in records:
        grouped[record.device_key].append(record)
    device_order = sorted(grouped)
    random.Random(_derived_seed(seed, f"{namespace}/device-order")).shuffle(device_order)
    for device_key, values in grouped.items():
        values.sort(key=lambda item: item.sample_id)
        random.Random(_derived_seed(seed, f"{namespace}/trace-order/{device_key}")).shuffle(values)
    selected: list[SampleControlRecord] = []
    depth = 0
    while len(selected) < count:
        progressed = False
        for device_key in device_order:
            values = grouped[device_key]
            if depth < len(values):
                selected.append(values[depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ExperimentManifestError("device-aware selection exhausted eligible records")
        depth += 1
    return tuple(sorted(selected, key=lambda item: item.sample_id))


def _fingerprint(kind: str, repetition: int, payload: Mapping[str, Any]) -> str:
    return _canonical_hash({
        "experiment_version": EXPERIMENT_MANIFEST_VERSION,
        "kind": kind,
        "repetition": repetition,
        **dict(payload),
    })


def _primary_device_diagnostics(
    train: Sequence[SampleControlRecord], test: Sequence[SampleControlRecord],
) -> dict[str, Any]:
    train_devices, test_devices = set(_devices(train)), set(_devices(test))
    return {
        "training": _device_coverage(train),
        "test": _device_coverage(test),
        "devices_in_both": sorted(train_devices & test_devices),
        "overlap_count": len(train_devices & test_devices),
        "device_disjoint": not bool(train_devices & test_devices),
    }


def build_repetition(
    records: Sequence[SampleControlRecord], *, repetition: int,
    split_seed: int, cohort_seed: int,
) -> dict[str, Any]:
    train_pool, common_test = primary_role_stratified_split(records, split_seed)
    pool_by_role = _records_by_role(train_pool)

    c0_ordinary = tuple(pool_by_role[ORDINARY])
    common_anomalies = tuple(sorted(
        [*pool_by_role[TIMING_ANOMALY], *pool_by_role[NONCE_ANOMALY]],
        key=lambda item: item.sample_id,
    ))
    c0 = tuple(sorted([*c0_ordinary, *common_anomalies], key=lambda item: item.sample_id))

    retained_ordinary = device_aware_select(
        c0_ordinary, 36, seed=cohort_seed, namespace="c1/ordinary-retained",
    )
    retained_ids = {record.sample_id for record in retained_ordinary}
    replaced_ordinary = tuple(
        record for record in c0_ordinary if record.sample_id not in retained_ids
    )
    selected_hns = {
        role: device_aware_select(
            pool_by_role[role], 12, seed=cohort_seed,
            namespace=f"c1/{role}",
        )
        for role in HN_ROLES
    }
    introduced_hns = tuple(sorted(
        [record for role in HN_ROLES for record in selected_hns[role]],
        key=lambda item: item.sample_id,
    ))
    c1 = tuple(sorted(
        [*retained_ordinary, *introduced_hns, *common_anomalies],
        key=lambda item: item.sample_id,
    ))

    train_ids, test_ids = set(_ids(train_pool)), set(_ids(common_test))
    c0_ids, c1_ids = set(_ids(c0)), set(_ids(c1))
    if train_ids & test_ids or train_ids | test_ids != {record.sample_id for record in records}:
        raise ExperimentManifestError("primary split does not partition the candidate")
    if c0_ids - train_ids or c1_ids - train_ids or c0_ids & test_ids or c1_ids & test_ids:
        raise ExperimentManifestError("training cohort crosses the common test boundary")

    split_payload = {
        "split_seed": split_seed,
        "training_pool_trace_ids": _ids(train_pool),
        "common_test_trace_ids": _ids(common_test),
    }
    c0_payload = {"training_trace_ids": _ids(c0), "common_test_trace_ids": _ids(common_test)}
    c1_payload = {"training_trace_ids": _ids(c1), "common_test_trace_ids": _ids(common_test)}
    return {
        "repetition": repetition,
        "split_seed": split_seed,
        "c1_cohort_seed": cohort_seed,
        "fingerprints": {
            "primary_split": _fingerprint("primary_split", repetition, split_payload),
            "C0": _fingerprint("C0", repetition, c0_payload),
            "C1": _fingerprint("C1", repetition, c1_payload),
            "C2_membership": _fingerprint("C2_membership", repetition, c1_payload),
        },
        "primary_split": {
            **split_payload,
            "training_pool_counts": {
                "roles": _role_counts(train_pool),
                "binary": _binary_counts(train_pool),
            },
            "common_test_counts": {
                "roles": _role_counts(common_test),
                "binary": _binary_counts(common_test),
            },
            "device_diagnostics": _primary_device_diagnostics(train_pool, common_test),
        },
        "C0": {
            "feature_profile": "C1",
            "training_trace_ids": _ids(c0),
            "common_test_trace_ids": _ids(common_test),
            "training_counts": {"roles": _role_counts(c0), "binary": _binary_counts(c0)},
            "device_coverage": _device_coverage(c0),
            "dedicated_hard_negatives_in_training": 0,
        },
        "C1": {
            "feature_profile": "C1",
            "training_trace_ids": _ids(c1),
            "common_test_trace_ids": _ids(common_test),
            "training_counts": {"roles": _role_counts(c1), "binary": _binary_counts(c1)},
            "device_coverage": _device_coverage(c1),
            "retained_ordinary_normal_trace_ids": _ids(retained_ordinary),
            "replaced_ordinary_normal_trace_ids": _ids(replaced_ordinary),
            "introduced_hard_negative_trace_ids": _ids(introduced_hns),
            "introduced_hard_negatives_by_role": {
                role: _ids(selected_hns[role]) for role in HN_ROLES
            },
            "common_anomaly_trace_ids": _ids(common_anomalies),
        },
        "C2": {
            "feature_profile": "C2",
            "sample_membership_reference": f"repetition-{repetition}/C1",
            "training_trace_ids": _ids(c1),
            "common_test_trace_ids": _ids(common_test),
            "training_membership_equals_C1": True,
            "test_membership_equals_C1": True,
        },
    }


def grouped_sensitivity_search(records: Sequence[SampleControlRecord]) -> dict[str, Any]:
    """Find the first feature-blind structurally adequate 48/16-device split."""
    devices = sorted({record.device_key for record in records})
    criteria = {
        "train_device_count": 48,
        "test_device_count": 16,
        "device_overlap": 0,
        "minimum_test_traces_per_role": 10,
        "minimum_train_traces_per_role": 30,
        "all_six_roles_in_both": True,
        "interpretation": (
            "non-statistical structural floor for descriptive subgroup tables; "
            "researcher approval required before sensitivity evaluation"
        ),
    }
    attempted: list[int] = []
    accepted: dict[str, Any] | None = None
    for seed in GROUPED_SEARCH_SEEDS:
        attempted.append(seed)
        shuffled = list(devices)
        random.Random(seed).shuffle(shuffled)
        test_devices = set(shuffled[:16])
        train = tuple(record for record in records if record.device_key not in test_devices)
        test = tuple(record for record in records if record.device_key in test_devices)
        train_counts, test_counts = _role_counts(train), _role_counts(test)
        if (
            len(_devices(train)) == 48
            and len(_devices(test)) == 16
            and set(_devices(train)).isdisjoint(_devices(test))
            and all(train_counts[role] >= 30 for role in ROLES)
            and all(test_counts[role] >= 10 for role in ROLES)
        ):
            payload = {
                "seed": seed,
                "train_device_ids": _devices(train),
                "test_device_ids": _devices(test),
                "train_trace_ids": _ids(train),
                "test_trace_ids": _ids(test),
                "train_role_counts": train_counts,
                "test_role_counts": test_counts,
            }
            accepted = {
                **payload,
                "train_binary_counts": _binary_counts(train),
                "test_binary_counts": _binary_counts(test),
                "fingerprint": _canonical_hash({
                    "version": GROUPED_SENSITIVITY_VERSION,
                    **payload,
                }),
            }
            break
    return {
        "version": GROUPED_SENSITIVITY_VERSION,
        "status": "STRUCTURALLY_ADEQUATE" if accepted else "NOT_STRUCTURALLY_ADEQUATE",
        "search_seed_range": [GROUPED_SEARCH_SEEDS[0], GROUPED_SEARCH_SEEDS[-1]],
        "seeds_attempted": attempted,
        "acceptance_criteria": criteria,
        "accepted_split": accepted,
        "c0_c1_subsampling_constructed": False,
        "models_or_features_inspected": False,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Mechanically enforce every frozen Phase-2 cohort invariant."""
    if len(manifest["repetitions"]) != 5:
        raise ExperimentManifestError("exactly five repetitions are required")
    all_candidate_ids = set(manifest["candidate_trace_ids"])
    for item in manifest["repetitions"]:
        primary, c0, c1, c2 = item["primary_split"], item["C0"], item["C1"], item["C2"]
        train, test = set(primary["training_pool_trace_ids"]), set(primary["common_test_trace_ids"])
        if len(train) != 252 or len(test) != 84 or train & test or train | test != all_candidate_ids:
            raise ExperimentManifestError("invalid primary split membership")
        if primary["training_pool_counts"]["roles"] != {
            role: EXPECTED_ROLE_COUNTS[role] - TEST_ROLE_COUNTS[role] for role in ROLES
        }:
            raise ExperimentManifestError("invalid primary training role counts")
        if primary["common_test_counts"]["roles"] != TEST_ROLE_COUNTS:
            raise ExperimentManifestError("invalid common test role counts")
        c0_ids, c1_ids = set(c0["training_trace_ids"]), set(c1["training_trace_ids"])
        if len(c0_ids) != 144 or len(c1_ids) != 144:
            raise ExperimentManifestError("C0/C1 must each contain 144 training traces")
        if c0["training_counts"]["binary"] != {"normal": 72, "anomaly": 72, "total": 144}:
            raise ExperimentManifestError("C0 binary prevalence changed")
        if c1["training_counts"]["binary"] != {"normal": 72, "anomaly": 72, "total": 144}:
            raise ExperimentManifestError("C1 binary prevalence changed")
        expected_c0 = {role: 0 for role in ROLES} | {
            ORDINARY: 72, TIMING_ANOMALY: 36, NONCE_ANOMALY: 36,
        }
        expected_c1 = {
            ORDINARY: 36, TIMING_HN: 12, TIMING_ANOMALY: 36,
            SAME_ATTEMPT_HN: 12, FRESH_ATTEMPT_HN: 12, NONCE_ANOMALY: 36,
        }
        if c0["training_counts"]["roles"] != expected_c0:
            raise ExperimentManifestError("C0 scientific-role composition changed")
        if c1["training_counts"]["roles"] != expected_c1:
            raise ExperimentManifestError("C1 scientific-role composition changed")
        c0_anomalies = {
            sample_id for sample_id in c0_ids
            if manifest["trace_role_index"][sample_id] in ANOMALY_ROLES
        }
        c1_anomalies = {
            sample_id for sample_id in c1_ids
            if manifest["trace_role_index"][sample_id] in ANOMALY_ROLES
        }
        if c0_anomalies != c1_anomalies or c0_ids & test or c1_ids & test:
            raise ExperimentManifestError("controlled anomaly/common-test membership failed")
        retained = set(c1["retained_ordinary_normal_trace_ids"])
        replaced = set(c1["replaced_ordinary_normal_trace_ids"])
        introduced = set(c1["introduced_hard_negative_trace_ids"])
        if len(retained) != 36 or len(replaced) != 36 or len(introduced) != 36:
            raise ExperimentManifestError("controlled replacement counts changed")
        c0_ordinary = {
            sample_id for sample_id in c0_ids
            if manifest["trace_role_index"][sample_id] == ORDINARY
        }
        if retained | replaced != c0_ordinary or retained & replaced:
            raise ExperimentManifestError("C1 ordinary replacement partition failed")
        if set(c2["training_trace_ids"]) != c1_ids or set(c2["common_test_trace_ids"]) != test:
            raise ExperimentManifestError("C2 membership must equal C1")

    grouped = manifest["device_grouped_sensitivity"]
    accepted = grouped["accepted_split"]
    if grouped["status"] == "STRUCTURALLY_ADEQUATE":
        if accepted is None:
            raise ExperimentManifestError("adequate grouped split has no persisted membership")
        train_devices = set(accepted["train_device_ids"])
        test_devices = set(accepted["test_device_ids"])
        train_ids = set(accepted["train_trace_ids"])
        test_ids = set(accepted["test_trace_ids"])
        if len(train_devices) != 48 or len(test_devices) != 16 or train_devices & test_devices:
            raise ExperimentManifestError("grouped split is not a disjoint 48/16 device split")
        if train_ids & test_ids or train_ids | test_ids != all_candidate_ids:
            raise ExperimentManifestError("grouped split does not partition all candidate traces")
        if any(accepted["train_role_counts"][role] < 30 for role in ROLES):
            raise ExperimentManifestError("grouped training split violates its structural floor")
        if any(accepted["test_role_counts"][role] < 10 for role in ROLES):
            raise ExperimentManifestError("grouped test split violates its structural floor")


def build_experiment_manifest(
    candidate: Mapping[str, Any], build_manifest: Mapping[str, Any],
    phase1_audit: Mapping[str, Any], *, candidate_file_sha256: str,
    feature_spec_file_sha256: str,
) -> dict[str, Any]:
    if candidate_file_sha256 != EXPECTED_CANDIDATE_FILE_SHA256:
        raise ExperimentManifestError("candidate file SHA-256 changed")
    if feature_spec_file_sha256 != EXPECTED_FEATURE_SPEC_FILE_SHA256:
        raise ExperimentManifestError("Phase-1 feature specification changed")
    if phase1_audit["determinism"]["c1_feature_rows_fingerprint"] != EXPECTED_C1_FINGERPRINT:
        raise ExperimentManifestError("Phase-1 C1 fingerprint changed")
    if phase1_audit["determinism"]["c2_feature_rows_fingerprint"] != EXPECTED_C2_FINGERPRINT:
        raise ExperimentManifestError("Phase-1 C2 fingerprint changed")
    if phase1_audit["feature_counts"] != {"C1": 90, "C2": 87}:
        raise ExperimentManifestError("Phase-1 feature counts changed")
    records = candidate_control_records(candidate)
    repetitions = [
        build_repetition(
            records, repetition=index, split_seed=split_seed, cohort_seed=cohort_seed,
        )
        for index, (split_seed, cohort_seed) in enumerate(
            zip(SPLIT_SEEDS, COHORT_SEEDS), start=1,
        )
    ]
    grouped = grouped_sensitivity_search(records)
    manifest: dict[str, Any] = {
        "schema_version": EXPERIMENT_MANIFEST_VERSION,
        "dataset_identity": {
            "dataset_version": DATASET_VERSION,
            "candidate_path": CANDIDATE_PATH,
            "candidate_file_sha256": candidate_file_sha256,
            "candidate_semantic_fingerprint": candidate["candidate_fingerprint"],
            "candidate_schema_version": candidate["schema_version"],
            "candidate_build_manifest_path": CANDIDATE_BUILD_MANIFEST_PATH,
            "candidate_build_manifest_version": build_manifest["schema_version"],
            "candidate_publication_commit": EXPECTED_CANDIDATE_COMMIT,
            "generator_release": GENERATOR_SOFTWARE_VERSION,
            "generator_source_checkpoint": build_manifest["source"]["source_checkpoint"],
            "behavior_model_version": BEHAVIOR_MODEL_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "field_manifest_version": FIELD_MANIFEST_VERSION,
            "detector_observation_schema": build_manifest["integrity"]["detector_feature_schema"],
        },
        "phase1_feature_contract": {
            "feature_spec_path": FEATURE_SPEC_PATH,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "feature_spec_file_sha256": feature_spec_file_sha256,
            "phase1_audit_path": PHASE1_AUDIT_PATH,
            "C1_feature_count": len(C1_FEATURE_NAMES),
            "C2_feature_count": len(C2_FEATURE_NAMES),
            "C1_feature_rows_fingerprint": EXPECTED_C1_FINGERPRINT,
            "C2_feature_rows_fingerprint": EXPECTED_C2_FINGERPRINT,
            "C2_excluded_features": sorted(set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES)),
        },
        "experiment_definition": {
            "experiment_version": EXPERIMENT_MANIFEST_VERSION,
            "prediction_unit": "one complete canonical scientific trace",
            "prediction_horizon": "post_trace_offline",
            "binary_target": {"normal": 0, "anomaly": 1},
            "scientific_role_to_binary_target": ROLE_TO_BINARY_TARGET,
            "C0": "72 ordinary normals plus 72 common anomalies; C1 feature profile",
            "C1": "36 ordinary normals plus 36 HNs plus 72 common anomalies; C1 feature profile",
            "C2": "identical membership to C1; C2 feature profile only",
            "split_strategy": "role-stratified random trace-level 75/25",
            "split_version": PRIMARY_SPLIT_VERSION,
            "split_ratio": {"train": 0.75, "test": 0.25},
            "stratification_variable": "privileged scientific role",
            "repetition_count": 5,
            "split_seeds": list(SPLIT_SEEDS),
            "cohort_seeds": list(COHORT_SEEDS),
            "device_aware_selection_version": DEVICE_AWARE_SELECTION_VERSION,
            "device_aware_selection": (
                "seeded device order plus per-device trace order; breadth-first round robin"
            ),
            "common_test_invariant": "Test_C0 == Test_C1 == Test_C2",
            "learned_preprocessing_performed": False,
            "model_training_performed": False,
        },
        "candidate_trace_ids": _ids(records),
        "trace_role_index": {record.sample_id: record.role for record in records},
        "repetitions": repetitions,
        "device_grouped_sensitivity": grouped,
        "safety_boundary": {
            "metadata_read": [
                "sample_id", "scientific_role", "detector_truth_for_integrity",
                "population_device_key",
            ],
            "o03_features_read": False,
            "device_profile_parameters_read": False,
            "validator_or_transformation_details_read": False,
            "classifier_information_read": False,
        },
    }
    validate_manifest(manifest)
    manifest["manifest_fingerprint"] = _canonical_hash(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-build-manifest", type=Path, required=True)
    parser.add_argument("--feature-spec", type=Path, required=True)
    parser.add_argument("--phase1-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    build_manifest = json.loads(args.candidate_build_manifest.read_text(encoding="utf-8"))
    phase1_audit = json.loads(args.phase1_audit.read_text(encoding="utf-8"))
    manifest = build_experiment_manifest(
        candidate,
        build_manifest,
        phase1_audit,
        candidate_file_sha256=file_sha256(args.candidate),
        feature_spec_file_sha256=file_sha256(args.feature_spec),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "repetitions": len(manifest["repetitions"]),
        "grouped_sensitivity": manifest["device_grouped_sensitivity"]["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
