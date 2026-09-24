"""Build the frozen controlled Timing+Nonce ICDS scientific candidate.

The candidate is a semantic scientific artifact, not an ML feature table. Its
detector observations and privileged methodology metadata are explicitly
separated within every sample record.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulator.composition.build import (
    ADAPTER_DISPATCH_REGISTRY_VERSION,
    CompositionBuildResult,
    build_composition,
)
from simulator.composition.eligibility import build_eligibility_edges, eligibility_digest
from simulator.composition.fingerprint import (
    DETECTOR_OBSERVATION_CONTRACT,
    detector_evidence_record,
)
from simulator.composition.model import ORDINARY_NORMAL, RoleKind
from simulator.composition.scheduler import build_composition_plan
from simulator.composition.timing_nonce_feasibility import (
    EXPECTED_BASELINE_COMMIT,
    FEASIBILITY_ROLES,
    FRESH_NONCE_HN_ROLE,
    NONCE_ANOMALY,
    PLANNING_HYPOTHESIS,
    SAME_ATTEMPT_NONCE_HN_ROLE,
    TIMING_ANOMALY,
    TIMING_HN,
    default_feasibility_configuration,
    planning_hypothesis_specification,
)
from simulator.device_profiles.scientific_population import scientific_configuration_record
from simulator.role_integration.direct_timing import resolve_timing_relation
from simulator.role_integration.integrated_stress import build_stress_population_and_pool
from simulator.role_integration.models import stable_reference
from simulator.role_integration.nonce_boundary import classify_multi_attempt_source
from simulator.provenance import FEATURE_SCHEMA_VERSION


CANDIDATE_SCHEMA_VERSION = "4d4c3b-timing-nonce-candidate-v1"
CANDIDATE_MANIFEST_VERSION = "4d4c3b-candidate-build-manifest-v1"
CANDIDATE_CHARACTERIZATION_VERSION = "4d4c3b-candidate-characterization-v1"
CANDIDATE_BRANCH = "4d4c3b-icds-timing-nonce-candidate"
CANDIDATE_SOURCE_COMMIT = "fc97fdcc71012f4a7fa421a4b8cec08cb0b880fd"

EXPECTED_POPULATION_FINGERPRINT = (
    "sha256:ec38b07c7b1831a4e4e2daec1d63fb73adae17aadfd442d22032cdb9c255cb6f"
)
EXPECTED_POOL_FINGERPRINT = (
    "sha256:09a0554012f77f3da8e5c4ec3766d56e0fa9cfd8504239447c9ce255b6945504"
)
EXPECTED_APPLICABILITY_DIGEST = (
    "sha256:a66eae00676dba4fd303b553e5a486c568124c87341a3f30fb8e049421529904"
)
EXPECTED_PLAN_FINGERPRINT = (
    "sha256:fdb19b628bd511a2415e2eb745637e81c6ef28b60d996b0f4c0023867d5dd154"
)
EXPECTED_REJECTED_KEYS = (
    "base/population-device-00000037/opportunity-3",
    "base/population-device-00000060/opportunity-8",
)
EXPECTED_DEVICE_DIVERSITY = {
    ORDINARY_NORMAL.key: (64, 2),
    TIMING_HN.key: (43, 2),
    TIMING_ANOMALY.key: (44, 2),
    SAME_ATTEMPT_NONCE_HN_ROLE.key: (46, 2),
    FRESH_NONCE_HN_ROLE.key: (43, 2),
    NONCE_ANOMALY.key: (43, 2),
}

_OBSERVABLE_FIELDS = (
    "event_type", "observed_timestamp", "observed_delay_since_previous_event",
    "result", "failure_reason", "retry_count", "token_id", "nonce", "topic",
    "resource_id", "requested_action",
)


class CandidateBuildError(RuntimeError):
    """The frozen candidate failed a required reproduction or integrity gate."""


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "q25": None, "median": None,
                "mean": None, "q75": None, "maximum": None}
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        location = fraction * (len(ordered) - 1)
        low, high = math.floor(location), math.ceil(location)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (location - low)

    return {
        "minimum": ordered[0], "q25": at(0.25), "median": at(0.5),
        "mean": statistics.fmean(ordered), "q75": at(0.75),
        "maximum": ordered[-1],
    }


def _concentration(counts: Iterable[int]) -> dict[str, float]:
    ordered = sorted((value for value in counts if value), reverse=True)
    total = sum(ordered)
    return {
        f"top_{limit}_fraction": (sum(ordered[:limit]) / total if total else 0.0)
        for limit in (1, 5, 10)
    }


def _profile_record(assignment: Any) -> dict[str, Any]:
    profile = assignment.profile
    return {
        "behavioral_profile_id": profile.behavioral_profile_id,
        "profile_contract_version": profile.contract_version,
        "population_device_key": assignment.population_device_key,
        "profile_seed_reference": assignment.profile_seed_reference,
        "scalar_strata": dict(assignment.scalar_strata),
        "lifecycle_cell": assignment.lifecycle_cell_id,
        "latent_tendencies": {
            "T": {
                "AUTH": asdict(profile.timing.auth),
                "TRANSITION": asdict(profile.timing.transition),
                "ACCESS": asdict(profile.timing.access),
                "RENEWAL": asdict(profile.timing.renewal),
            },
            "A": profile.activity,
            "R": profile.recovery,
            "N": profile.renewal,
            "L": asdict(profile.lifecycle),
        },
    }


def _observed(record: Mapping[str, Any]) -> float:
    return float(record.get("observed_timestamp", record["timestamp"]))


def _canonical_detector_evidence_record(
    records: Sequence[Mapping[str, Any]], *, trace_namespace: str,
    device_namespace: str,
) -> dict[str, Any]:
    """Publish stable O03 evidence without runtime-generated opaque material.

    Token and nonce values are opaque identifiers in O03.  Their scientific
    content for this within-trace candidate is their equality/inequality
    relationship, not the random bytes selected by the legacy generator.
    Namespace-scoped encounter-order pseudonyms preserve those relationships,
    remain distinct across traces, and keep runtime randomness out of candidate
    artifacts and fingerprints.
    """
    evidence = detector_evidence_record(records)
    references: dict[str, dict[str, str]] = {"token_id": {}, "nonce": {}}
    device_alias = stable_reference({
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "population_device_key": device_namespace,
        "purpose": "detector-visible-device-alias",
    })[7:15]
    for row in evidence["rows"]:
        for field in references:
            value = row.get(field)
            if value is None:
                continue
            values = references[field]
            raw = str(value)
            if raw not in values:
                values[raw] = stable_reference({
                    "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                    "trace_namespace": trace_namespace,
                    "opaque_field": field,
                    "encounter_ordinal": len(values) + 1,
                })
            row[field] = values[raw]
        topic = row.get("topic")
        if topic:
            parts = str(topic).split("/")
            if len(parts) >= 3 and parts[0] == "iot":
                parts[1] = device_alias
                row["topic"] = "/".join(parts)
        resource = row.get("resource_id")
        if resource:
            prefix = "resource://"
            suffix = str(resource)[len(prefix):] if str(resource).startswith(prefix) else None
            if suffix is not None:
                parts = suffix.split("/")
                if len(parts) >= 2:
                    parts[0] = device_alias
                    row["resource_id"] = prefix + "/".join(parts)
    return evidence


def _duration(records: Sequence[Mapping[str, Any]]) -> float:
    values = [float(record["timestamp"]) for record in records]
    return max(values) - min(values) if values else 0.0


def _observable_delta(
    base_records: Sequence[Mapping[str, Any]],
    derived_records: Sequence[Mapping[str, Any]],
    *,
    trace_namespace: str,
    device_namespace: str,
) -> dict[str, Any]:
    base_rows = _canonical_detector_evidence_record(
        base_records, trace_namespace=trace_namespace, device_namespace=device_namespace,
    )["rows"]
    derived_rows = _canonical_detector_evidence_record(
        derived_records, trace_namespace=trace_namespace, device_namespace=device_namespace,
    )["rows"]
    changed = Counter()
    for left, right in zip(base_rows, derived_rows):
        for field in _OBSERVABLE_FIELDS:
            if left.get(field) != right.get(field):
                changed[field] += 1
    if len(base_rows) != len(derived_rows):
        changed["row_count"] = abs(len(base_rows) - len(derived_rows))
    return {
        "o03_equal_to_base": base_rows == derived_rows,
        "changed_observable_fields": dict(sorted(changed.items())),
        "changed_observable_field_names": sorted(changed),
        "event_count_delta": len(derived_records) - len(base_records),
        "semantic_duration_delta": _duration(derived_records) - _duration(base_records),
    }


def _timing_margin(base: Any, records: Sequence[Mapping[str, Any]]) -> float | None:
    relation = resolve_timing_relation(base)
    if relation is None:
        return None
    predecessor = records[relation.predecessor_ordinal - 1]
    target = records[relation.target_ordinal - 1]
    return _observed(target) - _observed(predecessor)


def _execute_frozen_build(
    population: Any, pool: Any, plan: Any, transformation_seed: int,
) -> CompositionBuildResult:
    """Single attempt only: a failure propagates without reassignment."""
    return build_composition(
        population, pool, plan, transformation_seed=transformation_seed,
    )


def _assert_reproduction(population: Any, pool: Any, plan: Any) -> None:
    if population.population_fingerprint != EXPECTED_POPULATION_FINGERPRINT:
        raise CandidateBuildError("scientific population fingerprint regression")
    if pool.pool_fingerprint != EXPECTED_POOL_FINGERPRINT:
        raise CandidateBuildError("canonical opportunity-pool fingerprint regression")
    if len(pool.bases) + len(pool.rejected) != 512 or len(pool.bases) != 510:
        raise CandidateBuildError("canonical opportunity admission count regression")
    if tuple(item.base_construction_key for item in pool.rejected) != EXPECTED_REJECTED_KEYS:
        raise CandidateBuildError("canonical rejected-opportunity identity regression")
    if plan.eligibility_digest != EXPECTED_APPLICABILITY_DIGEST:
        raise CandidateBuildError("applicability digest regression")
    if plan.composition_fingerprint != EXPECTED_PLAN_FINGERPRINT:
        raise CandidateBuildError("CompositionPlan fingerprint regression")
    if plan.feasibility_status != "FEASIBLE":
        raise CandidateBuildError("frozen plan is no longer feasible")


def _assert_build_integrity(result: CompositionBuildResult) -> None:
    counts = Counter(sample.assignment.role.key for sample in result.samples)
    if dict(counts) != PLANNING_HYPOTHESIS:
        raise CandidateBuildError("candidate role counts differ from frozen composition")
    references = [sample.assignment.base_trace_reference for sample in result.samples]
    if len(references) != 336 or len(set(references)) != 336:
        raise CandidateBuildError("candidate violates one-role-per-base exclusivity")
    hn_samples = [sample for sample in result.samples
                  if sample.assignment.role.kind is RoleKind.HARD_NEGATIVE]
    if len(hn_samples) != 144 or any(
        sample.detector_truth != "normal"
        or sample.complete_invariant_ids
        or "qualified" not in sample.semantic_qualifications
        or not sample.canonical_reload_valid
        for sample in hn_samples
    ):
        raise CandidateBuildError("HN normality or qualification gate failed")
    timing = [sample for sample in result.samples if sample.assignment.role == TIMING_ANOMALY]
    nonce = [sample for sample in result.samples if sample.assignment.role == NONCE_ANOMALY]
    if len(timing) != 48 or any(
        "INV-TIME-OBSERVED-CAUSAL" not in sample.complete_invariant_ids
        for sample in timing
    ):
        raise CandidateBuildError("Timing anomaly invariant gate failed")
    if len(nonce) != 48 or any(
        "INV-REPLAY-NONCE-CROSS-ATTEMPT" not in sample.complete_invariant_ids
        for sample in nonce
    ):
        raise CandidateBuildError("nonce anomaly invariant gate failed")
    if result.diagnostics["same_truth_duplicate_groups"]:
        raise CandidateBuildError("strict candidate contains same-truth O03 duplicates")
    if result.diagnostics["conflicting_truth_duplicate_groups"]:
        raise CandidateBuildError("candidate contains conflicting-truth O03 duplicates")


def _canonical_integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Check strict integrity again on the published canonical O03 evidence."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        evidence_hash = stable_reference({
            "observation_contract": DETECTOR_OBSERVATION_CONTRACT,
            "rows": row["detector_observations"],
        })
        groups[evidence_hash].append(row["privileged_metadata"]["detector_truth"])
    same_truth = 0
    conflicting = 0
    for truths in groups.values():
        if len(truths) < 2:
            continue
        if len(set(truths)) == 1:
            same_truth += 1
        else:
            conflicting += 1
    if same_truth:
        raise CandidateBuildError(
            "published candidate contains same-truth canonical O03 duplicates"
        )
    if conflicting:
        raise CandidateBuildError(
            "published candidate contains conflicting-truth canonical O03 duplicates"
        )
    return {
        "same_truth_duplicate_groups": same_truth,
        "conflicting_truth_duplicate_groups": conflicting,
        "unique_detector_evidence_hashes": len(groups),
    }


def _selection_characterization(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    devices = Counter(row["privileged_metadata"]["device"]["population_device_key"]
                      for row in selected)
    profiles = [row["privileged_metadata"]["device_profile"] for row in selected]
    structures = [row["privileged_metadata"]["structure"] for row in selected]
    return {
        "samples": len(selected),
        "distinct_devices": len(devices),
        "samples_per_device": _quantiles(tuple(devices.values())),
        "maximum_per_device": max(devices.values(), default=0),
        "concentration": _concentration(devices.values()),
        "profile_coverage": {
            "scalar_strata": {
                name: dict(sorted(Counter(
                    str(profile["scalar_strata"][name]) for profile in profiles
                ).items()))
                for name in sorted(profiles[0]["scalar_strata"] if profiles else ())
            },
            "lifecycle_cells": dict(sorted(Counter(
                profile["lifecycle_cell"] for profile in profiles
            ).items())),
            "distinct_lifecycle_cells": len({
                profile["lifecycle_cell"] for profile in profiles
            }),
        },
        "structure": {
            name: dict(sorted(Counter(str(item[name]) for item in structures).items()))
            for name in ("lifecycle", "recovery", "renewal", "activity_count")
        } | {
            name: _quantiles(tuple(float(item[name]) for item in structures))
            for name in ("authentication_attempt_count", "base_event_count",
                         "base_duration")
        },
    }


def _role_characterization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        role.key: _selection_characterization([
            row for row in rows
            if row["privileged_metadata"]["role"]["key"] == role.key
        ])
        for role in FEASIBILITY_ROLES
    }


def _delta_characterization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in FEASIBILITY_ROLES:
        selected = [row for row in rows if row["privileged_metadata"]["role"]["key"] == role.key]
        deltas = [row["privileged_metadata"]["transformation_delta"] for row in selected]
        fields = Counter(field for delta in deltas
                         for field, count in delta["changed_observable_fields"].items()
                         for _ in range(count))
        result[role.key] = {
            "samples": len(selected),
            "o03_equal_to_base": sum(delta["o03_equal_to_base"] for delta in deltas),
            "o03_different_from_base": sum(not delta["o03_equal_to_base"] for delta in deltas),
            "changed_observable_fields": dict(sorted(fields.items())),
            "event_count_delta": dict(sorted(Counter(
                str(delta["event_count_delta"]) for delta in deltas
            ).items())),
            "semantic_duration_delta": _quantiles(tuple(
                delta["semantic_duration_delta"] for delta in deltas
            )),
        }
    return result


def build_timing_nonce_candidate() -> dict[str, Any]:
    configuration = default_feasibility_configuration()
    rng_state = random.getstate()
    try:
        run_population, pool = build_stress_population_and_pool(configuration)
        population = run_population.behavioral_profiles
        specification = planning_hypothesis_specification(configuration)
        plan = build_composition_plan(population, pool, specification)
        _assert_reproduction(population, pool, plan)
        result = _execute_frozen_build(
            population, pool, plan, configuration.transformation_seed,
        )
        _assert_build_integrity(result)

        bases = {base.base_trace_reference: base for base in pool.bases}
        assignments_by_device = {
            item.population_device_key: item for item in population.assignments
        }
        edges = build_eligibility_edges(
            population, pool, FEASIBILITY_ROLES, configuration.role_assignment_seed,
        )
        applicable_by_base: dict[str, list[str]] = defaultdict(list)
        edge_by_pair = {}
        for edge in edges:
            applicable_by_base[edge.base_trace_reference].append(edge.role.key)
            edge_by_pair[(edge.role.key, edge.base_trace_reference)] = edge

        rows = []
        timing_margins: dict[str, list[float]] = defaultdict(list)
        nonce_sources: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in result.samples:
            assignment = sample.assignment
            base = bases[assignment.base_trace_reference]
            edge = edge_by_pair[(assignment.role.key, assignment.base_trace_reference)]
            profile_assignment = assignments_by_device[assignment.device_key]
            o03 = _canonical_detector_evidence_record(
                sample.event_records, trace_namespace=base.base_trace_reference,
                device_namespace=assignment.device_key,
            )
            delta = _observable_delta(
                base.event_records, sample.event_records,
                trace_namespace=base.base_trace_reference,
                device_namespace=assignment.device_key,
            )
            margin = None
            if assignment.role in {TIMING_HN, TIMING_ANOMALY}:
                margin = _timing_margin(base, sample.event_records)
                if margin is None:
                    raise CandidateBuildError("assigned Timing role lost its causal relation")
                timing_margins[assignment.role.key].append(margin)
            source = None
            if assignment.role in {FRESH_NONCE_HN_ROLE, NONCE_ANOMALY}:
                source = classify_multi_attempt_source(base)
                nonce_sources[assignment.role.key][source] += 1
            sample_id = stable_reference({
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "plan_fingerprint": plan.composition_fingerprint,
                "base_trace_reference": assignment.base_trace_reference,
                "scientific_role": assignment.role.to_record(),
            })
            canonical_evidence_hash = stable_reference(o03)
            rows.append({
                "sample_id": sample_id,
                "detector_observations": o03["rows"],
                "privileged_metadata": {
                    "observation_contract": o03["observation_contract"],
                    "detector_evidence_hash": canonical_evidence_hash,
                    "detector_truth": sample.detector_truth,
                    "role": assignment.role.to_record(),
                    "semantic_family": assignment.family,
                    "device": {
                        "population_device_key": assignment.device_key,
                    },
                    "device_profile": _profile_record(profile_assignment),
                    "base": {
                        "base_trace_reference": base.base_trace_reference,
                        "base_construction_key": base.base_construction_key,
                        "opportunity_randomness_reference": base.opportunity_randomness_reference,
                    },
                    "applicable_scientific_roles": sorted(
                        applicable_by_base[base.base_trace_reference]
                    ),
                    "assigned_applicability_reference": assignment.applicability_reference,
                    "control_memberships": list(assignment.control_memberships),
                    "composition_plan_fingerprint": plan.composition_fingerprint,
                    "intended_invariant_ids": list(sample.intended_invariant_ids),
                    "complete_validated_invariant_ids": list(sample.complete_invariant_ids),
                    "hn_qualification": (
                        "qualified" if assignment.role.kind is RoleKind.HARD_NEGATIVE else None
                    ),
                    "collision_classification": sample.collision_classification,
                    "aggregate_gate_passed": sample.aggregate_gate_passed,
                    "canonical_reload_valid": sample.canonical_reload_valid,
                    "transformation": {
                        "applied": assignment.role.kind is not RoleKind.ORDINARY_NORMAL,
                        "adapter_dispatch_version": ADAPTER_DISPATCH_REGISTRY_VERSION,
                        "transformation_seed": configuration.transformation_seed,
                        "provenance_reference": stable_reference({
                            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                            "base_trace_reference": base.base_trace_reference,
                            "scientific_role": assignment.role.to_record(),
                            "canonical_detector_evidence_hash": canonical_evidence_hash,
                            "intended_invariant_ids": list(sample.intended_invariant_ids),
                            "complete_invariant_ids": list(sample.complete_invariant_ids),
                        }),
                    },
                    "counterfactual_base_reference": base.base_trace_reference,
                    "structure": dict(edge.structural_descriptors),
                    "transformation_delta": delta,
                    "timing_observed_margin": margin,
                    "cross_attempt_source": source,
                },
            })
        rows.sort(key=lambda row: row["sample_id"])
        canonical_integrity = _canonical_integrity(rows)

        candidate_payload = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "observation_contract": DETECTOR_OBSERVATION_CONTRACT,
            "publication_status": "controlled_scientific_candidate_pending_researcher_review",
            "sample_count": len(rows),
            "samples": rows,
        }
        candidate_payload["candidate_fingerprint"] = stable_reference(candidate_payload)
        privileged_fingerprint = stable_reference([
            row["privileged_metadata"] for row in rows
        ])
        candidate_build_fingerprint = stable_reference({
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "composition_plan": plan.composition_fingerprint,
            "transformation_seed": configuration.transformation_seed,
            "candidate_rows": candidate_payload["candidate_fingerprint"],
            "privileged_metadata": privileged_fingerprint,
        })

        role_characterization = _role_characterization(rows)
        deltas = _delta_characterization(rows)
        complete_sets = {
            role.key: dict(sorted(Counter(
                json.dumps(row["privileged_metadata"]["complete_validated_invariant_ids"])
                for row in rows if row["privileged_metadata"]["role"]["key"] == role.key
            ).items()))
            for role in FEASIBILITY_ROLES
        }
        c0_ids = sorted(
            row["sample_id"] for row in rows
            if row["privileged_metadata"]["role"]["key"] in {
                ORDINARY_NORMAL.key, TIMING_ANOMALY.key, NONCE_ANOMALY.key,
            }
        )
        c1_ids = sorted(row["sample_id"] for row in rows)
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_MANIFEST_VERSION,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "source": {
                "branch": CANDIDATE_BRANCH,
                "source_checkpoint": CANDIDATE_SOURCE_COMMIT,
                "methodological_baseline": EXPECTED_BASELINE_COMMIT,
            },
            "configuration": {
                "integrated": asdict(configuration),
                "scientific_population": scientific_configuration_record(
                    configuration.population_configuration()
                ),
                "composition": specification.to_record(),
            },
            "base_pool": {
                "attempted": len(pool.bases) + len(pool.rejected),
                "admitted": len(pool.bases),
                "rejected": len(pool.rejected),
                "rejected_keys": [item.base_construction_key for item in pool.rejected],
                "replacement_or_resampling": False,
            },
            "fingerprints": {
                "population": population.population_fingerprint,
                "opportunity_pool": pool.pool_fingerprint,
                "applicability": eligibility_digest(edges),
                "composition_plan": plan.composition_fingerprint,
                "composition_build": candidate_build_fingerprint,
                "candidate_rows": candidate_payload["candidate_fingerprint"],
                "privileged_metadata": privileged_fingerprint,
            },
            "counts": {
                "total": len(rows),
                "truth": dict(sorted(Counter(
                    row["privileged_metadata"]["detector_truth"] for row in rows
                ).items())),
                "roles": dict(sorted(Counter(
                    row["privileged_metadata"]["role"]["key"] for row in rows
                ).items())),
                "qualified_hard_negatives": sum(
                    row["privileged_metadata"]["hn_qualification"] == "qualified"
                    for row in rows
                ),
            },
            "validation": {
                "all_canonical_reload_valid": all(
                    row["privileged_metadata"]["canonical_reload_valid"] for row in rows
                ),
                "all_aggregate_gates_passed": all(
                    row["privileged_metadata"]["aggregate_gate_passed"] for row in rows
                ),
                "complete_invariant_sets_by_role": complete_sets,
                "unexpected_multi_invariant_samples": sum(
                    len(row["privileged_metadata"]["complete_validated_invariant_ids"]) > 1
                    for row in rows
                ),
            },
            "integrity": {
                "one_role_per_base": len({
                    row["privileged_metadata"]["base"]["base_trace_reference"] for row in rows
                }) == len(rows),
                **canonical_integrity,
                "detector_feature_schema": FEATURE_SCHEMA_VERSION,
                "privileged_metadata_excluded_from_detector_observations": True,
                "opaque_runtime_material_policy": (
                    "trace-namespaced stable pseudonyms preserve token/nonce equality "
                    "without publishing runtime-generated values"
                ),
                "out_of_scope_role_count": 0,
                "replacement_after_failure": False,
            },
            "characterization": {
                "overall": _selection_characterization(rows),
                "roles": role_characterization,
                "transformation_delta": deltas,
                "timing": {
                    "relation": {
                        "predecessor_event_type": "nonce_received",
                        "target_event_type": "response_sent",
                        "relationship": "same authentication attempt and nonce",
                    },
                    "canonical_observed_margin": {
                        role: _quantiles(tuple(values))
                        for role, values in sorted(timing_margins.items())
                    },
                },
                "nonce_cross_attempt_source": {
                    role: dict(sorted(values.items()))
                    for role, values in sorted(nonce_sources.items())
                },
            },
            "projections": {
                "C0": {
                    "description": "ordinary normals plus Timing and nonce anomalies",
                    "privileged_role_filter": [
                        ORDINARY_NORMAL.key, TIMING_ANOMALY.key, NONCE_ANOMALY.key,
                    ],
                    "count": len(c0_ids), "sample_ids": c0_ids,
                },
                "C1": {
                    "description": "complete controlled Timing+Nonce candidate",
                    "privileged_role_filter": [role.key for role in FEASIBILITY_ROLES],
                    "count": len(c1_ids), "sample_ids": c1_ids,
                },
                "caveat": (
                    "C0 to C1 changes both HN presence and sample count/class prevalence; "
                    "04 owns weighting, matching, evaluation sets, and prevalence control."
                ),
            },
            "capabilities": [
                "controlled synthetic authentication-lifecycle behavior",
                "64 persistent Device Profiles",
                "ordinary valid behavior and three qualified HN variants",
                "Timing causal-order and cross-attempt nonce-reuse anomalies",
                "validator-backed GT and controlled scientific composition",
                "device/profile/applicability/control provenance",
                "canonical O03 observations and derivable C0/C1 projections",
            ],
            "limitations": [
                "not empirically calibrated or operationally representative",
                "not comprehensive IoT or behavioral anomaly coverage",
                "no unseen-anomaly or deployment-performance claim",
                "no universal-observability or optimal-monitoring claim",
                "token replay and access/identity/binding roles are excluded",
                "no final ML features, horizon, split regime, or detector claim",
            ],
        }
        manifest["manifest_fingerprint"] = stable_reference(manifest)
        return {"candidate": candidate_payload, "manifest": manifest}
    finally:
        if random.getstate() != rng_state:
            random.setstate(rng_state)
            raise AssertionError("candidate build perturbed global RNG state")


def render_candidate_report(package: Mapping[str, Any]) -> str:
    manifest = package["manifest"]
    counts = manifest["counts"]
    lines = [
        "# Controlled Timing+Nonce ICDS scientific candidate", "",
        "This is a controlled scientific candidate pending researcher inspection, not an ML dataset.",
        "", "## Build result", "",
        f"- Samples: {counts['total']} (normal {counts['truth']['normal']}; anomaly {counts['truth']['anomaly']})",
        f"- Qualified HNs: {counts['qualified_hard_negatives']}",
        f"- Base opportunities attempted/admitted/rejected: "
        f"{manifest['base_pool']['attempted']}/{manifest['base_pool']['admitted']}/"
        f"{manifest['base_pool']['rejected']}",
        f"- Candidate fingerprint: `{manifest['fingerprints']['candidate_rows']}`",
        f"- Build fingerprint: `{manifest['fingerprints']['composition_build']}`",
        "", "## Scientific roles", "",
        "| Role | Count | Devices | Maximum/device | O03 equal/different from base |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in FEASIBILITY_ROLES:
        characterization = manifest["characterization"]["roles"][role.key]
        delta = manifest["characterization"]["transformation_delta"][role.key]
        lines.append(
            f"| `{role.key}` | {characterization['samples']} | "
            f"{characterization['distinct_devices']} | {characterization['maximum_per_device']} | "
            f"{delta['o03_equal_to_base']}/{delta['o03_different_from_base']} |"
        )
    lines.extend([
        "", "## Integrity", "",
        f"- Unique O03 evidence: {manifest['integrity']['unique_detector_evidence_hashes']}/336",
        f"- Same-truth duplicate groups: {manifest['integrity']['same_truth_duplicate_groups']}",
        f"- Conflicting-truth duplicate groups: {manifest['integrity']['conflicting_truth_duplicate_groups']}",
        f"- Unexpected multi-invariant samples: {manifest['validation']['unexpected_multi_invariant_samples']}",
        "- All canonical reload and aggregate validation gates passed.",
        "", "## C0/C1 projections", "",
        f"- C0: {manifest['projections']['C0']['count']} samples",
        f"- C1: {manifest['projections']['C1']['count']} samples",
        f"- Caveat: {manifest['projections']['caveat']}",
        "", "## Limitations", "",
    ])
    lines.extend(f"- {item}" for item in manifest["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_candidate_artifacts(package: Mapping[str, Any], directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / "scientific-candidate.json"
    manifest_path = directory / "build-manifest.json"
    report_path = directory / "candidate-characterization.md"
    candidate_path.write_text(
        json.dumps(package["candidate"], indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(package["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report_path.write_text(render_candidate_report(package), encoding="utf-8")
    return candidate_path, manifest_path, report_path


def main() -> None:
    first = build_timing_nonce_candidate()
    second = build_timing_nonce_candidate()
    if first != second:
        raise CandidateBuildError("repeated candidate builds are not canonically identical")
    first["manifest"]["reproducibility"] = {
        "complete_builds_compared": 2,
        "canonically_identical": True,
        "candidate_fingerprint": first["candidate"]["candidate_fingerprint"],
        "composition_build_fingerprint": first["manifest"]["fingerprints"][
            "composition_build"
        ],
    }
    first["manifest"].pop("manifest_fingerprint")
    first["manifest"]["manifest_fingerprint"] = stable_reference(first["manifest"])
    paths = write_candidate_artifacts(
        first, Path("artifacts/icds-timing-nonce-candidate"),
    )
    print(json.dumps({
        "candidate_fingerprint": first["candidate"]["candidate_fingerprint"],
        "build_fingerprint": first["manifest"]["fingerprints"]["composition_build"],
        "manifest_fingerprint": first["manifest"]["manifest_fingerprint"],
        "paths": [str(path) for path in paths],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
