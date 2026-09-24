"""Privileged 4D.4C.3A Timing+Nonce opportunity characterization.

This module reuses the frozen scientific population, canonical base admission,
applicability, and global-exclusive scheduling implementations. It never
invokes a role transformation and does not create a dataset.
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

from simulator.composition.eligibility import build_eligibility_edges, eligibility_digest
from simulator.composition.model import (
    CompositionSpecification,
    DuplicateEvidencePolicy,
    EligibleRoleEdge,
    ORDINARY_NORMAL,
    RoleKind,
    RoleRequest,
    ScientificRole,
)
from simulator.composition.scheduler import build_composition_plan
from simulator.config.integrated_role_stress import IntegratedRoleStressConfiguration
from simulator.device_profiles.scientific_population import scientific_configuration_record
from simulator.role_integration.direct_timing import (
    POSITIVE_MARGIN_HN,
    SAME_ATTEMPT_NONCE_HN,
    TIMESTAMP_ANOMALY,
)
from simulator.role_integration.integrated_stress import (
    SPECIFICATION_BY_VARIANT,
    build_stress_population_and_pool,
)
from simulator.role_integration.models import canonical_json, stable_reference
from simulator.role_integration.nonce_boundary import FRESH_NONCE_HN, NONCE_REUSE_ANOMALY


FEASIBILITY_CHARACTERIZATION_VERSION = "4d4c3a-timing-nonce-feasibility-v1"
FEASIBILITY_ARTIFACT_SCHEMA_VERSION = "4d4c3a-feasibility-artifact-v1"
EXPECTED_BASELINE_COMMIT = "6183fdc96774a9a33c2feb6442e93efc252b4ffb"
EXPECTED_ANALYSIS_BRANCH = "4d4c3a-timing-nonce-feasibility"


def _hn(variant_id: str) -> ScientificRole:
    return ScientificRole(RoleKind.HARD_NEGATIVE, variant_id)


def _anomaly(variant_id: str) -> ScientificRole:
    return ScientificRole(RoleKind.ANOMALY, variant_id)


TIMING_HN = _hn(POSITIVE_MARGIN_HN)
TIMING_ANOMALY = _anomaly(TIMESTAMP_ANOMALY)
SAME_ATTEMPT_NONCE_HN_ROLE = _hn(SAME_ATTEMPT_NONCE_HN)
FRESH_NONCE_HN_ROLE = _hn(FRESH_NONCE_HN)
NONCE_ANOMALY = _anomaly(NONCE_REUSE_ANOMALY)

FEASIBILITY_ROLES = (
    ORDINARY_NORMAL,
    TIMING_HN,
    TIMING_ANOMALY,
    SAME_ATTEMPT_NONCE_HN_ROLE,
    FRESH_NONCE_HN_ROLE,
    NONCE_ANOMALY,
)

PLANNING_HYPOTHESIS = {
    ORDINARY_NORMAL.key: 96,
    TIMING_HN.key: 48,
    TIMING_ANOMALY.key: 48,
    SAME_ATTEMPT_NONCE_HN_ROLE.key: 48,
    FRESH_NONCE_HN_ROLE.key: 48,
    NONCE_ANOMALY.key: 48,
}

_STRATUM_KEY = {
    "T_AUTH_center": "T.AUTH.center",
    "T_AUTH_dispersion": "T.AUTH.dispersion",
    "T_TRANSITION_center": "T.TRANSITION.center",
    "T_TRANSITION_dispersion": "T.TRANSITION.dispersion",
    "T_ACCESS_center": "T.ACCESS.center",
    "T_ACCESS_dispersion": "T.ACCESS.dispersion",
    "T_RENEWAL_center": "T.RENEWAL.center",
    "T_RENEWAL_dispersion": "T.RENEWAL.dispersion",
    "A": "A",
    "R": "R",
    "N": "N",
}


def default_feasibility_configuration() -> IntegratedRoleStressConfiguration:
    """Extend the accepted 4D.3.7 controls only to eight attempts/device."""
    return IntegratedRoleStressConfiguration(device_count=64, opportunities_per_device=8)


def planning_hypothesis_specification(
    configuration: IntegratedRoleStressConfiguration,
) -> CompositionSpecification:
    return CompositionSpecification(
        role_requests=tuple(
            RoleRequest(role, PLANNING_HYPOTHESIS[role.key]) for role in FEASIBILITY_ROLES
        ),
        schedule_seed=configuration.role_assignment_seed,
        duplicate_evidence_policy=DuplicateEvidencePolicy.REJECT_DUPLICATE_EVIDENCE,
        purpose="4d4c3a_non_final_timing_nonce_feasibility_dry_run",
    )


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "q25": None, "median": None,
                "mean": None, "q75": None, "maximum": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return float(ordered[low])
        return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))

    return {
        "minimum": float(ordered[0]), "q25": at(0.25), "median": at(0.5),
        "mean": float(statistics.fmean(ordered)), "q75": at(0.75),
        "maximum": float(ordered[-1]),
    }


def _concentration(counts: Iterable[int]) -> dict[str, float]:
    ordered = sorted((value for value in counts if value), reverse=True)
    total = sum(ordered)
    if not total:
        return {"top_1_fraction": 0.0, "top_5_fraction": 0.0, "top_10_fraction": 0.0}
    return {
        "top_1_fraction": sum(ordered[:1]) / total,
        "top_5_fraction": sum(ordered[:5]) / total,
        "top_10_fraction": sum(ordered[:10]) / total,
    }


def _value_summary(values: Iterable[float]) -> dict[str, float | None]:
    return _quantiles(tuple(float(value) for value in values))


def _role_support(
    role: ScientificRole,
    edges: Sequence[EligibleRoleEdge],
    admitted_count: int,
    device_count: int,
) -> dict[str, Any]:
    selected = [edge for edge in edges if edge.role == role]
    by_device = Counter(edge.device_key for edge in selected)
    scalar_names = tuple(_STRATUM_KEY)
    all_lifecycle_cells = {
        str(edge.profile_descriptors["lifecycle_cell"]) for edge in edges
    }
    scalar_occupancy = {
        name: dict(sorted(Counter(
            str(edge.profile_descriptors["scalar_strata"][_STRATUM_KEY[name]])
            for edge in selected
        ).items()))
        for name in scalar_names
    }
    lifecycle_occupancy = dict(sorted(Counter(
        str(edge.profile_descriptors["lifecycle_cell"]) for edge in selected
    ).items()))
    profile = {
        "scalar_stratum_occupancy": scalar_occupancy,
        "empty_scalar_strata": {
            name: [stratum for stratum in range(4) if str(stratum) not in occupancy]
            for name, occupancy in scalar_occupancy.items()
        },
        "continuous_summary": {
            name: _value_summary(edge.profile_descriptors[name] for edge in selected)
            for name in scalar_names
        },
        "lifecycle_cell_occupancy": lifecycle_occupancy,
        "empty_lifecycle_cells": sorted(all_lifecycle_cells - set(lifecycle_occupancy)),
        "distinct_lifecycle_cells": len({
            edge.profile_descriptors["lifecycle_cell"] for edge in selected
        }),
        "lifecycle_coordinate_summary": {
            name: _value_summary(edge.profile_descriptors[name] for edge in selected)
            for name in ("L_AUTH", "L_SESSION", "L_ACCESS")
        },
    }
    structure = {
        name: dict(sorted(Counter(
            str(edge.structural_descriptors[name]) for edge in selected
        ).items()))
        for name in ("lifecycle", "recovery", "renewal", "activity_count")
    }
    structure.update({
        name: _value_summary(edge.structural_descriptors[name] for edge in selected)
        for name in ("authentication_attempt_count", "token_count", "access_count",
                     "base_event_count", "base_duration")
    })
    structure["cross_attempt_source"] = dict(sorted(Counter(
        (
            "retry_and_renewal"
            if edge.structural_descriptors["recovery"] == "ONE_RETRY"
            and edge.structural_descriptors["renewal"] == "RENEWAL"
            else "retry_only"
            if edge.structural_descriptors["recovery"] == "ONE_RETRY"
            else "renewal_only"
            if edge.structural_descriptors["renewal"] == "RENEWAL"
            else "none"
        )
        for edge in selected
    ).items()))
    return {
        "role": role.to_record(),
        "eligible_bases": len(selected),
        "eligible_base_fraction": len(selected) / admitted_count if admitted_count else 0.0,
        "eligible_devices": len(by_device),
        "eligible_device_fraction": len(by_device) / device_count if device_count else 0.0,
        "eligible_opportunities_per_supporting_device": _quantiles(tuple(by_device.values())),
        "concentration": _concentration(by_device.values()),
        "profile_support": profile,
        "structural_support": structure,
    }


def _overlap(
    left: ScientificRole,
    right: ScientificRole,
    by_role: Mapping[str, Sequence[EligibleRoleEdge]],
) -> dict[str, Any]:
    left_edges, right_edges = by_role[left.key], by_role[right.key]
    left_bases = {edge.base_trace_reference for edge in left_edges}
    right_bases = {edge.base_trace_reference for edge in right_edges}
    left_devices = {edge.device_key for edge in left_edges}
    right_devices = {edge.device_key for edge in right_edges}
    return {
        "left_role": left.key, "right_role": right.key,
        "bases": {
            "intersection": len(left_bases & right_bases),
            "union": len(left_bases | right_bases),
            "left_only": len(left_bases - right_bases),
            "right_only": len(right_bases - left_bases),
        },
        "devices": {
            "intersection": len(left_devices & right_devices),
            "union": len(left_devices | right_devices),
            "left_only": len(left_devices - right_devices),
            "right_only": len(right_devices - left_devices),
        },
    }


def _ordinary_control_support(
    device_keys: Sequence[str],
    by_role: Mapping[str, Sequence[EligibleRoleEdge]],
) -> dict[str, Any]:
    timing = {edge.base_trace_reference for edge in by_role[TIMING_HN.key]}
    nonce = {
        edge.base_trace_reference
        for role in (SAME_ATTEMPT_NONCE_HN_ROLE, FRESH_NONCE_HN_ROLE, NONCE_ANOMALY)
        for edge in by_role[role.key]
    }
    all_bases = {edge.base_trace_reference for edge in by_role[ORDINARY_NORMAL.key]}
    reference_to_device = {
        edge.base_trace_reference: edge.device_key for edge in by_role[ORDINARY_NORMAL.key]
    }

    def record(references: set[str]) -> dict[str, int]:
        return {
            "bases": len(references),
            "devices": len({reference_to_device[reference] for reference in references}),
        }

    return {
        "timing_applicable": record(timing),
        "nonce_applicable_any": record(nonce),
        "both": record(timing & nonce),
        "timing_only": record(timing - nonce),
        "nonce_only": record(nonce - timing),
        "neither": record(all_bases - timing - nonce),
        "population_devices": len(device_keys),
        "membership_policy": (
            "privileged applicability/control membership; one base may carry multiple "
            "memberships without row duplication"
        ),
    }


def _selected_role_diagnostics(plan: Any) -> dict[str, Any]:
    by_role: dict[str, list[Any]] = defaultdict(list)
    for assignment in plan.assignments:
        by_role[assignment.role.key].append(assignment)
    result: dict[str, Any] = {}
    for role in FEASIBILITY_ROLES:
        assignments = by_role.get(role.key, [])
        device_counts = Counter(item.device_key for item in assignments)
        result[role.key] = {
            "assigned": len(assignments),
            "distinct_devices": len(device_counts),
            "samples_per_device": _quantiles(tuple(device_counts.values())),
            "maximum_from_one_device": max(device_counts.values(), default=0),
            "concentration": _concentration(device_counts.values()),
        }
    return result


def _selected_device_overlap(plan: Any) -> dict[str, int]:
    devices: dict[str, set[str]] = defaultdict(set)
    for assignment in plan.assignments:
        devices[assignment.role.key].add(assignment.device_key)

    def intersection(left: ScientificRole, right: ScientificRole) -> int:
        return len(devices[left.key] & devices[right.key])

    return {
        "timing_anomaly_vs_timing_hn": intersection(TIMING_ANOMALY, TIMING_HN),
        "nonce_anomaly_vs_same_attempt_hn": intersection(
            NONCE_ANOMALY, SAME_ATTEMPT_NONCE_HN_ROLE,
        ),
        "nonce_anomaly_vs_fresh_cross_attempt_hn": intersection(
            NONCE_ANOMALY, FRESH_NONCE_HN_ROLE,
        ),
        "ordinary_vs_timing_anomaly": intersection(ORDINARY_NORMAL, TIMING_ANOMALY),
        "ordinary_vs_nonce_anomaly": intersection(ORDINARY_NORMAL, NONCE_ANOMALY),
    }


def _rejection_summary(item: Any) -> dict[str, Any]:
    """Retain canonical failure facts without runtime trace UUIDs."""
    evidence = json.loads(item.validation_evidence_json)
    return {
        "base_construction_key": item.base_construction_key,
        "population_device_key": item.population_device_key,
        "behavioral_profile_id": item.behavioral_profile_id,
        "validation_status": item.validation_status,
        "rejection_stage": item.rejection_stage,
        "reason": item.reason,
        "rejection_reference": item.rejection_reference,
        "canonical_serialization_version": evidence["canonical_serialization_version"],
        "normalized_failures": evidence["failures"],
    }


def characterize_timing_nonce_feasibility(
    configuration: IntegratedRoleStressConfiguration | None = None,
) -> dict[str, Any]:
    """Build one immutable opportunity graph and dry-run the planning hypothesis."""
    configuration = configuration or default_feasibility_configuration()
    state = random.getstate()
    try:
        population, pool = build_stress_population_and_pool(configuration)
        scientific_population = population.behavioral_profiles
        population_before = scientific_population.population_fingerprint
        pool_before = pool.pool_fingerprint
        base_before = tuple(base.to_record() for base in pool.bases)

        edges = build_eligibility_edges(
            scientific_population, pool, FEASIBILITY_ROLES,
            configuration.role_assignment_seed,
        )
        digest = eligibility_digest(edges)
        by_role: dict[str, list[EligibleRoleEdge]] = defaultdict(list)
        for edge in edges:
            by_role[edge.role.key].append(edge)

        device_keys = tuple(sorted(
            assignment.population_device_key for assignment in scientific_population.assignments
        ))
        matrix = []
        for device_key in device_keys:
            counts = {
                role.key: sum(edge.device_key == device_key for edge in by_role[role.key])
                for role in FEASIBILITY_ROLES
            }
            matrix.append({
                "population_device_key": device_key,
                "eligible_base_counts": counts,
                "has_eligible_base": {key: bool(value) for key, value in counts.items()},
            })

        support = {
            role.key: _role_support(role, edges, len(pool.bases), configuration.device_count)
            for role in FEASIBILITY_ROLES
        }
        overlaps = {
            "timing_hn_vs_anomaly": _overlap(TIMING_HN, TIMING_ANOMALY, by_role),
            "nonce_same_attempt_vs_fresh": _overlap(
                SAME_ATTEMPT_NONCE_HN_ROLE, FRESH_NONCE_HN_ROLE, by_role,
            ),
            "nonce_same_attempt_vs_reuse": _overlap(
                SAME_ATTEMPT_NONCE_HN_ROLE, NONCE_ANOMALY, by_role,
            ),
            "nonce_fresh_vs_reuse": _overlap(FRESH_NONCE_HN_ROLE, NONCE_ANOMALY, by_role),
        }
        predicate_relationships = {
            role.key: (
                "validated_normal"
                if role is ORDINARY_NORMAL
                else SPECIFICATION_BY_VARIANT[role.variant_id].broad_predicate_id
            )
            for role in FEASIBILITY_ROLES
        }

        specification = planning_hypothesis_specification(configuration)
        plan = build_composition_plan(scientific_population, pool, specification)
        dry_run = {
            "status": plan.feasibility_status,
            "requested_counts": dict(PLANNING_HYPOTHESIS),
            "assigned_counts": dict(sorted(Counter(
                item.role.key for item in plan.assignments
            ).items())),
            "assigned_total": len(plan.assignments),
            "unassigned_bases": len(plan.unassigned_base_references),
            "composition_fingerprint": plan.composition_fingerprint,
            "eligibility_digest": plan.eligibility_digest,
            "role_device_diagnostics": _selected_role_diagnostics(plan),
            "selected_device_overlap": _selected_device_overlap(plan),
            "infeasibility": plan.infeasibility.to_record() if plan.infeasibility else None,
            "transformations_invoked": False,
            "final_candidate_generated": False,
        }

        record: dict[str, Any] = {
            "schema_version": FEASIBILITY_ARTIFACT_SCHEMA_VERSION,
            "characterization_version": FEASIBILITY_CHARACTERIZATION_VERSION,
            "authority": "privileged_pre_composition_opportunity_characterization",
            "repository_checkpoint": {
                "analysis_branch": EXPECTED_ANALYSIS_BRANCH,
                "baseline_commit": EXPECTED_BASELINE_COMMIT,
            },
            "configuration": {
                "integrated_configuration": asdict(configuration),
                "scientific_population_configuration": scientific_configuration_record(
                    configuration.population_configuration()
                ),
                "attempted_opportunities": (
                    configuration.device_count * configuration.opportunities_per_device
                ),
                "roles": [role.to_record() for role in FEASIBILITY_ROLES],
            },
            "canonical_admission": {
                "attempted": len(pool.bases) + len(pool.rejected),
                "admitted": len(pool.bases),
                "rejected": len(pool.rejected),
                "rejection_references": [item.rejection_reference for item in pool.rejected],
                "rejections": [_rejection_summary(item) for item in pool.rejected],
                "replacement_or_resampling": False,
            },
            "fingerprints": {
                "population": scientific_population.population_fingerprint,
                "opportunity_pool": pool.pool_fingerprint,
                "applicability": digest,
                "dry_run_composition": plan.composition_fingerprint,
            },
            "population_coverage": {
                "diagnostics": dict(scientific_population.diagnostics),
                "duplicate_behavioral_profile_ids": (
                    len(scientific_population.assignments)
                    - len({item.profile.behavioral_profile_id
                           for item in scientific_population.assignments})
                ),
            },
            "role_opportunity_summary": support,
            "device_role_support": matrix,
            "applicability_predicate_relationships": {
                "role_to_frozen_predicate": predicate_relationships,
                "same_predicate_role_groups": [
                    [TIMING_HN.key, TIMING_ANOMALY.key,
                     SAME_ATTEMPT_NONCE_HN_ROLE.key],
                    [FRESH_NONCE_HN_ROLE.key, NONCE_ANOMALY.key],
                ],
                "semantic_logic_redefined": False,
            },
            "cross_role_overlap": overlaps,
            "ordinary_control_support": _ordinary_control_support(device_keys, by_role),
            "counterfactual_linkage": {
                "sufficient": all(
                    base.base_trace_reference and base.base_construction_key
                    and base.opportunity_randomness_reference for base in pool.bases
                ),
                "base_identity": "base_trace_reference",
                "construction_identity": "base_construction_key",
                "opportunity_identity": "opportunity_randomness_reference",
                "siblings_generated": False,
            },
            "planning_hypothesis_dry_run": dry_run,
            "integrity": {
                "population_immutable": (
                    scientific_population.population_fingerprint == population_before
                ),
                "pool_immutable": pool.pool_fingerprint == pool_before,
                "bases_immutable": tuple(base.to_record() for base in pool.bases) == base_before,
                "global_rng_unchanged": random.getstate() == state,
                "profile_metadata_privileged": True,
                "detector_schema_modified": False,
                "adapters_invoked": False,
                "final_candidate_generated": False,
            },
            "limitations": [
                "descriptive controlled-synthetic support, not statistical sufficiency",
                "not real-world representative and not empirically calibrated",
                "does not define an ML split or prediction horizon",
                "does not define final detector inputs",
                "does not select final role counts or prevalence",
                "does not generate the final ICDS candidate",
                "does not establish operational anomaly-detection performance",
            ],
        }
        record["characterization_fingerprint"] = stable_reference(record)
        return record
    finally:
        if random.getstate() != state:
            random.setstate(state)
            raise AssertionError("feasibility characterization perturbed global RNG state")


def render_characterization_markdown(report: Mapping[str, Any]) -> str:
    admission = report["canonical_admission"]
    dry_run = report["planning_hypothesis_dry_run"]
    config = report["configuration"]["integrated_configuration"]
    lines = [
        "# 4D.4C.3A Timing+Nonce feasibility characterization", "",
        "Privileged, pre-composition controlled-synthetic analysis. This is not a dataset.",
        "", "## Configuration and admission", "",
        f"- Population: {config['device_count']} devices",
        f"- Attempts/device: {config['opportunities_per_device']}",
        f"- Attempted/admitted/rejected: {admission['attempted']}/{admission['admitted']}/{admission['rejected']}",
        f"- Population fingerprint: `{report['fingerprints']['population']}`",
        f"- Pool fingerprint: `{report['fingerprints']['opportunity_pool']}`",
        f"- Applicability digest: `{report['fingerprints']['applicability']}`",
        "", "## Role support", "",
        "| Role | Eligible bases | Pool % | Devices | Population % | Per-device min/median/mean/max | Top-1 / top-5 / top-10 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for role in FEASIBILITY_ROLES:
        item = report["role_opportunity_summary"][role.key]
        per_device = item["eligible_opportunities_per_supporting_device"]
        concentration = item["concentration"]
        lines.append(
            f"| `{role.key}` | {item['eligible_bases']} | "
            f"{100 * item['eligible_base_fraction']:.2f}% | {item['eligible_devices']} | "
            f"{100 * item['eligible_device_fraction']:.2f}% | "
            f"{per_device['minimum']:.0f}/{per_device['median']:.1f}/"
            f"{per_device['mean']:.2f}/{per_device['maximum']:.0f} | "
            f"{100 * concentration['top_1_fraction']:.2f}% / "
            f"{100 * concentration['top_5_fraction']:.2f}% / "
            f"{100 * concentration['top_10_fraction']:.2f}% |"
        )
    lines.extend([
        "", "## Applicability overlap and ordinary controls", "",
        (
            "Timing HN, Timing anomaly, and same-attempt nonce HN share the frozen "
            "same-attempt nonce-evidence predicate. Fresh cross-attempt nonce HN and "
            "nonce-reuse anomaly share the frozen distinct-attempt predicate."
        ),
        "",
    ])
    timing_overlap = report["cross_role_overlap"]["timing_hn_vs_anomaly"]
    nonce_overlap = report["cross_role_overlap"]["nonce_fresh_vs_reuse"]
    controls = report["ordinary_control_support"]
    lines.extend([
        f"- Timing HN/anomaly overlap: {timing_overlap['bases']['intersection']} bases, "
        f"{timing_overlap['devices']['intersection']} devices.",
        f"- Fresh-nonce/reuse overlap: {nonce_overlap['bases']['intersection']} bases, "
        f"{nonce_overlap['devices']['intersection']} devices.",
        f"- Timing-applicable ordinary controls: {controls['timing_applicable']['bases']} "
        f"bases across {controls['timing_applicable']['devices']} devices.",
        f"- Any nonce-applicable ordinary controls: {controls['nonce_applicable_any']['bases']} "
        f"bases across {controls['nonce_applicable_any']['devices']} devices.",
        "- Control memberships are privileged and do not duplicate observable rows.",
        "", "## Planning-hypothesis dry run", "",
        f"- Status: **{dry_run['status']}**",
        f"- Assigned: {dry_run['assigned_total']} of {sum(PLANNING_HYPOTHESIS.values())}",
        f"- Composition fingerprint: `{dry_run['composition_fingerprint']}`",
        "- No adapter ran, no transformation was applied, and no final candidate was generated.",
        "", "| Role | Assigned | Distinct devices | Maximum/device | Top-device concentration |",
        "|---|---:|---:|---:|---:|",
    ])
    for role in FEASIBILITY_ROLES:
        item = dry_run["role_device_diagnostics"][role.key]
        lines.append(
            f"| `{role.key}` | {item['assigned']} | {item['distinct_devices']} | "
            f"{item['maximum_from_one_device']} | "
            f"{100 * item['concentration']['top_1_fraction']:.2f}% |"
        )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend([
        "",
        "The adjacent JSON artifact contains the complete 64-device support matrix, "
        "profile/structural summaries, overlaps, controls, and dry-run diagnostics.", "",
    ])
    return "\n".join(lines)


def write_characterization_artifacts(
    report: Mapping[str, Any], directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "4d4c3a-timing-nonce-feasibility.json"
    markdown_path = directory / "4d4c3a-timing-nonce-feasibility.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_characterization_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    report = characterize_timing_nonce_feasibility()
    write_characterization_artifacts(report, Path("audits"))
    print(canonical_json({
        "characterization_fingerprint": report["characterization_fingerprint"],
        "status": report["planning_hypothesis_dry_run"]["status"],
        "artifacts": [
            "audits/4d4c3a-timing-nonce-feasibility.json",
            "audits/4d4c3a-timing-nonce-feasibility.md",
        ],
    }))


if __name__ == "__main__":
    main()
