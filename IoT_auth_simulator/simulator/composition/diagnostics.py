"""Privileged pre/post composition diagnostics; never detector features."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from simulator.composition.model import CompositionAssignment, EligibleRoleEdge


PRE_COMPOSITION_DIAGNOSTICS_VERSION = "4d4b-pre-composition-diagnostics-v1"
POST_COMPOSITION_DIAGNOSTICS_VERSION = "4d4b-post-composition-diagnostics-v1"
SCG_REGISTRY_VERSION = "4d4b-scg-registry-v1"

SCG_REGISTRY = {
    "lifecycle": "S+C", "recovery": "S+C", "renewal": "S+C",
    "activity_count": "S+C", "authentication_attempt_count": "S+G",
    "token_count": "S+G", "access_count": "S+G", "base_event_count": "G",
    "base_duration": "G", "profile_strata": "C",
}


def pre_composition_diagnostics(
    edges: Iterable[EligibleRoleEdge], assignments: Iterable[CompositionAssignment],
    requested: Mapping[str, int], reserves: Mapping[str, int],
) -> dict[str, Any]:
    edge_rows = tuple(edges)
    selected = tuple(assignments)
    by_role: dict[str, list[EligibleRoleEdge]] = defaultdict(list)
    for edge in edge_rows:
        by_role[edge.role.key].append(edge)
    selected_by_role: dict[str, list[CompositionAssignment]] = defaultdict(list)
    for assignment in selected:
        selected_by_role[assignment.role.key].append(assignment)
    base_degrees = Counter(edge.base_trace_reference for edge in edge_rows)
    overlaps: dict[str, int] = {}
    keys = sorted(by_role)
    for index, left in enumerate(keys):
        left_bases = {item.base_trace_reference for item in by_role[left]}
        for right in keys[index + 1:]:
            overlap = len(left_bases & {item.base_trace_reference for item in by_role[right]})
            if overlap:
                overlaps[f"{left}|{right}"] = overlap
    role_diagnostics: dict[str, Any] = {}
    for role in sorted(requested):
        eligible = by_role.get(role, [])
        chosen = selected_by_role.get(role, [])
        selected_edges = {
            item.base_trace_reference: item for item in eligible
            if any(value.base_trace_reference == item.base_trace_reference for value in chosen)
        }
        role_diagnostics[role] = {
            "requested": requested[role], "assigned": len(chosen),
            "eligible_bases": len(eligible),
            "eligible_devices": len({item.device_key for item in eligible}),
            "selected_devices": len({item.device_key for item in chosen}),
            "device_repetition": dict(sorted(Counter(item.device_key for item in chosen).items())),
            "lifecycle": dict(sorted(Counter(
                item.structural_descriptors["lifecycle"] for item in selected_edges.values()
            ).items())),
            "recovery": dict(sorted(Counter(
                item.structural_descriptors["recovery"] for item in selected_edges.values()
            ).items())),
            "renewal": dict(sorted(Counter(
                item.structural_descriptors["renewal"] for item in selected_edges.values()
            ).items())),
            "activity": dict(sorted(Counter(
                str(item.structural_descriptors["activity_count"]) for item in selected_edges.values()
            ).items())),
            "profile_cells": dict(sorted(Counter(
                item.profile_descriptors["lifecycle_cell"] for item in selected_edges.values()
            ).items())),
            "base_event_counts": [
                item.structural_descriptors["base_event_count"] for item in selected_edges.values()
            ],
            "base_durations": [
                item.structural_descriptors["base_duration"] for item in selected_edges.values()
            ],
            "attempt_token_access_counts": [
                [item.structural_descriptors[name] for name in (
                    "authentication_attempt_count", "token_count", "access_count"
                )] for item in selected_edges.values()
            ],
            "opportunity_sources": [
                item.structural_descriptors["opportunity_source"] for item in selected_edges.values()
            ],
        }
    return {
        "version": PRE_COMPOSITION_DIAGNOSTICS_VERSION,
        "authority": "privileged_scientific_composition_diagnostics",
        "requested_vs_assigned": role_diagnostics,
        "eligible_set_overlap": overlaps,
        "base_role_degree": dict(sorted(Counter(base_degrees.values()).items())),
        "ordinary_control_reserve": dict(sorted(reserves.items())),
        "scg_registry_version": SCG_REGISTRY_VERSION,
        "scg_registry": dict(SCG_REGISTRY),
        "constraints_influence_eligibility": False,
    }


def post_composition_diagnostics(samples: Iterable[object]) -> dict[str, Any]:
    rows = tuple(samples)
    by_role: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        by_role[row.assignment.role.key].append(row)
    roles: dict[str, Any] = {}
    for role, values in sorted(by_role.items()):
        observed_delays = []
        for item in values:
            observed = [
                float(event.get("observed_timestamp", event["timestamp"]))
                for event in item.event_records
            ]
            observed_delays.extend(
                right - left for left, right in zip(observed, observed[1:])
            )
        roles[role] = {
            "count": len(values),
            "final_event_counts": [len(item.event_records) for item in values],
            "event_count_deltas": [item.event_count_delta for item in values],
            "durations": [item.duration for item in values],
            "observed_delay_summary": {
                "count": len(observed_delays),
                "minimum": min(observed_delays) if observed_delays else None,
                "maximum": max(observed_delays) if observed_delays else None,
                "mean": (
                    sum(observed_delays) / len(observed_delays)
                    if observed_delays else None
                ),
            },
            "event_type_histogram": dict(sorted(Counter(
                event["event_type"] for item in values for event in item.event_records
            ).items())),
            "event_type_delta": dict(sorted(Counter(
                key for item in values for key in item.event_type_delta
            ).items())),
            "success_failure_denial_close_token_access": dict(sorted(Counter(
                label for item in values for label in item.outcome_markers
            ).items())),
            "intended_invariant_sets": [list(item.intended_invariant_ids) for item in values],
            "complete_invariant_sets": [list(item.complete_invariant_ids) for item in values],
            "collision_classes": dict(sorted(Counter(item.collision_classification for item in values).items())),
            "aggregate_gate_outcomes": dict(sorted(Counter(str(item.aggregate_gate_passed) for item in values).items())),
            "canonical_reload_valid": sum(item.canonical_reload_valid for item in values),
        }
    groups: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        groups[row.detector_evidence_hash].append(row)
    duplicates = {key: value for key, value in groups.items() if len(value) > 1}
    duplicate_records = []
    for key, values in sorted(duplicates.items()):
        truths = sorted({item.detector_truth for item in values})
        duplicate_records.append({
            "detector_evidence_hash": key, "count": len(values), "truths": truths,
            "roles": sorted(item.assignment.role.key for item in values),
            "families": sorted(item.assignment.family for item in values),
            "conflicting_truth": len(truths) > 1,
        })
    return {
        "version": POST_COMPOSITION_DIAGNOSTICS_VERSION,
        "authority": "privileged_scientific_composition_diagnostics",
        "roles": roles,
        "unique_detector_evidence_hashes": len(groups),
        "duplicate_groups": duplicate_records,
        "same_truth_duplicate_groups": sum(not item["conflicting_truth"] for item in duplicate_records),
        "conflicting_truth_duplicate_groups": sum(item["conflicting_truth"] for item in duplicate_records),
        "timing_boundary_summary": {
            "positive_margin_hn": sum(row.assignment.variant_id == "small_positive_causal_margin" for row in rows),
            "negative_timestamp_anomaly": sum(row.assignment.variant_id == "timestamp_inconsistency" for row in rows),
        },
    }
