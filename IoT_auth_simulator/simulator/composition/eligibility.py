"""Build immutable role edges and conditional ordinary-control memberships."""

from __future__ import annotations

from typing import Iterable

from simulator.composition.model import (
    CONTROL_MEMBERSHIP_SCHEMA_VERSION,
    ControlMembership,
    EligibleRoleEdge,
    ORDINARY_NORMAL,
    RoleKind,
    ScientificRole,
)
from simulator.device_profiles.scientific_population import ScientificDeviceProfilePopulation
from simulator.role_integration.integrated_stress import SPECIFICATION_BY_VARIANT
from simulator.role_integration.models import ValidBaseTrace, stable_reference
from simulator.role_integration.pool import BaseOpportunityPool


CONTROL_MEMBERSHIP_PREDICATES = {
    ControlMembership.ACCESS: ("protected_access_exists",),
    ControlMembership.MULTI_ATTEMPT: ("distinct_authentication_attempts_exist",),
    ControlMembership.REPEATED_CURRENT_TOKEN: ("multiple_current_token_uses_exist",),
    ControlMembership.REPLACEMENT: (
        "current_t1_use_before_replacement_exists",
        "t2_presentation_after_replacement_exists",
    ),
    ControlMembership.IDENTITY_LIFECYCLE: ("persistent_identity_association_exists",),
    ControlMembership.BINDING: (
        "token_authenticated_identity_binding_evidence_exists",
        "token_protected_context_binding_evidence_exists",
    ),
}


def control_memberships(vector: Iterable[object]) -> tuple[str, ...]:
    applicable = {item.predicate_id for item in vector if item.applicable}
    values = [ControlMembership.BROAD_POPULATION.value]
    for membership, predicates in CONTROL_MEMBERSHIP_PREDICATES.items():
        if any(predicate in applicable for predicate in predicates):
            values.append(membership.value)
    return tuple(sorted(values))


def _structure(base: ValidBaseTrace) -> dict[str, object]:
    events = base.event_records
    times = [float(item["timestamp"]) for item in events]
    return {
        "lifecycle": base.structural_summary.lifecycle,
        "recovery": base.structural_summary.recovery,
        "renewal": base.structural_summary.renewal,
        "activity_count": base.structural_summary.activity_count,
        "authentication_attempt_count": len({
            item.get("auth_attempt_id") for item in events if item.get("auth_attempt_id")
        }),
        "token_count": len({item.get("token_id") for item in events if item.get("token_id")}),
        "access_count": sum(item.get("event_type") == "access_request" for item in events),
        "base_event_count": len(events),
        "base_duration": (max(times) - min(times)) if times else 0.0,
        "opportunity_source": base.base_construction_key,
    }


def _profile(population: ScientificDeviceProfilePopulation, base: ValidBaseTrace) -> dict[str, object]:
    assignment = population.assignment_for_device(base.device_id)
    profile = assignment.profile
    return {
        "T_AUTH_center": profile.timing.auth.center_modifier,
        "T_AUTH_dispersion": profile.timing.auth.dispersion_modifier,
        "T_TRANSITION_center": profile.timing.transition.center_modifier,
        "T_TRANSITION_dispersion": profile.timing.transition.dispersion_modifier,
        "T_ACCESS_center": profile.timing.access.center_modifier,
        "T_ACCESS_dispersion": profile.timing.access.dispersion_modifier,
        "T_RENEWAL_center": profile.timing.renewal.center_modifier,
        "T_RENEWAL_dispersion": profile.timing.renewal.dispersion_modifier,
        "A": profile.activity, "R": profile.recovery, "N": profile.renewal,
        "L_AUTH": profile.lifecycle.auth, "L_SESSION": profile.lifecycle.session,
        "L_ACCESS": profile.lifecycle.access,
        "lifecycle_cell": assignment.lifecycle_cell_id,
        "scalar_strata": dict(assignment.scalar_strata),
    }


def build_eligibility_edges(
    population: ScientificDeviceProfilePopulation,
    pool: BaseOpportunityPool,
    roles: Iterable[ScientificRole],
    schedule_seed: int,
) -> tuple[EligibleRoleEdge, ...]:
    """Materialize eligibility without transforming a base or creating support."""
    if population.population_fingerprint != pool.population_reference:
        raise ValueError("population/pool provenance mismatch")
    requested = tuple(sorted(set(roles), key=lambda item: item.key))
    edges: list[EligibleRoleEdge] = []
    for base, vector in zip(pool.bases, pool.applicability_matrix):
        by_predicate = {item.predicate_id: item for item in vector}
        memberships = control_memberships(vector)
        structure = _structure(base)
        profile = _profile(population, base)
        for role in requested:
            if role.kind is RoleKind.ORDINARY_NORMAL:
                family = "ordinary_normal"
                applicability_reference = stable_reference({
                    "base_trace_reference": base.base_trace_reference,
                    "role": role.to_record(), "validated_normal": base.validation.accepted,
                })
                eligible = base.validation.accepted
            else:
                specification = SPECIFICATION_BY_VARIANT.get(role.variant_id or "")
                if specification is None or specification.role_type != role.kind.value:
                    raise ValueError(f"role is not in frozen 4D.3 catalogue: {role.key}")
                predicate = by_predicate[specification.broad_predicate_id]
                eligible = predicate.applicable
                family = specification.adapter_family
                applicability_reference = predicate.evaluation_reference
            if not eligible:
                continue
            tie = stable_reference({
                "rng_derivation_version": "4d4b-scheduler-sha256-v1",
                "scheduler_namespace": "scientific-composition-selection",
                "schedule_seed": schedule_seed,
                "base_trace_reference": base.base_trace_reference,
                "role": role.to_record(),
            })
            edges.append(EligibleRoleEdge(
                base_trace_reference=base.base_trace_reference,
                role=role,
                device_key=base.population_device_key,
                family=family,
                applicability_reference=applicability_reference,
                control_memberships=memberships,
                structural_descriptors=structure,
                profile_descriptors=profile,
                scheduler_tie_break_reference=tie,
            ))
    return tuple(sorted(edges, key=lambda item: (item.role.key, item.base_trace_reference)))


def eligibility_digest(edges: Iterable[EligibleRoleEdge]) -> str:
    records = []
    for item in sorted(edges, key=lambda edge: (edge.role.key, edge.base_trace_reference)):
        record = item.to_record()
        # Scheduling randomness ranks an already-frozen edge; it is not part
        # of applicability/eligibility identity.
        record.pop("scheduler_tie_break_reference")
        records.append(record)
    return stable_reference({
        "schema_version": "4d4b-normalized-eligibility-digest-v1",
        "control_membership_schema_version": CONTROL_MEMBERSHIP_SCHEMA_VERSION,
        "edges": records,
    })
