"""Deterministic global exclusive allocator for frozen base opportunities."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from simulator.composition.diagnostics import pre_composition_diagnostics
from simulator.composition.eligibility import build_eligibility_edges, eligibility_digest
from simulator.composition.fingerprint import composition_fingerprint
from simulator.composition.model import (
    CompositionAssignment,
    CompositionInfeasibilityReport,
    CompositionPlan,
    CompositionSpecification,
    EligibleRoleEdge,
    ORDINARY_NORMAL,
    RoleRequest,
    SCIENTIFIC_ROLE_SCHEMA_VERSION,
)
from simulator.device_profiles.scientific_population import ScientificDeviceProfilePopulation
from simulator.role_integration.models import canonical_json, stable_reference
from simulator.role_integration.pool import BaseOpportunityPool


SCHEDULER_ALGORITHM_VERSION = "4d4b-global-augmenting-allocation-v1"
SCHEDULER_RNG_DERIVATION_VERSION = "4d4b-scheduler-sha256-v1"


@dataclass(frozen=True)
class _Slot:
    slot_id: str
    request: RoleRequest
    required_membership: Optional[str]
    coverage_slot: bool


def _slots(specification: CompositionSpecification) -> tuple[_Slot, ...]:
    reserves = tuple(specification.ordinary_control_reserves)
    ordinary = next((item for item in specification.role_requests if item.role == ORDINARY_NORMAL), None)
    reserve_total = sum(item.minimum_ordinary_count for item in reserves)
    if reserve_total and (ordinary is None or reserve_total > ordinary.count):
        return ()
    result: list[_Slot] = []
    for request in specification.role_requests:
        ordinal = 0
        if request.role == ORDINARY_NORMAL:
            for reserve in sorted(reserves, key=lambda item: item.membership.value):
                for _ in range(reserve.minimum_ordinary_count):
                    ordinal += 1
                    result.append(_Slot(
                        f"{request.role.key}/slot-{ordinal:06d}", request,
                        reserve.membership.value,
                        ordinal <= request.minimum_distinct_devices,
                    ))
        while ordinal < request.count:
            ordinal += 1
            result.append(_Slot(
                f"{request.role.key}/slot-{ordinal:06d}", request, None,
                ordinal <= request.minimum_distinct_devices,
            ))
    return tuple(result)


def _matches_declared_constraints(edge: EligibleRoleEdge, specification: CompositionSpecification) -> bool:
    for name, value in specification.structural_matching.items():
        if str(edge.structural_descriptors.get(name)) != str(value):
            return False
    for name, value in specification.profile_balancing.items():
        if str(edge.profile_descriptors.get(name)) != str(value):
            return False
    return True


def _candidate_order(slot: _Slot, edges: Iterable[EligibleRoleEdge]) -> tuple[EligibleRoleEdge, ...]:
    filtered = [
        item for item in edges
        if item.role == slot.request.role
        and (slot.required_membership is None or slot.required_membership in item.control_memberships)
    ]
    by_device: dict[str, list[EligibleRoleEdge]] = defaultdict(list)
    for edge in filtered:
        by_device[edge.device_key].append(edge)
    positions: dict[str, int] = {}
    for device, values in by_device.items():
        for position, edge in enumerate(sorted(values, key=lambda item: item.scheduler_tie_break_reference)):
            positions[edge.base_trace_reference] = position
    return tuple(sorted(filtered, key=lambda item: (
        positions[item.base_trace_reference],
        stable_reference({"slot": slot.slot_id, "device": item.device_key,
                          "tie": item.scheduler_tie_break_reference}),
        item.base_trace_reference,
    )))


def _allocate(slots: tuple[_Slot, ...], edges: tuple[EligibleRoleEdge, ...]):
    edge_by_pair = {(item.role.key, item.base_trace_reference): item for item in edges}
    candidates = {slot.slot_id: _candidate_order(slot, edges) for slot in slots}
    # Complete-problem scarcity ordering is internal only; assignment identity
    # is normalized and does not depend on caller/catalogue traversal order.
    ordered_slots = tuple(sorted(slots, key=lambda item: (
        len(candidates[item.slot_id]), item.request.role.key, item.slot_id,
    )))
    slot_by_id = {item.slot_id: item for item in slots}
    slot_to_base: dict[str, str] = {}
    base_to_slot: dict[str, str] = {}

    def permitted(slot: _Slot, edge: EligibleRoleEdge) -> bool:
        same_role = [
            (slot_by_id[key], edge_by_pair[(slot_by_id[key].request.role.key, base_ref)])
            for key, base_ref in slot_to_base.items()
            if key != slot.slot_id and slot_by_id[key].request.role == slot.request.role
        ]
        if slot.coverage_slot and any(
            other.coverage_slot and chosen.device_key == edge.device_key
            for other, chosen in same_role
        ):
            return False
        maximum = slot.request.maximum_per_device
        if maximum is not None and sum(
            chosen.device_key == edge.device_key for _other, chosen in same_role
        ) >= maximum:
            return False
        return True

    def place(slot: _Slot, visited_bases: set[str], visited_slots: set[str]) -> bool:
        if slot.slot_id in visited_slots:
            return False
        visited_slots.add(slot.slot_id)
        for edge in candidates[slot.slot_id]:
            base_ref = edge.base_trace_reference
            if base_ref in visited_bases or not permitted(slot, edge):
                continue
            visited_bases.add(base_ref)
            occupant_id = base_to_slot.get(base_ref)
            if occupant_id is None or place(slot_by_id[occupant_id], visited_bases, visited_slots):
                old = slot_to_base.get(slot.slot_id)
                if old is not None:
                    base_to_slot.pop(old, None)
                slot_to_base[slot.slot_id] = base_ref
                base_to_slot[base_ref] = slot.slot_id
                return True
        return False

    for slot in ordered_slots:
        place(slot, set(), set())
    return slot_to_base, candidates, slot_by_id, edge_by_pair


def _infeasibility(
    specification: CompositionSpecification, edges: tuple[EligibleRoleEdge, ...],
    slot_to_base: dict[str, str], candidates: dict[str, tuple[EligibleRoleEdge, ...]],
    slot_by_id: dict[str, _Slot],
) -> CompositionInfeasibilityReport:
    requested = {item.role.key: item.count for item in specification.role_requests}
    by_role: dict[str, set[str]] = defaultdict(set)
    devices: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        by_role[edge.role.key].add(edge.base_trace_reference)
        devices[edge.role.key].add(edge.device_key)
    matched = Counter(slot_by_id[key].request.role.key for key in slot_to_base)
    overlaps: dict[str, int] = {}
    keys = sorted(requested)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            value = len(by_role[left] & by_role[right])
            if value:
                overlaps[f"{left}|{right}"] = value
    selected = set(slot_to_base.values())
    blocked = {
        role: len(by_role[role] & selected) - matched.get(role, 0) for role in keys
    }
    reserve_conflicts = []
    ordinary = requested.get(ORDINARY_NORMAL.key, 0)
    if sum(item.minimum_ordinary_count for item in specification.ordinary_control_reserves) > ordinary:
        reserve_conflicts.append("ordinary control reserves exceed requested ordinary-normal count")
    unsatisfied = [
        f"{role}: requested {count}, globally assigned {matched.get(role, 0)}"
        for role, count in requested.items() if matched.get(role, 0) != count
    ]
    for request in specification.role_requests:
        chosen_devices = {
            next(edge.device_key for edge in candidates[slot_id]
                 if edge.base_trace_reference == base_ref)
            for slot_id, base_ref in slot_to_base.items()
            if slot_by_id[slot_id].request.role == request.role
        }
        if len(chosen_devices) < request.minimum_distinct_devices:
            unsatisfied.append(
                f"{request.role.key}: selected {len(chosen_devices)} distinct devices; "
                f"minimum is {request.minimum_distinct_devices}"
            )
    return CompositionInfeasibilityReport(
        requested_counts=requested,
        maximum_immediately_eligible_support={key: len(by_role[key]) for key in keys},
        distinct_eligible_devices={key: len(devices[key]) for key in keys},
        eligible_overlap=overlaps,
        bases_blocked_by_exclusivity=blocked,
        reserve_conflicts=tuple(reserve_conflicts),
        structural_conflicts=tuple(
            f"declared exact structural constraint: {key}={value}"
            for key, value in specification.structural_matching.items()
        ),
        profile_conflicts=tuple(
            f"declared exact profile stratum constraint: {key}={value}"
            for key, value in specification.profile_balancing.items()
        ),
        unsatisfied_constraints=tuple(unsatisfied),
        matched_counts=dict(matched),
    )


def build_composition_plan(
    population: ScientificDeviceProfilePopulation,
    pool: BaseOpportunityPool,
    specification: CompositionSpecification,
) -> CompositionPlan:
    """Freeze a plan before any family transformation is invoked."""
    state = random.getstate()
    try:
        roles = tuple(item.role for item in specification.role_requests)
        edges = tuple(
            edge for edge in build_eligibility_edges(
                population, pool, roles, specification.schedule_seed,
            ) if _matches_declared_constraints(edge, specification)
        )
        digest = eligibility_digest(edges)
        slots = _slots(specification)
        if not slots and any(item.count for item in specification.role_requests):
            slot_to_base, candidates, slot_by_id, edge_by_pair = {}, {}, {}, {}
        else:
            slot_to_base, candidates, slot_by_id, edge_by_pair = _allocate(slots, edges)
        requested_total = sum(item.count for item in specification.role_requests)
        provisional_feasible = len(slot_to_base) == requested_total
        assignments: list[CompositionAssignment] = []
        for slot_id, base_ref in slot_to_base.items():
            slot = slot_by_id[slot_id]
            edge = edge_by_pair[(slot.request.role.key, base_ref)]
            assignments.append(CompositionAssignment(
                base_trace_reference=base_ref, role=edge.role,
                variant_id=edge.role.variant_id, family=edge.family,
                device_key=edge.device_key,
                applicability_reference=edge.applicability_reference,
                control_memberships=edge.control_memberships,
                structural_stratum_reference=stable_reference(dict(edge.structural_descriptors)),
                profile_stratum_reference=stable_reference(dict(edge.profile_descriptors)),
                scheduler_tie_break_reference=edge.scheduler_tie_break_reference,
                rationale=("eligible global exclusive assignment; semantics before balance; "
                           "device breadth before deterministic tie-break"),
            ))
        assignments.sort(key=lambda item: (item.role.key, item.base_trace_reference))
        matched_devices = defaultdict(set)
        for item in assignments:
            matched_devices[item.role.key].add(item.device_key)
        feasible = provisional_feasible and all(
            len(matched_devices[request.role.key]) >= request.minimum_distinct_devices
            for request in specification.role_requests
        )
        report = None if feasible else _infeasibility(
            specification, edges, slot_to_base, candidates, slot_by_id,
        )
        requested = {item.role.key: item.count for item in specification.role_requests}
        reserves = {
            item.membership.value: item.minimum_ordinary_count
            for item in specification.ordinary_control_reserves
        }
        diagnostics = pre_composition_diagnostics(edges, assignments, requested, reserves)
        unassigned = tuple(sorted(
            {base.base_trace_reference for base in pool.bases}
            - {item.base_trace_reference for item in assignments}
        ))
        fingerprint_payload = {
            "population_fingerprint": population.population_fingerprint,
            "pool_fingerprint": pool.pool_fingerprint,
            "composition_contract_version": specification.specification_version,
            "scientific_role_schema_version": SCIENTIFIC_ROLE_SCHEMA_VERSION,
            "scheduler_version": SCHEDULER_ALGORITHM_VERSION,
            "scheduling_seed": specification.schedule_seed,
            "rng_derivation_version": SCHEDULER_RNG_DERIVATION_VERSION,
            "specification": specification.to_record(),
            "eligibility_digest": digest,
            "selected_assignments": [item.to_record() for item in assignments],
            "ordinary_control_reserve": reserves,
            "planned_counts": dict(sorted(Counter(item.role.key for item in assignments).items())),
            "feasibility_status": "FEASIBLE" if feasible else "INFEASIBLE",
        }
        return CompositionPlan(
            specification=specification,
            population_fingerprint=population.population_fingerprint,
            pool_fingerprint=pool.pool_fingerprint,
            scheduler_version=SCHEDULER_ALGORITHM_VERSION,
            rng_derivation_version=SCHEDULER_RNG_DERIVATION_VERSION,
            eligibility_digest=digest,
            assignments=tuple(assignments),
            unassigned_base_references=unassigned,
            feasibility_status="FEASIBLE" if feasible else "INFEASIBLE",
            pre_transformation_diagnostics_json=canonical_json(diagnostics),
            composition_fingerprint=composition_fingerprint(fingerprint_payload),
            infeasibility=report,
        )
    finally:
        if random.getstate() != state:
            random.setstate(state)
            raise AssertionError("composition planning perturbed global RNG state")
