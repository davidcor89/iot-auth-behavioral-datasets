"""Execute a frozen feasible composition plan through frozen 4D.3 adapters."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from simulator.composition.diagnostics import post_composition_diagnostics
from simulator.composition.fingerprint import detector_evidence_hash
from simulator.composition.model import (
    CompositionAssignment,
    CompositionPlan,
    DuplicateEvidencePolicy,
    RoleKind,
)
from simulator.device_profiles.scientific_population import ScientificDeviceProfilePopulation
from simulator.role_integration.integrated_stress import (
    SPECIFICATION_BY_VARIANT,
    _broad,
    _integrate,
    _outcome,
)
from simulator.role_integration.models import ValidBaseTrace, canonical_json, stable_reference
from simulator.role_integration.pool import BaseOpportunityPool
from simulator.role_integration.transformation import TransformationStatus


ADAPTER_DISPATCH_REGISTRY_VERSION = "4d4b-frozen-4d3-adapter-dispatch-v1"
ADAPTER_DISPATCH_REGISTRY = {
    "direct_timing": "role_integration.integrated_stress._integrate/direct_timing",
    "access": "role_integration.integrated_stress._integrate/access",
    "nonce": "role_integration.integrated_stress._integrate/nonce",
    "token": "role_integration.integrated_stress._integrate/token",
    "identity_binding": "role_integration.integrated_stress._integrate/identity_binding",
}
COMPOSITION_BUILD_VERSION = "4d4b-canonical-composition-build-v1"
COMPOSITION_SAMPLE_SCHEMA_VERSION = "4d4b-scientific-sample-v1"
COMPOSITION_BUILD_RESULT_SCHEMA_VERSION = "4d4b-composition-build-result-v1"


class CompositionBuildError(RuntimeError):
    """A frozen assignment failed; no replacement or resampling is attempted."""

    def __init__(self, message: str, diagnostics: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class CompositionIntegrityError(CompositionBuildError):
    """Canonical detector evidence conflicts with the composition truth."""


@dataclass(frozen=True)
class ScientificSample:
    assignment: CompositionAssignment
    detector_truth: str
    event_records: tuple[Mapping[str, Any], ...]
    detector_evidence_hash: str
    intended_invariant_ids: tuple[str, ...]
    complete_invariant_ids: tuple[str, ...]
    semantic_qualifications: tuple[str, ...]
    collision_classification: str
    aggregate_gate_passed: bool
    canonical_reload_valid: bool
    event_count_delta: int
    event_type_delta: tuple[str, ...]
    duration: float
    outcome_markers: tuple[str, ...]
    provenance_reference: str
    schema_version: str = COMPOSITION_SAMPLE_SCHEMA_VERSION


@dataclass(frozen=True)
class CompositionBuildResult:
    plan_fingerprint: str
    transformation_seed: int
    samples: tuple[ScientificSample, ...]
    post_diagnostics_json: str
    build_fingerprint: str
    status: str = "BUILT"
    schema_version: str = COMPOSITION_BUILD_RESULT_SCHEMA_VERSION

    @property
    def diagnostics(self) -> dict[str, Any]:
        import json
        return json.loads(self.post_diagnostics_json)


def enforce_detector_evidence_integrity(
    samples: tuple[ScientificSample, ...] | tuple[object, ...],
    policy: DuplicateEvidencePolicy,
) -> dict[str, Any]:
    """Apply the declared policy without deduplication or replacement.

    Conflicting truth is an unconditional integrity failure. Same-truth
    duplicates may be reported for engineering characterization, while strict
    scientific composition can reject them explicitly.
    """
    diagnostics = post_composition_diagnostics(samples)
    if diagnostics["conflicting_truth_duplicate_groups"]:
        raise CompositionIntegrityError(
            "identical canonical detector-observation evidence carries incompatible truth",
            diagnostics,
        )
    if (
        policy is DuplicateEvidencePolicy.REJECT_DUPLICATE_EVIDENCE
        and diagnostics["same_truth_duplicate_groups"]
    ):
        raise CompositionIntegrityError(
            "strict policy rejects duplicate canonical detector-observation evidence",
            diagnostics,
        )
    return diagnostics


def _event_type_delta(base: ValidBaseTrace, records: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    left = Counter(item["event_type"] for item in base.event_records)
    right = Counter(item["event_type"] for item in records)
    values = []
    for name in sorted(set(left) | set(right)):
        delta = right[name] - left[name]
        if delta:
            values.extend([f"{name}:{delta:+d}"] * abs(delta))
    return tuple(values)


def _markers(records: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    values: list[str] = []
    for item in records:
        result = str(item.get("result", ""))
        event_type = str(item.get("event_type", ""))
        if result in {"success", "failure", "rejected"}:
            values.append(result)
        if event_type == "access_denied":
            values.append("denial")
        if event_type in {"session_closed", "disconnect"}:
            values.append("close")
        if item.get("token_id"):
            values.append("token_use")
        if event_type == "access_request":
            values.append("access_request")
    return tuple(values)


def _sample(
    assignment: CompositionAssignment, base: ValidBaseTrace,
    transformation_seed: int,
) -> ScientificSample:
    if assignment.role.kind is RoleKind.ORDINARY_NORMAL:
        records = base.event_records
        intended: tuple[str, ...] = ()
        complete: tuple[str, ...] = ()
        qualifications = tuple(assignment.control_memberships)
        collision = "none"
        gate = base.validation.build_gate_passed
        canonical = base.validation.accepted
        event_delta = 0
        detector_truth = "normal"
    else:
        variant = assignment.variant_id
        assert variant is not None
        broad, broad_ref = _broad(base, variant)
        result = _integrate(base, variant, transformation_seed)
        if result.status is not TransformationStatus.APPLIED:
            raise CompositionBuildError(
                f"assigned role failed without replacement: {assignment.role.key}: {result.reason}",
                {"base_trace_reference": base.base_trace_reference,
                 "status": result.status.value, "adapter_dispatch_version": ADAPTER_DISPATCH_REGISTRY_VERSION},
            )
        outcome = _outcome(
            base, variant, result, broad, broad_ref,
            expected_profile_id=base.behavioral_profile_id,
            expected_population_device_key=base.population_device_key,
        )
        records = result.event_records
        intended = outcome.intended_invariant_ids
        complete = outcome.canonical_reload_validated_invariant_ids
        qualifications = tuple(filter(None, (
            outcome.canonical_reload_hn_qualification_status,
            *assignment.control_memberships,
        )))
        collision = outcome.collision_classification
        gate = outcome.canonical_reload_build_gate_passed
        canonical = outcome.canonical_reload_integrity_passed
        event_delta = outcome.structural_difference.event_count_delta
        detector_truth = "normal" if assignment.role.kind is RoleKind.HARD_NEGATIVE else "anomaly"
        if assignment.role.kind is RoleKind.HARD_NEGATIVE and not (
            canonical and outcome.canonical_reload_validation_status == "validated_normal"
            and outcome.canonical_reload_hn_qualification_status == "qualified"
            and not complete
        ):
            raise CompositionBuildError(
                f"HN postcondition failed without replacement: {assignment.role.key}"
            )
        if assignment.role.kind is RoleKind.ANOMALY and not (
            canonical and set(intended).issubset(complete)
        ):
            raise CompositionBuildError(
                f"anomaly postcondition failed without replacement: {assignment.role.key}"
            )
    times = [float(item["timestamp"]) for item in records]
    provenance = {
        "build_version": COMPOSITION_BUILD_VERSION,
        "adapter_dispatch_version": ADAPTER_DISPATCH_REGISTRY_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "assignment": assignment.to_record(),
        "detector_evidence_hash": detector_evidence_hash(records),
        "canonical_reload_valid": canonical,
        "intended_invariant_ids": intended,
        "complete_invariant_ids": complete,
    }
    return ScientificSample(
        assignment=assignment, detector_truth=detector_truth,
        event_records=tuple(records),
        detector_evidence_hash=provenance["detector_evidence_hash"],
        intended_invariant_ids=tuple(intended), complete_invariant_ids=tuple(complete),
        semantic_qualifications=qualifications,
        collision_classification=collision, aggregate_gate_passed=gate,
        canonical_reload_valid=canonical, event_count_delta=event_delta,
        event_type_delta=_event_type_delta(base, tuple(records)),
        duration=(max(times) - min(times)) if times else 0.0,
        outcome_markers=_markers(tuple(records)),
        provenance_reference=stable_reference(provenance),
    )


def build_composition(
    population: ScientificDeviceProfilePopulation,
    pool: BaseOpportunityPool,
    plan: CompositionPlan,
    *,
    transformation_seed: int,
) -> CompositionBuildResult:
    """Build exactly the frozen plan; never replace a failed assignment."""
    if plan.feasibility_status != "FEASIBLE":
        raise CompositionBuildError("infeasible plan cannot invoke adapters")
    if plan.population_fingerprint != population.population_fingerprint:
        raise ValueError("plan/population fingerprint mismatch")
    if plan.pool_fingerprint != pool.pool_fingerprint:
        raise ValueError("plan/pool fingerprint mismatch")
    state = random.getstate()
    population_before = population.population_fingerprint
    pool_before = pool.pool_fingerprint
    base_records = {item.base_trace_reference: item.to_record() for item in pool.bases}
    plan_before = plan.normalized_record()
    by_reference = {item.base_trace_reference: item for item in pool.bases}
    try:
        samples = tuple(
            _sample(assignment, by_reference[assignment.base_trace_reference], transformation_seed)
            for assignment in plan.assignments
        )
        diagnostics = enforce_detector_evidence_integrity(
            samples, plan.specification.duplicate_evidence_policy,
        )
        payload = {
            "schema_version": COMPOSITION_BUILD_RESULT_SCHEMA_VERSION,
            "build_version": COMPOSITION_BUILD_VERSION,
            "plan_fingerprint": plan.composition_fingerprint,
            "transformation_seed": transformation_seed,
            "samples": [
                {
                    "base_trace_reference": item.assignment.base_trace_reference,
                    "role": item.assignment.role.to_record(),
                    "detector_truth": item.detector_truth,
                    "detector_evidence_hash": item.detector_evidence_hash,
                    "intended_invariant_ids": item.intended_invariant_ids,
                    "complete_invariant_ids": item.complete_invariant_ids,
                    "canonical_reload_valid": item.canonical_reload_valid,
                    "provenance_reference": item.provenance_reference,
                } for item in samples
            ],
            "post_diagnostics": diagnostics,
        }
        return CompositionBuildResult(
            plan_fingerprint=plan.composition_fingerprint,
            transformation_seed=transformation_seed,
            samples=samples,
            post_diagnostics_json=canonical_json(diagnostics),
            build_fingerprint=stable_reference(payload),
        )
    finally:
        if population.population_fingerprint != population_before:
            raise AssertionError("composition build mutated population")
        if pool.pool_fingerprint != pool_before:
            raise AssertionError("composition build mutated pool")
        if {item.base_trace_reference: item.to_record() for item in pool.bases} != base_records:
            raise AssertionError("composition build mutated a base")
        if plan.normalized_record() != plan_before:
            raise AssertionError("composition build mutated the frozen plan")
        if random.getstate() != state:
            random.setstate(state)
            raise AssertionError("composition build perturbed global RNG state")
