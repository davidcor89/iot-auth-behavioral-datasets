"""Privileged integrated stress orchestration for the frozen 4D.3 adapters.

This module adds no semantic construction.  Every derivative starts from one
immutable valid base and is delegated to exactly one already-versioned adapter.
The compact outcomes preserve construction, qualification, validated truth,
collision, and aggregate-gate status as distinct facts.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from simulator.anomaly_contract import VARIANT_CAPABILITIES
from simulator.config.integrated_role_stress import IntegratedRoleStressConfiguration
from simulator.config.settings import cfg
from simulator.hard_negative import HN_VARIANT_BY_ID
from simulator.role_integration.access_boundary import (
    ANOMALY_VARIANTS as ACCESS_ANOMALIES,
    HN_VARIANTS as ACCESS_HNS,
    AccessRoleTransformationRequest,
    integrate_access_role,
)
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.direct_timing import (
    IDENTITY_COHERENT_HN,
    POSITIVE_MARGIN_HN,
    SAME_ATTEMPT_NONCE_HN,
    TIMESTAMP_ANOMALY,
    RoleTransformationRequest,
    integrate_role,
    materialize_base,
    materialize_derived,
    _validation,
)
from simulator.role_integration.identity_binding_boundary import (
    PERSISTENT_IDENTITY_ANOMALY,
    TOKEN_CONTEXT_COHERENT_HN,
    TOKEN_IDENTITY_ANOMALY,
    TOKEN_IDENTITY_COHERENT_HN,
    IdentityBindingRoleTransformationRequest,
    integrate_identity_binding_role,
)
from simulator.role_integration.models import (
    ValidBaseTrace,
    canonical_json,
    normalized_event_records,
    stable_reference,
)
from simulator.role_integration.nonce_boundary import (
    FRESH_NONCE_HN,
    NONCE_REUSE_ANOMALY,
    NonceRoleTransformationRequest,
    integrate_nonce_role,
)
from simulator.role_integration.pool import (
    BaseOpportunityPool,
    build_opportunity_pool,
)
from simulator.role_integration.token_boundary import (
    CORRECT_REPLACEMENT_T2_HN,
    LAST_VALID_T1_HN,
    REPEATED_CURRENT_TOKEN_HN,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY,
    TokenRoleTransformationRequest,
    integrate_token_role,
)
from simulator.role_integration.transformation import TransformationStatus
from simulator.runner import RunPopulation, build_run_population
from simulator.hard_negative import QualificationStatus
from simulator.hard_negative_qualification import qualify_hard_negative


INTEGRATED_VARIANT_CATALOGUE_VERSION = "4d3.7-complete-variant-catalogue-v1"
INTENDED_INVARIANT_REGISTRY_VERSION = "4d3.7-intended-invariant-registry-v1"
INTEGRATED_STRESS_ORCHESTRATOR_VERSION = "4d3.7-stress-orchestrator-v1"
INTEGRATED_STRESS_OUTCOME_SCHEMA_VERSION = "4d3.7-stress-outcome-schema-v2"
COLLISION_CLASSIFICATION_VERSION = "4d3.7-collision-classification-v1"
STRUCTURAL_DIFFERENCE_NORMALIZATION_VERSION = "4d3.7-structural-difference-v1"
INTEGRATED_STRESS_PROVENANCE_VERSION = "4d3.7-integrated-provenance-v1"
HANDOFF_SCHEMA_VERSION = "4d3.7-4d4-handoff-v1"
STRESS_POOL_ADMISSION_VERSION = "4d3.7-canonical-admission-defense-v2"

DIRECT_VARIANTS = (
    SAME_ATTEMPT_NONCE_HN, IDENTITY_COHERENT_HN,
    POSITIVE_MARGIN_HN, TIMESTAMP_ANOMALY,
)
IDENTITY_BINDING_VARIANTS = (
    TOKEN_IDENTITY_COHERENT_HN, TOKEN_CONTEXT_COHERENT_HN,
    PERSISTENT_IDENTITY_ANOMALY, TOKEN_IDENTITY_ANOMALY,
)
HN_VARIANTS = (
    SAME_ATTEMPT_NONCE_HN,
    IDENTITY_COHERENT_HN,
    POSITIVE_MARGIN_HN,
    *ACCESS_HNS,
    FRESH_NONCE_HN,
    REPEATED_CURRENT_TOKEN_HN,
    LAST_VALID_T1_HN,
    CORRECT_REPLACEMENT_T2_HN,
    TOKEN_IDENTITY_COHERENT_HN,
    TOKEN_CONTEXT_COHERENT_HN,
)
ANOMALY_VARIANTS = (
    TIMESTAMP_ANOMALY,
    *ACCESS_ANOMALIES,
    NONCE_REUSE_ANOMALY,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY,
    PERSISTENT_IDENTITY_ANOMALY,
    TOKEN_IDENTITY_ANOMALY,
)
INTEGRATED_VARIANTS = HN_VARIANTS + ANOMALY_VARIANTS


@dataclass(frozen=True)
class IntegratedVariantSpecification:
    variant_id: str
    role_type: str
    adapter_family: str
    intended_invariant_ids: tuple[str, ...]
    broad_predicate_id: str


_BROAD_PREDICATE = {
    SAME_ATTEMPT_NONCE_HN: ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE,
    IDENTITY_COHERENT_HN: ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION,
    POSITIVE_MARGIN_HN: ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE,
    TIMESTAMP_ANOMALY: ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE,
    **{variant: ApplicabilityPredicate.PROTECTED_ACCESS for variant in (*ACCESS_HNS, *ACCESS_ANOMALIES)},
    FRESH_NONCE_HN: ApplicabilityPredicate.DISTINCT_AUTHENTICATION_ATTEMPTS,
    NONCE_REUSE_ANOMALY: ApplicabilityPredicate.DISTINCT_AUTHENTICATION_ATTEMPTS,
    REPEATED_CURRENT_TOKEN_HN: ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES,
    LAST_VALID_T1_HN: ApplicabilityPredicate.CURRENT_T1_USE_BEFORE_REPLACEMENT,
    CORRECT_REPLACEMENT_T2_HN: ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY: ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT,
    TOKEN_IDENTITY_COHERENT_HN: ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING,
    TOKEN_CONTEXT_COHERENT_HN: ApplicabilityPredicate.TOKEN_PROTECTED_CONTEXT_BINDING,
    PERSISTENT_IDENTITY_ANOMALY: ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION,
    TOKEN_IDENTITY_ANOMALY: ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING,
}


def _family(variant: str) -> str:
    if variant in DIRECT_VARIANTS:
        return "direct_timing"
    if variant in (*ACCESS_HNS, *ACCESS_ANOMALIES):
        return "access"
    if variant in (FRESH_NONCE_HN, NONCE_REUSE_ANOMALY):
        return "nonce"
    if variant in (
        REPEATED_CURRENT_TOKEN_HN, LAST_VALID_T1_HN,
        CORRECT_REPLACEMENT_T2_HN, SUPERSEDED_TOKEN_REPLAY_ANOMALY,
    ):
        return "token"
    if variant in IDENTITY_BINDING_VARIANTS:
        return "identity_binding"
    raise KeyError(variant)


def _intended(variant: str) -> tuple[str, ...]:
    if variant in HN_VARIANT_BY_ID:
        return tuple(HN_VARIANT_BY_ID[variant].invariant_ids)
    return tuple(VARIANT_CAPABILITIES[variant].invariant_ids)


INTEGRATED_VARIANT_SPECIFICATIONS = tuple(
    IntegratedVariantSpecification(
        variant_id=variant,
        role_type="hard_negative" if variant in HN_VARIANTS else "anomaly",
        adapter_family=_family(variant),
        intended_invariant_ids=_intended(variant),
        broad_predicate_id=_BROAD_PREDICATE[variant].value,
    )
    for variant in INTEGRATED_VARIANTS
)
SPECIFICATION_BY_VARIANT = {
    item.variant_id: item for item in INTEGRATED_VARIANT_SPECIFICATIONS
}


@dataclass(frozen=True)
class StructuralDifference:
    event_count_delta: int
    event_type_change_count: int
    semantic_timestamp_change_count: int
    observed_timestamp_change_count: int
    identity_field_change_count: int
    token_field_change_count: int
    nonce_field_change_count: int
    enforcement_field_change_count: int
    access_operation_delta: int
    endpoint_changed: bool

    @property
    def exact_event_equality(self) -> bool:
        return not any((
            self.event_count_delta, self.event_type_change_count,
            self.semantic_timestamp_change_count,
            self.observed_timestamp_change_count,
            self.identity_field_change_count, self.token_field_change_count,
            self.nonce_field_change_count, self.enforcement_field_change_count,
            self.access_operation_delta, self.endpoint_changed,
        ))


@dataclass(frozen=True)
class IntegratedStressOutcome:
    schema_version: str
    base_trace_reference: str
    population_device_key: str
    behavioral_profile_id: str
    variant_id: str
    role_type: str
    adapter_family: str
    broad_applicable: bool
    broad_evaluation_reference: str
    precise_applicable: bool
    construction_status: str
    construction_reason: str
    construction_mode: Optional[str]
    intended_invariant_ids: tuple[str, ...]
    intended_invariants_validated: bool
    complete_validated_invariant_ids: tuple[str, ...]
    additional_invariant_ids: tuple[str, ...]
    collision_classification: str
    independent_validation_status: Optional[str]
    aggregate_build_gate_passed: bool
    hn_qualification_status: Optional[str]
    hn_independently_normal: bool
    gt_consolidated: bool
    derived_trace_reference: str
    integrated_reference: str
    selected_target_references: tuple[str, ...]
    structural_difference: StructuralDifference
    profile_preserved: bool
    base_reference_preserved: bool
    provenance_complete: bool
    lifecycle: str
    recovery: str
    renewal: str
    activity_count: int
    canonical_reload_validation_status: Optional[str]
    canonical_reload_build_gate_passed: bool
    canonical_reload_validated_invariant_ids: tuple[str, ...]
    canonical_reload_hn_qualification_status: Optional[str]
    canonical_reload_integrity_passed: bool

    def normalized_record(self) -> dict[str, Any]:
        record = asdict(self)
        # Adapter references are retained for privileged audit, but several
        # pre-4D.3.7 formats legitimately include runtime event UUIDs.  The
        # integrated reference above is the stable normalized identity used by
        # this checkpoint, so raw adapter references are excluded only from the
        # reproducibility fingerprint.
        record.pop("derived_trace_reference")
        record.pop("construction_reason")
        return record


@dataclass(frozen=True)
class IntegratedStressResult:
    orchestrator_version: str
    configuration_json: str
    population_fingerprint: str
    population_diagnostics_json: str
    opportunity_pool_fingerprint: str
    opportunity_diagnostics_json: str
    variant_catalogue_version: str
    intended_invariant_registry_version: str
    outcomes: tuple[IntegratedStressOutcome, ...]
    result_fingerprint: str

    def normalized_record(self) -> dict[str, Any]:
        return {
            "orchestrator_version": self.orchestrator_version,
            "configuration": json.loads(self.configuration_json),
            "population_fingerprint": self.population_fingerprint,
            "population_diagnostics": json.loads(self.population_diagnostics_json),
            "opportunity_pool_fingerprint": self.opportunity_pool_fingerprint,
            "opportunity_diagnostics": json.loads(self.opportunity_diagnostics_json),
            "variant_catalogue_version": self.variant_catalogue_version,
            "intended_invariant_registry_version": self.intended_invariant_registry_version,
            "outcomes": [item.normalized_record() for item in self.outcomes],
        }


def build_stress_population_and_pool(
    configuration: IntegratedRoleStressConfiguration,
) -> tuple[RunPopulation, BaseOpportunityPool]:
    """Construct one role-independent campaign without leaking RNG state."""
    previous_count = cfg.simulation.num_devices
    state = random.getstate()
    try:
        cfg.simulation.num_devices = configuration.device_count
        random.seed(configuration.device_construction_seed)
        population = build_run_population(configuration.population_configuration())
        pool = build_opportunity_pool(
            population, configuration.opportunity_configuration(),
        )
        # Defense in depth: the common pool builder is now the canonical
        # admission authority.  Stress must never be the first layer to find an
        # invalid serialized base, and it never repairs or resamples one.
        for base in pool.bases:
            events, context = materialize_base(base)
            validation = _validation(events, context)
            if not (
                validation.trace_ground_truth_statuses.get(context.trace_id)
                == "validated_normal"
                and validation.build_gate_passed
                and not validation.trace_validated_invariant_ids.get(
                    context.trace_id, ()
                )
            ):
                raise AssertionError(
                    "canonical pool admission exposed an invalid serialized base"
                )
        return population, pool
    finally:
        cfg.simulation.num_devices = previous_count
        random.setstate(state)


def _integrate(base: ValidBaseTrace, variant: str, seed: int):
    family = _family(variant)
    if family == "direct_timing":
        return integrate_role(base, RoleTransformationRequest(
            base.base_trace_reference, variant, seed,
        ))
    if family == "access":
        return integrate_access_role(base, AccessRoleTransformationRequest(
            base.base_trace_reference, variant, seed,
        ))
    if family == "nonce":
        return integrate_nonce_role(base, NonceRoleTransformationRequest(
            base.base_trace_reference, variant, seed,
        ))
    if family == "token":
        return integrate_token_role(base, TokenRoleTransformationRequest(
            base.base_trace_reference, variant, seed,
        ))
    return integrate_identity_binding_role(
        base, IdentityBindingRoleTransformationRequest(
            base.base_trace_reference, variant, seed,
        ),
    )


def _broad(base: ValidBaseTrace, variant: str) -> tuple[bool, str]:
    predicate = SPECIFICATION_BY_VARIANT[variant].broad_predicate_id
    result = next(
        item for item in base_applicability(base) if item.predicate_id == predicate
    )
    return result.applicable, result.evaluation_reference


def base_applicability(base: ValidBaseTrace):
    # Imported lazily only to keep the orchestration dependency explicit.
    from simulator.role_integration.applicability import evaluate_applicability
    return evaluate_applicability(base)


_IDENTITY_FIELDS = {"authenticated_identity", "identity_claim"}
_TOKEN_FIELDS = {
    "token_id", "current_token_id", "token_generation", "token_lineage_status",
    "supersedes_token_id", "superseded_by_token_id", "token_replacement_timestamp",
    "token_expiry", "token_scope", "token_bound_identity",
    "token_bound_protected_session_id",
}
_ENFORCEMENT_FIELDS = {
    "result", "failure_reason", "new_state", "token_validation_result",
    "authorization_decision", "resource_operation_outcome",
    "authenticated_context_active", "token_context_active",
    "protected_session_active",
}
_ACCESS_TYPES = {"access_request", "access_granted", "access_denied"}


def structural_difference(
    base_records: Sequence[Mapping[str, Any]],
    derived_records: Sequence[Mapping[str, Any]],
) -> StructuralDifference:
    paired = zip(base_records, derived_records)
    type_changes = semantic_changes = observed_changes = 0
    identity_changes = token_changes = nonce_changes = enforcement_changes = 0
    for original, derived in paired:
        type_changes += original.get("event_type") != derived.get("event_type")
        semantic_changes += original.get("timestamp") != derived.get("timestamp")
        observed_changes += original.get("observed_timestamp") != derived.get("observed_timestamp")
        identity_changes += sum(original.get(name) != derived.get(name) for name in _IDENTITY_FIELDS)
        token_changes += sum(original.get(name) != derived.get(name) for name in _TOKEN_FIELDS)
        nonce_changes += original.get("nonce") != derived.get("nonce")
        enforcement_changes += sum(original.get(name) != derived.get(name) for name in _ENFORCEMENT_FIELDS)
    base_access = sum(item.get("event_type") in _ACCESS_TYPES for item in base_records)
    derived_access = sum(item.get("event_type") in _ACCESS_TYPES for item in derived_records)
    return StructuralDifference(
        event_count_delta=len(derived_records) - len(base_records),
        event_type_change_count=type_changes,
        semantic_timestamp_change_count=semantic_changes,
        observed_timestamp_change_count=observed_changes,
        identity_field_change_count=identity_changes,
        token_field_change_count=token_changes,
        nonce_field_change_count=nonce_changes,
        enforcement_field_change_count=enforcement_changes,
        access_operation_delta=derived_access - base_access,
        endpoint_changed=(
            bool(base_records) != bool(derived_records)
            or (
                bool(base_records) and bool(derived_records)
                and base_records[-1].get("event_type")
                != derived_records[-1].get("event_type")
            )
        ),
    )


def _selected_refs(provenance: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "selected_event_references", "transformed_event_references",
        "evidence_references",
    ):
        item = provenance.get(key)
        if isinstance(item, Mapping):
            values.extend(str(value) for value in item.values())
        elif isinstance(item, (list, tuple)):
            values.extend(str(value) for value in item)
    return tuple(dict.fromkeys(values))


def _collision_classification(variant: str, additions: tuple[str, ...]) -> str:
    if variant in {
        "access_success_despite_explicit_deny",
        "access_success_without_security_prerequisites",
    }:
        return "expected_frozen_multi_invariant"
    if not additions:
        return "none"
    if variant == PERSISTENT_IDENTITY_ANOMALY and additions == (
        "INV-ACCESS-AUTH-CONTEXT",
    ):
        return "known_representation_coupling"
    return "newly_observed_requires_review"


def _canonical_reload_audit(
    result: Any, specification: IntegratedVariantSpecification,
) -> tuple[Optional[str], bool, tuple[str, ...], Optional[str], bool]:
    """Independently validate/qualify exact serialized derivative evidence."""
    if result.status is not TransformationStatus.APPLIED:
        return None, False, (), None, True
    events, context = materialize_derived(result)
    report = _validation(events, context)
    status = report.trace_ground_truth_statuses.get(context.trace_id)
    invariant_ids = tuple(sorted(
        report.trace_validated_invariant_ids.get(context.trace_id, ())
    ))
    qualification_status: Optional[str] = None
    if specification.role_type == "hard_negative":
        candidate = result.hard_negative_qualification
        if candidate is None:
            return status, report.build_gate_passed, invariant_ids, None, False
        qualified = qualify_hard_negative(events, context, candidate, report)
        qualification_status = qualified.qualification_status.value
        final = _validation(
            events, context, qualification_records=[qualified],
        )
        status = final.trace_ground_truth_statuses.get(context.trace_id)
        invariant_ids = tuple(sorted(
            final.trace_validated_invariant_ids.get(context.trace_id, ())
        ))
        passed = (
            qualified.qualification_status is QualificationStatus.QUALIFIED
            and final.build_gate_passed
            and status == "validated_normal"
            and not invariant_ids
        )
        return status, final.build_gate_passed, invariant_ids, qualification_status, passed
    if specification.variant_id == PERSISTENT_IDENTITY_ANOMALY:
        atomic_collisions = {
            record.invariant_identifier
            for record in report.records
            if record.trace_id == context.trace_id
            and record.invariant_identifier
            and record.result.value == "PASS"
            and record.validation_id.startswith("c2.")
            and record.invariant_identifier
            not in specification.intended_invariant_ids
        }
        invariant_ids = tuple(sorted(set(invariant_ids) | atomic_collisions))
    passed = (
        status == result.validation_status
        and not report.gt_consistency_failures
        and set(specification.intended_invariant_ids).issubset(invariant_ids)
        and invariant_ids == tuple(sorted(result.validated_invariant_ids))
    )
    return status, report.build_gate_passed, invariant_ids, None, passed


def _outcome(
    base: ValidBaseTrace, variant: str, result: Any, broad: bool, broad_ref: str,
    *, expected_profile_id: str, expected_population_device_key: str,
) -> IntegratedStressOutcome:
    specification = SPECIFICATION_BY_VARIANT[variant]
    complete = tuple(sorted(result.validated_invariant_ids))
    intended = tuple(sorted(specification.intended_invariant_ids))
    intended_validated = (
        set(intended).issubset(complete)
        if specification.role_type == "anomaly" and result.status is TransformationStatus.APPLIED
        else False
    )
    additions = tuple(sorted(set(complete) - set(intended)))
    provenance = result.transformation_provenance
    qualification = (
        result.hard_negative_qualification.qualification_status.value
        if result.hard_negative_qualification else None
    )
    records = result.event_records
    difference = structural_difference(base.event_records, records)
    construction_mode = provenance.get("construction_mode")
    (
        canonical_status,
        canonical_gate,
        canonical_invariants,
        canonical_qualification,
        canonical_integrity,
    ) = _canonical_reload_audit(result, specification)
    integrated_payload = {
        "provenance_version": INTEGRATED_STRESS_PROVENANCE_VERSION,
        "structural_difference_version": STRUCTURAL_DIFFERENCE_NORMALIZATION_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": variant,
        "construction_status": result.status.value,
        "construction_mode": construction_mode,
        "normalized_derived_evidence": normalized_event_records(records),
        "intended_invariant_ids": intended,
        "complete_validated_invariant_ids": complete,
        "qualification_status": qualification,
        "aggregate_build_gate_passed": result.validation_build_gate_passed,
        "selected_target_references": _selected_refs(provenance),
        "structural_difference": asdict(difference),
        "canonical_reload": {
            "validation_status": canonical_status,
            "build_gate_passed": canonical_gate,
            "validated_invariant_ids": canonical_invariants,
            "hn_qualification_status": canonical_qualification,
            "integrity_passed": canonical_integrity,
        },
    }
    integrated_reference = stable_reference(integrated_payload)
    provenance_complete = bool(
        base.population_reference
        and base.behavioral_profile_id
        and base.base_trace_reference
        and broad_ref
        and provenance.get("base_trace_reference") == base.base_trace_reference
        and result.derived_trace_reference
    )
    return IntegratedStressOutcome(
        schema_version=INTEGRATED_STRESS_OUTCOME_SCHEMA_VERSION,
        base_trace_reference=base.base_trace_reference,
        population_device_key=base.population_device_key,
        behavioral_profile_id=base.behavioral_profile_id,
        variant_id=variant,
        role_type=specification.role_type,
        adapter_family=specification.adapter_family,
        broad_applicable=broad,
        broad_evaluation_reference=broad_ref,
        precise_applicable=result.status in {
            TransformationStatus.APPLIED,
            TransformationStatus.CONSTRUCTION_ERROR,
        },
        construction_status=result.status.value,
        construction_reason=result.reason,
        construction_mode=construction_mode,
        intended_invariant_ids=intended,
        intended_invariants_validated=intended_validated,
        complete_validated_invariant_ids=complete,
        additional_invariant_ids=additions,
        collision_classification=_collision_classification(variant, additions),
        independent_validation_status=result.validation_status,
        aggregate_build_gate_passed=result.validation_build_gate_passed,
        hn_qualification_status=qualification,
        hn_independently_normal=(
            specification.role_type == "hard_negative"
            and result.validation_status == "validated_normal"
            and not complete
        ),
        gt_consolidated=(
            specification.role_type == "anomaly"
            and result.status is TransformationStatus.APPLIED
            and intended_validated
        ),
        derived_trace_reference=result.derived_trace_reference,
        integrated_reference=integrated_reference,
        selected_target_references=_selected_refs(provenance),
        structural_difference=difference,
        profile_preserved=(
            base.behavioral_profile_id == expected_profile_id
            and base.structural_summary.behavioral_profile_id == expected_profile_id
            and base.population_device_key == expected_population_device_key
            and base.structural_summary.population_device_key
            == expected_population_device_key
        ),
        base_reference_preserved=result.base_trace_reference == base.base_trace_reference,
        provenance_complete=provenance_complete,
        lifecycle=base.structural_summary.lifecycle,
        recovery=base.structural_summary.recovery,
        renewal=base.structural_summary.renewal,
        activity_count=base.structural_summary.activity_count,
        canonical_reload_validation_status=canonical_status,
        canonical_reload_build_gate_passed=canonical_gate,
        canonical_reload_validated_invariant_ids=canonical_invariants,
        canonical_reload_hn_qualification_status=canonical_qualification,
        canonical_reload_integrity_passed=canonical_integrity,
    )


def exercise_integrated_roles(
    population: RunPopulation,
    pool: BaseOpportunityPool,
    configuration: IntegratedRoleStressConfiguration,
    *,
    variants: Sequence[str] = INTEGRATED_VARIANTS,
    bases: Optional[Sequence[ValidBaseTrace]] = None,
) -> IntegratedStressResult:
    """Exercise every requested base/variant independently and canonically sort."""
    if set(variants) - set(INTEGRATED_VARIANTS):
        raise ValueError("stress request contains a non-frozen variant")
    scientific = population.behavioral_profiles
    if scientific.population_fingerprint != pool.population_reference:
        raise ValueError("population/pool provenance mismatch")
    source_bases = tuple(pool.bases if bases is None else bases)
    assignments_by_device = {
        item.device_id: item for item in scientific.assignments
    }
    state = random.getstate()
    original_pool = pool.pool_fingerprint
    outcomes: list[IntegratedStressOutcome] = []
    try:
        for base in source_bases:
            original = base.to_record()
            assignment = assignments_by_device[base.device_id]
            for variant in variants:
                broad, broad_ref = _broad(base, variant)
                result = _integrate(base, variant, configuration.transformation_seed)
                outcomes.append(_outcome(
                    base, variant, result, broad, broad_ref,
                    expected_profile_id=assignment.profile.behavioral_profile_id,
                    expected_population_device_key=assignment.population_device_key,
                ))
                if base.to_record() != original:
                    raise AssertionError("role adapter mutated its immutable base")
            if pool.pool_fingerprint != original_pool:
                raise AssertionError("role adapter mutated the opportunity pool")
    finally:
        if random.getstate() != state:
            random.setstate(state)
            raise AssertionError("integrated role stress perturbed global RNG state")
    ordered = tuple(sorted(
        outcomes, key=lambda item: (item.base_trace_reference, item.variant_id),
    ))
    configuration_json = canonical_json(asdict(configuration))
    provisional = {
        "orchestrator_version": INTEGRATED_STRESS_ORCHESTRATOR_VERSION,
        "configuration": json.loads(configuration_json),
        "population_fingerprint": scientific.population_fingerprint,
        "population_diagnostics": dict(scientific.diagnostics),
        "opportunity_pool_fingerprint": pool.pool_fingerprint,
        "opportunity_diagnostics": pool.diagnostics,
        "variant_catalogue_version": INTEGRATED_VARIANT_CATALOGUE_VERSION,
        "intended_invariant_registry_version": INTENDED_INVARIANT_REGISTRY_VERSION,
        "outcomes": [item.normalized_record() for item in ordered],
    }
    return IntegratedStressResult(
        orchestrator_version=INTEGRATED_STRESS_ORCHESTRATOR_VERSION,
        configuration_json=configuration_json,
        population_fingerprint=scientific.population_fingerprint,
        population_diagnostics_json=canonical_json(dict(scientific.diagnostics)),
        opportunity_pool_fingerprint=pool.pool_fingerprint,
        opportunity_diagnostics_json=canonical_json(pool.diagnostics),
        variant_catalogue_version=INTEGRATED_VARIANT_CATALOGUE_VERSION,
        intended_invariant_registry_version=INTENDED_INVARIANT_REGISTRY_VERSION,
        outcomes=ordered,
        result_fingerprint=stable_reference(provisional),
    )


def summarize_outcomes(result: IntegratedStressResult) -> dict[str, dict[str, Any]]:
    """Return the minimum 4D.4 handoff taxonomy by variant."""
    summary: dict[str, dict[str, Any]] = {}
    for specification in INTEGRATED_VARIANT_SPECIFICATIONS:
        rows = [item for item in result.outcomes if item.variant_id == specification.variant_id]
        statuses = Counter(item.construction_status for item in rows)
        applied = [item for item in rows if item.construction_status == "APPLIED"]
        precise = [item for item in rows if item.precise_applicable]
        summary[specification.variant_id] = {
            "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
            "role_type": specification.role_type,
            "adapter_family": specification.adapter_family,
            "intended_invariant_ids": list(specification.intended_invariant_ids),
            "broad_applicable": sum(item.broad_applicable for item in rows),
            "precise_applicable": len(precise),
            "broad_to_precise_loss": sum(item.broad_applicable for item in rows) - len(precise),
            "applied": len(applied),
            "applicable_devices": len({item.population_device_key for item in applied}),
            "statuses": dict(sorted(statuses.items())),
            "qualified": sum(item.hn_qualification_status == "qualified" for item in rows),
            "hn_independently_normal": sum(item.hn_independently_normal for item in rows),
            "intended_validated": sum(item.intended_invariants_validated for item in rows),
            "gt_consolidated": sum(item.gt_consolidated for item in rows),
            "gate_passed": sum(item.aggregate_build_gate_passed for item in applied),
            "gate_rejected": sum(not item.aggregate_build_gate_passed for item in applied),
            "complete_invariant_patterns": sorted({item.complete_validated_invariant_ids for item in applied}),
            "collision_classifications": dict(Counter(
                item.collision_classification for item in applied
            )),
            "construction_modes": dict(Counter(item.construction_mode for item in applied)),
            "event_count_deltas": dict(Counter(
                item.structural_difference.event_count_delta for item in applied
            )),
            "structural_signatures": dict(Counter(canonical_json(
                asdict(item.structural_difference)
            ) for item in applied)),
            "lifecycle": dict(Counter(item.lifecycle for item in applied)),
            "recovery": dict(Counter(item.recovery for item in applied)),
            "renewal": dict(Counter(item.renewal for item in applied)),
            "activity": dict(Counter(item.activity_count for item in applied)),
        }
    return summary
