"""4D.3.2 direct-evidence and local timing-boundary role integration.

The adapters operate only on immutable 4D.3.1 valid bases.  They never search
for or regenerate a replacement base, consume ordinary planning randomness, or
alter the base object.  Qualification and anomaly truth remain owned by the
existing independent validator and frozen HN qualifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import InjectionRecord, VARIANT_CAPABILITIES
from simulator.config.role_transformation import DirectTimingTransformationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.data.synchronized_views import canonicalize_sequence_pairs
from simulator.engines.event_engine import SessionContext
from simulator.event_model import AuthEvent, AuthState, EventResult, EventType
from simulator.hard_negative import (
    CandidateStatus,
    ConstructionStatus,
    HN_VARIANT_BY_ID,
    HardNegativeQualificationRecord,
    QualificationStatus,
)
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.provenance import build_generation_provenance
from simulator.role_integration.applicability import (
    ApplicabilityPredicate,
    ApplicabilityResult,
    evaluate_applicability,
)
from simulator.role_integration.models import ValidBaseTrace, canonical_json, stable_reference
from simulator.role_integration.pool import derive_future_transformation_reference
from simulator.role_integration.transformation import (
    ROLE_TRANSFORMATION_INTERFACE_VERSION,
    TransformationFamily,
    TransformationOutcome,
    TransformationStatus,
)
from simulator.validation import ValidationReport, validate_synchronized_dataset


DIRECT_REFERENCE_ADAPTER_VERSION = "4d3.2-direct-reference-adapter-v1"
TIMING_BOUNDARY_ADAPTER_VERSION = "4d3.2-timing-boundary-adapter-v1"
TARGET_SELECTION_POLICY_VERSION = "4d3.2-canonical-first-nonce-response-v1"
DERIVED_TRACE_REFERENCE_VERSION = "4d3.2-derived-trace-sha256-v1"
TIMING_TRANSFORMATION_PROVENANCE_VERSION = "4d3.2-timing-provenance-v1"
COUNTERFACTUAL_QA_REFERENCE_VERSION = "4d3.2-counterfactual-timing-pair-v1"
TARGETED_CAUSAL_RELATION = "declared_event_time_after_predecessor"

SAME_ATTEMPT_NONCE_HN = "legitimate_same_attempt_nonce_recurrence"
IDENTITY_COHERENT_HN = "persistent_authenticated_identity_coherent"
POSITIVE_MARGIN_HN = "small_positive_causal_margin"
TIMESTAMP_ANOMALY = "timestamp_inconsistency"
SUPPORTED_ROLE_VARIANTS = (
    SAME_ATTEMPT_NONCE_HN,
    IDENTITY_COHERENT_HN,
    POSITIVE_MARGIN_HN,
    TIMESTAMP_ANOMALY,
)


@dataclass(frozen=True)
class RoleTransformationRequest:
    base_trace_reference: str
    variant_id: str
    transformation_seed: int
    configuration: DirectTimingTransformationConfiguration = (
        DirectTimingTransformationConfiguration()
    )


@dataclass(frozen=True)
class TimingRelation:
    predecessor_ordinal: int
    target_ordinal: int
    predecessor_event_id: str
    target_event_id: str
    auth_attempt_id: str
    nonce: str
    predecessor_semantic_timestamp: float
    target_semantic_timestamp: float
    predecessor_observed_timestamp: float
    target_observed_timestamp: float

    @property
    def base_observed_margin(self) -> float:
        return self.target_observed_timestamp - self.predecessor_observed_timestamp


@dataclass(frozen=True)
class DerivedRoleTrace:
    reference_version: str
    derived_trace_reference: str
    base_trace_reference: str
    variant_id: str
    status: TransformationStatus
    event_records_json: tuple[str, ...]
    session_context_json: str
    transformation_provenance_json: str
    validated_invariant_ids: tuple[str, ...]
    validation_status: Optional[str]
    validation_build_gate_passed: bool
    hard_negative_qualification: Optional[HardNegativeQualificationRecord]
    reason: str

    @property
    def event_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(item) for item in self.event_records_json)

    @property
    def transformation_provenance(self) -> dict[str, Any]:
        return json.loads(self.transformation_provenance_json)


def _event_from_record(record: Mapping[str, Any]) -> AuthEvent:
    allowed = {item.name for item in fields(AuthEvent)}
    values = {key: value for key, value in record.items() if key in allowed}
    values["event_type"] = EventType(values["event_type"])
    values["previous_state"] = AuthState(values["previous_state"])
    values["new_state"] = AuthState(values["new_state"])
    values["result"] = EventResult(values["result"])
    values["invariant_families"] = tuple(values.get("invariant_families", ()))
    return AuthEvent(**values)


def _context_from_json(value: str) -> SessionContext:
    data = json.loads(value)
    for name in (
        "auth_attempt_ids", "renewal_auth_attempt_ids", "access_request_ids",
        "authorization_decisions", "resource_operation_outcomes", "injection_records",
    ):
        data[name] = tuple(data.get(name, ()))
    return SessionContext(**data)


def materialize_base(base: ValidBaseTrace) -> tuple[list[AuthEvent], SessionContext]:
    """Materialize detached runtime values; the immutable base stays untouched."""
    return ([_event_from_record(record) for record in base.event_records],
            _context_from_json(base.session_context_json))


def materialize_derived(
    derived: DerivedRoleTrace,
) -> tuple[list[AuthEvent], SessionContext]:
    """Reload the exact canonical evidence carried by a derived trace."""
    return (
        [_event_from_record(json.loads(item)) for item in derived.event_records_json],
        _context_from_json(derived.session_context_json),
    )


def _observed(event: AuthEvent) -> float:
    return event.observed_timestamp if event.observed_timestamp is not None else event.timestamp


def resolve_timing_relation(base: ValidBaseTrace) -> Optional[TimingRelation]:
    """Select the first canonical nonce_received -> response_sent relation.

    Selection is event-order based and independent of profile values, role,
    validator results, and ground truth.  The same resolver serves both sides
    of the timing counterfactual.
    """
    events = base.event_records
    for predecessor_index, predecessor in enumerate(events):
        if predecessor.get("event_type") != EventType.NONCE_RECEIVED.value:
            continue
        attempt, nonce = predecessor.get("auth_attempt_id"), predecessor.get("nonce")
        if not attempt or not nonce:
            continue
        for target_index in range(predecessor_index + 1, len(events)):
            target = events[target_index]
            if (
                target.get("event_type") == EventType.RESPONSE_SENT.value
                and target.get("auth_attempt_id") == attempt
                and target.get("nonce") == nonce
                and float(predecessor["timestamp"]) < float(target["timestamp"])
            ):
                return TimingRelation(
                    predecessor_ordinal=predecessor_index + 1,
                    target_ordinal=target_index + 1,
                    predecessor_event_id=str(predecessor["event_id"]),
                    target_event_id=str(target["event_id"]),
                    auth_attempt_id=str(attempt), nonce=str(nonce),
                    predecessor_semantic_timestamp=float(predecessor["timestamp"]),
                    target_semantic_timestamp=float(target["timestamp"]),
                    predecessor_observed_timestamp=float(
                        predecessor.get("observed_timestamp", predecessor["timestamp"])
                    ),
                    target_observed_timestamp=float(
                        target.get("observed_timestamp", target["timestamp"])
                    ),
                )
    return None


def _predicate(base: ValidBaseTrace, predicate: ApplicabilityPredicate) -> ApplicabilityResult:
    return next(
        item for item in evaluate_applicability(base)
        if item.predicate_id == predicate.value
    )


def _resolve_event_references(
    base: ValidBaseTrace, references: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    result = []
    for reference in references:
        prefix, _, ordinal = reference.partition(":")
        if prefix != "event" or not ordinal.isdigit():
            raise ValueError(f"invalid event reference: {reference}")
        result.append(base.event_records[int(ordinal) - 1])
    return tuple(result)


def _validation(
    events: list[AuthEvent], context: SessionContext,
    *, qualification_records: Optional[list[HardNegativeQualificationRecord]] = None,
) -> ValidationReport:
    # Adapter acceptance and GT are established on the exact six-decimal
    # event representation published downstream, never only on transient
    # higher-precision objects.
    events, context = canonicalize_sequence_pairs([(events, context)])[0]
    provenance = build_generation_provenance(
        cfg, run_id=f"4d3.2-{context.trace_id}",
    )
    manifest = field_manifest()
    capabilities = dataset_capability_manifest(
        provenance, generation_profiles=(context.generation_profile,),
    )
    return validate_synchronized_dataset(
        [(events, context)], provenance=provenance, field_manifest=manifest,
        dataset_manifest=capabilities,
        hard_negative_qualification_records=qualification_records,
    )


def _derived_reference(
    base: ValidBaseTrace, variant_id: str, randomness_reference: str,
    event_records_json: tuple[str, ...], context_json: str, provenance_json: str,
) -> str:
    return stable_reference({
        "reference_version": DERIVED_TRACE_REFERENCE_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": variant_id,
        "randomness_reference": randomness_reference,
        "event_records": event_records_json,
        "session_context": context_json,
        "transformation_provenance": json.loads(provenance_json),
    })


def counterfactual_group_reference(
    base_trace_reference: str,
    configuration: DirectTimingTransformationConfiguration,
) -> str:
    return stable_reference({
        "version": COUNTERFACTUAL_QA_REFERENCE_VERSION,
        "base_trace_reference": base_trace_reference,
        "timing_configuration_reference": configuration.configuration_reference,
        "variants": (POSITIVE_MARGIN_HN, TIMESTAMP_ANOMALY),
    })


def _empty_derived(
    base: ValidBaseTrace, request: RoleTransformationRequest,
    status: TransformationStatus, reason: str, randomness_reference: str,
    family: TransformationFamily,
) -> DerivedRoleTrace:
    provenance_json = canonical_json({
        "adapter_version": (
            TIMING_BOUNDARY_ADAPTER_VERSION
            if family is TransformationFamily.TIMING else DIRECT_REFERENCE_ADAPTER_VERSION
        ),
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "status": status.value,
        "transformation_randomness_reference": randomness_reference,
        "applied": False,
        "reason": reason,
    })
    context_json = base.session_context_json
    derived_reference = _derived_reference(
        base, request.variant_id, randomness_reference,
        base.event_records_json, context_json, provenance_json,
    )
    return DerivedRoleTrace(
        reference_version=DERIVED_TRACE_REFERENCE_VERSION,
        derived_trace_reference=derived_reference,
        base_trace_reference=base.base_trace_reference,
        variant_id=request.variant_id, status=status,
        event_records_json=base.event_records_json,
        session_context_json=context_json,
        transformation_provenance_json=provenance_json,
        validated_invariant_ids=(), validation_status=None,
        validation_build_gate_passed=False, hard_negative_qualification=None,
        reason=reason,
    )


def _hn_candidate(
    *, context: SessionContext, variant_id: str,
    derived_reference: str, target_ids: tuple[str, ...], evidence_ids: tuple[str, ...],
    boundary_parameters: Mapping[str, Any], construction_parameters: Mapping[str, Any],
    counterfactual_reference: Optional[str] = None,
) -> HardNegativeQualificationRecord:
    variant = HN_VARIANT_BY_ID[variant_id]
    return HardNegativeQualificationRecord(
        hard_negative_candidate_id=derived_reference,
        trace_id=context.trace_id, scenario_id=context.scenario_id,
        device_id=context.device_id, candidate_status=CandidateStatus.REQUESTED,
        primary_intended_family_id=variant.family_id,
        intended_variant_id=variant.variant_id,
        intended_boundary_invariant_ids=variant.invariant_ids,
        construction_status=ConstructionStatus.APPLIED,
        construction_parameters=dict(construction_parameters),
        boundary_parameters=dict(boundary_parameters),
        target_event_ids=target_ids, evidence_event_ids=evidence_ids,
        qualification_reason="constructed from immutable 4D.3.1 base evidence",
        counterfactual_pair_id=counterfactual_reference,
    )


class DirectReferenceTransformation:
    """Reference existing base evidence without rewriting any event/context."""

    family = TransformationFamily.NONCE

    def __init__(self, variant_id: str):
        if variant_id not in {SAME_ATTEMPT_NONCE_HN, IDENTITY_COHERENT_HN}:
            raise ValueError("unsupported direct/reference variant")
        self.transformation_id = variant_id
        self.family = (
            TransformationFamily.NONCE
            if variant_id == SAME_ATTEMPT_NONCE_HN else TransformationFamily.IDENTITY
        )

    def applicability(self, base: ValidBaseTrace) -> tuple[ApplicabilityResult, ...]:
        predicate = (
            ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE
            if self.transformation_id == SAME_ATTEMPT_NONCE_HN
            else ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION
        )
        return (_predicate(base, predicate),)

    def transform(
        self, base: ValidBaseTrace, *, randomness_reference: str,
    ) -> TransformationOutcome:
        applicable = self.applicability(base)[0]
        if not applicable.applicable:
            return TransformationOutcome(
                ROLE_TRANSFORMATION_INTERFACE_VERSION, self.family,
                self.transformation_id, base.base_trace_reference,
                TransformationStatus.NOT_APPLICABLE, base.event_records_json,
                canonical_json({
                    "adapter_version": DIRECT_REFERENCE_ADAPTER_VERSION,
                    "variant_id": self.transformation_id,
                    "applicability": applicable.to_record(), "applied": False,
                }), applicable.reason,
            )
        return TransformationOutcome(
            ROLE_TRANSFORMATION_INTERFACE_VERSION, self.family,
            self.transformation_id, base.base_trace_reference,
            TransformationStatus.APPLIED, base.event_records_json,
            canonical_json({
                "adapter_version": DIRECT_REFERENCE_ADAPTER_VERSION,
                "variant_id": self.transformation_id,
                "base_trace_reference": base.base_trace_reference,
                "evidence_references": applicable.evidence_references,
                "transformation_randomness_reference": randomness_reference,
                "event_evidence_rewritten": False,
                "semantic_context_rewritten": False,
                "applied": True,
            }), "existing represented evidence selected without mutation",
        )


class TimingBoundaryTransformation:
    family = TransformationFamily.TIMING

    def __init__(
        self, variant_id: str,
        configuration: DirectTimingTransformationConfiguration,
    ):
        if variant_id not in {POSITIVE_MARGIN_HN, TIMESTAMP_ANOMALY}:
            raise ValueError("unsupported timing-boundary variant")
        self.transformation_id = variant_id
        self.configuration = configuration

    def applicability(self, base: ValidBaseTrace) -> tuple[ApplicabilityResult, ...]:
        # The represented nonce/response chain is the common causal relation.
        return (_predicate(base, ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE),)

    def transform(
        self, base: ValidBaseTrace, *, randomness_reference: str,
    ) -> TransformationOutcome:
        applicable = self.applicability(base)[0]
        if not applicable.applicable:
            return TransformationOutcome(
                ROLE_TRANSFORMATION_INTERFACE_VERSION, self.family,
                self.transformation_id, base.base_trace_reference,
                TransformationStatus.NOT_APPLICABLE, base.event_records_json,
                canonical_json({
                    "adapter_version": TIMING_BOUNDARY_ADAPTER_VERSION,
                    "variant_id": self.transformation_id,
                    "applicability": applicable.to_record(), "applied": False,
                }), applicable.reason,
            )
        relation = resolve_timing_relation(base)
        if relation is None:
            return TransformationOutcome(
                ROLE_TRANSFORMATION_INTERFACE_VERSION, self.family,
                self.transformation_id, base.base_trace_reference,
                TransformationStatus.UNSUPPORTED, base.event_records_json,
                canonical_json({
                    "adapter_version": TIMING_BOUNDARY_ADAPTER_VERSION,
                    "variant_id": self.transformation_id,
                    "target_selection_policy_version": TARGET_SELECTION_POLICY_VERSION,
                    "applied": False,
                }), "applicability evidence lacks a supported nonce-response relation",
            )
        records = [dict(record) for record in base.event_records]
        target = records[relation.target_ordinal - 1]
        if self.transformation_id == POSITIVE_MARGIN_HN:
            digest_unit = int(randomness_reference.removeprefix("sha256:")[:16], 16) / float(16**16 - 1)
            fraction = (
                self.configuration.hn_margin_lower_fraction
                + digest_unit * (
                    self.configuration.hn_margin_upper_fraction
                    - self.configuration.hn_margin_lower_fraction
                )
            )
            transformed_margin = self.configuration.epsilon_hn_s * fraction
            transformed = relation.predecessor_observed_timestamp + transformed_margin
            target["observed_timestamp_source"] = "4d3.2_hn_boundary_transformation"
        else:
            transformed_margin = -self.configuration.anomaly_negative_margin_s
            transformed = relation.predecessor_observed_timestamp + transformed_margin
            target["observed_timestamp_source"] = "injected_device_timestamp"
        target["observed_timestamp"] = round(transformed, 6)
        target["targeted_temporal_relationship"] = TARGETED_CAUSAL_RELATION
        provenance = {
            "provenance_version": TIMING_TRANSFORMATION_PROVENANCE_VERSION,
            "adapter_version": TIMING_BOUNDARY_ADAPTER_VERSION,
            "target_selection_policy_version": TARGET_SELECTION_POLICY_VERSION,
            "base_trace_reference": base.base_trace_reference,
            "variant_id": self.transformation_id,
            "targeted_relation": "nonce_received_precedes_response_sent",
            "predecessor_event_reference": f"event:{relation.predecessor_ordinal}",
            "target_event_reference": f"event:{relation.target_ordinal}",
            "predecessor_event_id": relation.predecessor_event_id,
            "target_event_id": relation.target_event_id,
            "authentication_attempt_id": relation.auth_attempt_id,
            "original_predecessor_observed_timestamp": relation.predecessor_observed_timestamp,
            "original_target_observed_timestamp": relation.target_observed_timestamp,
            "transformed_target_observed_timestamp": target["observed_timestamp"],
            "predecessor_semantic_timestamp": relation.predecessor_semantic_timestamp,
            "target_semantic_timestamp": relation.target_semantic_timestamp,
            "base_observed_margin_s": relation.base_observed_margin,
            "transformed_observed_margin_s": round(transformed_margin, 6),
            "epsilon_hn_s": (
                self.configuration.epsilon_hn_s
                if self.transformation_id == POSITIVE_MARGIN_HN else None
            ),
            "configuration_reference": self.configuration.configuration_reference,
            "transformation_randomness_reference": randomness_reference,
            "ordinary_base_timing_provenance_preserved": True,
            "base_timing_provenance_reference": stable_reference(base.timing_provenance),
            "applied": True,
        }
        return TransformationOutcome(
            ROLE_TRANSFORMATION_INTERFACE_VERSION, self.family,
            self.transformation_id, base.base_trace_reference,
            TransformationStatus.APPLIED,
            tuple(canonical_json(record) for record in records),
            canonical_json(provenance),
            "one target observed timestamp rewritten; semantic time unchanged",
        )


def integrate_role(
    base: ValidBaseTrace, request: RoleTransformationRequest,
) -> DerivedRoleTrace:
    """Construct, independently validate, and (for HNs) qualify one role."""
    if request.base_trace_reference != base.base_trace_reference:
        raise ValueError("request/base reference mismatch")
    randomness_reference = derive_future_transformation_reference(
        transformation_seed=request.transformation_seed,
        base_trace_reference=base.base_trace_reference,
        transformation_key=(
            f"{request.variant_id}|{request.configuration.configuration_reference}"
        ),
    )
    if request.variant_id in {SAME_ATTEMPT_NONCE_HN, IDENTITY_COHERENT_HN}:
        adapter: Any = DirectReferenceTransformation(request.variant_id)
    elif request.variant_id in {POSITIVE_MARGIN_HN, TIMESTAMP_ANOMALY}:
        adapter = TimingBoundaryTransformation(request.variant_id, request.configuration)
    else:
        return _empty_derived(
            base, request, TransformationStatus.UNSUPPORTED,
            "variant is outside the frozen 4D.3.2 integration scope",
            randomness_reference, TransformationFamily.TIMING,
        )
    outcome = adapter.transform(base, randomness_reference=randomness_reference)
    if outcome.status is not TransformationStatus.APPLIED:
        return _empty_derived(
            base, request, outcome.status, outcome.reason,
            randomness_reference, adapter.family,
        )

    events = [_event_from_record(json.loads(item)) for item in outcome.derived_event_records_json]
    context = _context_from_json(base.session_context_json)
    provenance = json.loads(outcome.transformation_provenance_json)
    context_json = base.session_context_json

    # Anomaly construction first supplies candidate evidence only.  The first
    # independent pass determines the positive event/invariant; only then are
    # final GT fields consolidated and independently validated again.
    if request.variant_id == TIMESTAMP_ANOMALY:
        relation = resolve_timing_relation(base)
        assert relation is not None
        target = events[relation.target_ordinal - 1]
        injection_id = stable_reference({
            "adapter_version": TIMING_BOUNDARY_ADAPTER_VERSION,
            "base_trace_reference": base.base_trace_reference,
            "variant_id": TIMESTAMP_ANOMALY,
            "target_event_id": target.event_id,
        })
        capability = VARIANT_CAPABILITIES[TIMESTAMP_ANOMALY]
        target.injection_id = injection_id
        target.anomaly_variant_name = TIMESTAMP_ANOMALY
        target.invariant_families = capability.invariant_families
        target.source_context = "controlled_injection"
        injection = InjectionRecord(
            injection_id=injection_id,
            historical_scenario_name=TIMESTAMP_ANOMALY,
            anomaly_variant_name=TIMESTAMP_ANOMALY,
            capability_status=capability.status.value,
            mapped_semantic_concept=capability.semantic_concept,
            invariant_families=capability.invariant_families,
            canonical_invariant_ids=capability.invariant_ids,
            targeted_event_type=target.event_type.value,
            targeted_action_or_context=TARGETED_CAUSAL_RELATION,
            source_evidence_event_ids=(relation.predecessor_event_id,),
            source_material_identifier=relation.predecessor_event_id,
            source_trace_id=context.trace_id,
            source_session_id=context.legacy_session_id,
            source_auth_attempt_id=relation.auth_attempt_id,
            original_value=relation.target_observed_timestamp,
            injected_value=_observed(target),
            injection_parameters={
                "adapter_version": TIMING_BOUNDARY_ADAPTER_VERSION,
                "target_selection_policy_version": TARGET_SELECTION_POLICY_VERSION,
                "negative_margin_s": request.configuration.anomaly_negative_margin_s,
                "transformation_randomness_reference": randomness_reference,
            },
            trace_id=context.trace_id, device_id=context.device_id,
            session_id=context.legacy_session_id,
            auth_attempt_id=relation.auth_attempt_id,
            declared_history_scope=capability.history_scope,
            intended_observable_violation=capability.semantic_concept,
            transformation_applied=True,
        )
        context = replace(
            context, attack_type=TIMESTAMP_ANOMALY,
            attack_phase=target.event_type.value, severity="medium",
            attack_scenario_intent=True, observable_anomaly=False,
            observable_violation_candidate=False,
            timestamp_delta_s=round(request.configuration.anomaly_negative_margin_s, 6),
            injection_records=(injection.to_dict(),),
        )
        evidence_report = _validation(events, context)
        expected_ids = set(capability.invariant_ids)
        confirmed_ids = set(
            evidence_report.trace_validated_invariant_ids.get(context.trace_id, ())
        )
        if not expected_ids.issubset(confirmed_ids):
            return _empty_derived(
                base, request, TransformationStatus.CONSTRUCTION_ERROR,
                "independent validation did not confirm the intended timestamp invariant",
                randomness_reference, adapter.family,
            )
        confirmed_event_ids = evidence_report.validated_positive_event_ids
        for event in events:
            if event.event_id in confirmed_event_ids:
                event.anomaly_label = TIMESTAMP_ANOMALY
        final_injection = dict(context.injection_records[0])
        final_injection.update({
            "evidence_check_passed": True,
            "observable_violation_candidate": True,
            "validation_status": "validated_positive",
        })
        context = replace(
            context, observable_anomaly=True, observable_violation_candidate=True,
            injection_records=(final_injection,),
        )
        report = _validation(events, context)
        event_records_json = tuple(canonical_json(event.to_dict()) for event in events)
        context_json = canonical_json(context.to_dict())
        provenance.update({
            "injection_id": injection_id,
            "validated_invariant_ids": sorted(
                report.trace_validated_invariant_ids.get(context.trace_id, ())
            ),
            "ground_truth_consolidation": "independent_evidence_pass_then_final_validation",
            "counterfactual_group_reference": counterfactual_group_reference(
                base.base_trace_reference, request.configuration,
            ),
        })
        provenance_json = canonical_json(provenance)
        derived_reference = _derived_reference(
            base, request.variant_id, randomness_reference,
            event_records_json, context_json, provenance_json,
        )
        return DerivedRoleTrace(
            DERIVED_TRACE_REFERENCE_VERSION, derived_reference,
            base.base_trace_reference, request.variant_id, (
                outcome.status
                if report.build_gate_passed
                and report.trace_ground_truth_statuses.get(context.trace_id)
                == "validated_positive"
                and set(report.trace_validated_invariant_ids.get(context.trace_id, ()))
                == expected_ids
                else TransformationStatus.CONSTRUCTION_ERROR
            ),
            event_records_json, context_json, provenance_json,
            tuple(report.trace_validated_invariant_ids.get(context.trace_id, ())),
            report.trace_ground_truth_statuses.get(context.trace_id),
            report.build_gate_passed, None, outcome.reason,
        )

    # HNs: the first validation establishes positive normality without HN
    # intent.  The frozen qualifier then owns the final HN decision.
    report = _validation(events, context)
    if request.variant_id == SAME_ATTEMPT_NONCE_HN:
        applicability = _predicate(base, ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE)
        selected = _resolve_event_references(base, applicability.evidence_references)
        if len(selected) < 2:
            return _empty_derived(
                base, request, TransformationStatus.CONSTRUCTION_ERROR,
                "same-attempt applicability lacks two resolvable nonce events",
                randomness_reference, adapter.family,
            )
        selected_ids = tuple(str(item["event_id"]) for item in selected[:2])
        target_ids, evidence_ids = (selected_ids[-1],), selected_ids
        boundary = {"relation": "same_attempt_protocol_recurrence"}
    elif request.variant_id == IDENTITY_COHERENT_HN:
        target = next((
            event for event in events
            if event.event_type is EventType.AUTHENTICATION_SUCCESS
            and event.result is EventResult.SUCCESS
            and event.authenticated_identity == context.device_id
        ), None)
        association = next((
            event for event in events
            if target is not None
            if event.auth_attempt_id == target.auth_attempt_id
            and event.timestamp < target.timestamp
            and event.device_id == context.device_id
        ), None)
        if target is None or association is None:
            return _empty_derived(
                base, request, TransformationStatus.CONSTRUCTION_ERROR,
                "persistent identity applicability lacks complete represented sources",
                randomness_reference, adapter.family,
            )
        target_ids, evidence_ids = (target.event_id,), (association.event_id, target.event_id)
        boundary = {"boundary_targeted": True}
    else:
        relation = resolve_timing_relation(base)
        assert relation is not None
        target_ids = (relation.target_event_id,)
        evidence_ids = (relation.predecessor_event_id, relation.target_event_id)
        boundary = {"epsilon_hn": request.configuration.epsilon_hn_s}

    # Compute the stable derived identity before candidate construction from
    # immutable transformed evidence and privileged construction provenance.
    provenance["counterfactual_group_reference"] = (
        counterfactual_group_reference(base.base_trace_reference, request.configuration)
        if request.variant_id == POSITIVE_MARGIN_HN else None
    )
    provenance_json = canonical_json(provenance)
    event_records_json = tuple(canonical_json(event.to_dict()) for event in events)
    derived_reference = _derived_reference(
        base, request.variant_id, randomness_reference,
        event_records_json, context_json, provenance_json,
    )
    candidate = _hn_candidate(
        context=context, variant_id=request.variant_id,
        derived_reference=derived_reference, target_ids=target_ids,
        evidence_ids=evidence_ids, boundary_parameters=boundary,
        construction_parameters={
            "adapter_version": provenance["adapter_version"],
            "base_trace_reference": base.base_trace_reference,
            "derived_trace_reference": derived_reference,
            "transformation_randomness_reference": randomness_reference,
        },
        counterfactual_reference=provenance.get("counterfactual_group_reference"),
    )
    qualified = qualify_hard_negative(events, context, candidate, report)
    final_report = _validation(events, context, qualification_records=[qualified])
    integration_status = (
        outcome.status
        if qualified.qualification_status is QualificationStatus.QUALIFIED
        and final_report.build_gate_passed
        and final_report.trace_ground_truth_statuses.get(context.trace_id)
        == "validated_normal"
        and not final_report.trace_validated_invariant_ids.get(context.trace_id, ())
        else TransformationStatus.CONSTRUCTION_ERROR
    )
    return DerivedRoleTrace(
        DERIVED_TRACE_REFERENCE_VERSION, derived_reference,
        base.base_trace_reference, request.variant_id, integration_status,
        event_records_json, context_json, provenance_json,
        tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ())),
        final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, qualified, outcome.reason,
    )
