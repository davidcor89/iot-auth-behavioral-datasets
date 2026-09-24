"""4D.3.3 profile-conditioned access-boundary integration.

The adapter derives access HNs and anomalies from immutable, independently
validated 4D.3.1 ACCESS bases.  It neither regenerates bases nor invokes the
legacy scenario templates.  Added HN evidence uses an isolated deterministic
timing namespace; anomaly changes remain local to one canonical request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import InjectionRecord, VARIANT_CAPABILITIES
from simulator.config.access_role_transformation import AccessRoleTransformationConfiguration
from simulator.event_model import AuthEvent, AuthState, EventResult, EventType
from simulator.hard_negative import QualificationStatus
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.direct_timing import (
    DERIVED_TRACE_REFERENCE_VERSION,
    DerivedRoleTrace,
    _derived_reference,
    _hn_candidate,
    _predicate,
    _validation,
    materialize_base,
)
from simulator.role_integration.models import ValidBaseTrace, canonical_json, stable_reference
from simulator.role_integration.pool import derive_future_transformation_reference
from simulator.role_integration.transformation import TransformationStatus
from simulator.security_context import AuthorizationDecision, ResourceOperationOutcome


ACCESS_EVIDENCE_RESOLVER_VERSION = "4d3.3-access-evidence-resolver-v1"
ACCESS_TARGET_SELECTION_POLICY_VERSION = "4d3.3-first-successful-request-v1"
ACCESS_HN_INSERTION_POLICY_VERSION = "4d3.3-after-last-base-success-v1"
ACCESS_HN_ADAPTER_VERSION = "4d3.3-access-hn-adapter-v1"
ACCESS_ANOMALY_ADAPTER_VERSION = "4d3.3-access-anomaly-adapter-v1"
ACCESS_ROLE_LOCAL_TIMING_VERSION = "4d3.3-role-local-access-timing-v1"
ACCESS_TRANSFORMATION_PROVENANCE_VERSION = "4d3.3-access-provenance-v1"
ACCESS_MATCHED_BOUNDARY_REFERENCE_VERSION = "4d3.3-matched-access-boundary-v1"

HN_VARIANTS = (
    "missing_auth_context_enforced", "invalid_credential_enforced",
    "inactive_session_enforced", "no_grant_enforced",
    "explicit_deny_enforced", "valid_context_legitimate_denial",
)
ANOMALY_VARIANTS = (
    "access_success_without_auth_context",
    "access_success_without_valid_credential",
    "access_success_without_active_session",
    "access_success_without_applicable_grant",
    "access_success_despite_explicit_deny",
    "access_success_without_security_prerequisites",
)
SUPPORTED_ACCESS_VARIANTS = HN_VARIANTS + ANOMALY_VARIANTS

MATCHED_VARIANT = {
    "missing_auth_context_enforced": "access_success_without_auth_context",
    "invalid_credential_enforced": "access_success_without_valid_credential",
    "inactive_session_enforced": "access_success_without_active_session",
    "no_grant_enforced": "access_success_without_applicable_grant",
    "explicit_deny_enforced": "access_success_despite_explicit_deny",
}


@dataclass(frozen=True)
class AccessRoleTransformationRequest:
    base_trace_reference: str
    variant_id: str
    transformation_seed: int
    configuration: AccessRoleTransformationConfiguration = (
        AccessRoleTransformationConfiguration()
    )


@dataclass(frozen=True)
class AccessOperationEvidence:
    request_ordinal: int
    operation_ordinal: int
    request_event_id: str
    operation_event_id: str
    access_request_id: str
    protected_session_id: str
    token_id: Optional[str]
    resource_id: str
    requested_action: str
    authenticated_context_active: bool
    token_context_active: bool
    protected_session_active: bool
    authorization_decision: str
    resource_operation_outcome: str


def resolve_access_operations(base: ValidBaseTrace) -> tuple[AccessOperationEvidence, ...]:
    """Resolve complete successful request/outcome relations in event order."""
    requests: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for ordinal, record in enumerate(base.event_records, 1):
        request_id = record.get("access_request_id")
        if record.get("event_type") == EventType.ACCESS_REQUEST.value and request_id:
            requests.setdefault(str(request_id), []).append((ordinal, record))
    resolved: list[AccessOperationEvidence] = []
    for operation_ordinal, operation in enumerate(base.event_records, 1):
        request_id = operation.get("access_request_id")
        if not (
            operation.get("event_type") == EventType.ACCESS_GRANTED.value
            and operation.get("result") == EventResult.SUCCESS.value
            and operation.get("resource_operation_outcome") == ResourceOperationOutcome.SUCCESS.value
            and request_id
        ):
            continue
        request = next((
            item for item in requests.get(str(request_id), ())
            if item[0] < operation_ordinal
            and item[1].get("device_id") == operation.get("device_id")
            and item[1].get("protected_session_id") == operation.get("protected_session_id")
            and item[1].get("resource_id") == operation.get("resource_id")
            and item[1].get("requested_action") == operation.get("requested_action")
        ), None)
        if request is None:
            continue
        request_ordinal, request_record = request
        required = (
            operation.get("protected_session_id"), operation.get("resource_id"),
            operation.get("requested_action"), operation.get("authorization_decision"),
        )
        complete_security_evidence = bool(
            operation.get("authentication_required") is True
            and operation.get("credential_required") is True
            and operation.get("active_session_required") is True
            and operation.get("authenticated_context_active") is True
            and operation.get("authenticated_identity") == operation.get("device_id")
            and operation.get("token_id")
            and operation.get("token_validation_result") == "validated"
            and operation.get("token_context_active") is True
            and operation.get("protected_session_active") is True
            and operation.get("authorization_applicable") is True
            and operation.get("authorization_evidence_complete") is True
            and operation.get("authorization_authoritative") is True
            and operation.get("authorization_decision") == AuthorizationDecision.GRANTED.value
        )
        if not all(required) or not complete_security_evidence:
            continue
        resolved.append(AccessOperationEvidence(
            request_ordinal=request_ordinal, operation_ordinal=operation_ordinal,
            request_event_id=str(request_record["event_id"]),
            operation_event_id=str(operation["event_id"]),
            access_request_id=str(request_id),
            protected_session_id=str(operation["protected_session_id"]),
            token_id=operation.get("token_id"),
            resource_id=str(operation["resource_id"]),
            requested_action=str(operation["requested_action"]),
            authenticated_context_active=bool(operation.get("authenticated_context_active")),
            token_context_active=bool(operation.get("token_context_active")),
            protected_session_active=bool(operation.get("protected_session_active")),
            authorization_decision=str(operation["authorization_decision"]),
            resource_operation_outcome=str(operation["resource_operation_outcome"]),
        ))
    return tuple(resolved)


def select_access_target(base: ValidBaseTrace) -> Optional[AccessOperationEvidence]:
    """Select the first complete success, independent of role and validation."""
    operations = resolve_access_operations(base)
    return operations[0] if operations else None


def matched_boundary_reference(base_reference: str, hn_variant: str) -> Optional[str]:
    anomaly = MATCHED_VARIANT.get(hn_variant)
    if anomaly is None:
        return None
    return stable_reference({
        "version": ACCESS_MATCHED_BOUNDARY_REFERENCE_VERSION,
        "base_trace_reference": base_reference,
        "hn_variant": hn_variant,
        "anomaly_variant": anomaly,
        "target_policy": ACCESS_TARGET_SELECTION_POLICY_VERSION,
    })


def _stable_id(kind: str, randomness_reference: str, variant: str, ordinal: int) -> str:
    return f"4d3.3-{kind}-{stable_reference((randomness_reference, variant, ordinal))[7:31]}"


def _role_times(
    events: list[AuthEvent], anchor_index: int, count: int,
    spacing: float,
) -> tuple[float, ...]:
    anchor = events[anchor_index].timestamp
    following = events[anchor_index + 1].timestamp if anchor_index + 1 < len(events) else None
    if following is None:
        return tuple(round(anchor + spacing * (index + 1), 6) for index in range(count))
    gap = following - anchor
    if gap <= 0:
        raise ValueError("base chronology does not provide a positive insertion interval")
    # Fractions preserve strict order without consuming ordinary T randomness.
    return tuple(round(anchor + gap * (index + 1) / (count + 1), 6) for index in range(count))


def _clone_access_event(
    template: AuthEvent, *, event_type: EventType, event_id: str,
    request_id: str, timestamp: float, previous_state: AuthState,
    new_state: AuthState, decision: str, outcome: str, reason: str,
) -> AuthEvent:
    return replace(
        template, event_type=event_type, event_id=event_id,
        access_request_id=request_id, timestamp=timestamp,
        observed_timestamp=timestamp, observed_timestamp_source="4d3.3_role_local_semantic_clock",
        delay_since_previous_event=0.0, previous_state=previous_state,
        new_state=new_state, result=EventResult.FAILURE,
        failure_reason=reason, authorization_decision=decision,
        resource_operation_outcome=outcome, anomaly_label=None,
        injection_id=None, anomaly_variant_name=None, invariant_families=(),
        source_context="normal",
    )


def _append_hn_attempt(
    base: ValidBaseTrace, variant: str, randomness_reference: str,
    configuration: AccessRoleTransformationConfiguration,
) -> tuple[list[AuthEvent], Any, dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    events, context = materialize_base(base)
    operations = resolve_access_operations(base)
    if not operations:
        raise ValueError("no complete successful access relation")
    source = operations[-1]
    request_template = events[source.request_ordinal - 1]
    operation_template = events[source.operation_ordinal - 1]
    anchor_index = source.operation_ordinal - 1
    added_close = False
    close_timestamp: Optional[float] = None

    if variant == "inactive_session_enforced":
        existing_close = next((
            (index, event) for index, event in enumerate(events)
            if index > anchor_index and event.event_type is EventType.SESSION_CLOSED
            and event.protected_session_id == source.protected_session_id
        ), None)
        if existing_close is not None:
            anchor_index = existing_close[0]
            close_timestamp = existing_close[1].timestamp
        else:
            times = _role_times(events, anchor_index, 3, configuration.local_event_spacing_s)
            close_event = replace(
                operation_template, event_type=EventType.SESSION_CLOSED,
                event_id=_stable_id("event", randomness_reference, variant, 0),
                timestamp=times[0], observed_timestamp=times[0],
                observed_timestamp_source="4d3.3_role_local_semantic_clock",
                delay_since_previous_event=round(times[0] - events[anchor_index].timestamp, 6),
                previous_state=AuthState.ACCESS_GRANTED, new_state=AuthState.ENROLLED,
                result=EventResult.SUCCESS, failure_reason=None,
                access_request_id=None, resource_operation_outcome=ResourceOperationOutcome.NOT_EXECUTED.value,
                authorization_decision=AuthorizationDecision.NOT_EVALUATED.value,
                authorization_evidence_complete=False, authorization_authoritative=False,
                authenticated_context_active=False, token_context_active=False,
                protected_session_active=False, anomaly_label=None, injection_id=None,
                anomaly_variant_name=None, invariant_families=(), source_context="normal",
            )
            events.insert(anchor_index + 1, close_event)
            anchor_index += 1
            added_close = True
            close_timestamp = close_event.timestamp

    times = _role_times(events, anchor_index, 2, configuration.local_event_spacing_s)
    request_id = _stable_id("access-request", randomness_reference, variant, 1)
    reason_by_variant = {
        "missing_auth_context_enforced": "missing_authenticated_context",
        "invalid_credential_enforced": "rejected_credential_context",
        "inactive_session_enforced": "inactive_protected_session",
        "no_grant_enforced": "no_applicable_grant",
        "explicit_deny_enforced": "explicit_authorization_denial",
        "valid_context_legitimate_denial": "legitimate_authorization_denial",
    }
    reason = reason_by_variant[variant]
    request = _clone_access_event(
        request_template, event_type=EventType.ACCESS_REQUEST,
        event_id=_stable_id("event", randomness_reference, variant, 1),
        request_id=request_id, timestamp=times[0],
        previous_state=AuthState.ACCESS_GRANTED, new_state=AuthState.ACCESS_REQUESTED,
        decision=AuthorizationDecision.NOT_EVALUATED.value,
        outcome=ResourceOperationOutcome.NOT_EXECUTED.value, reason=reason,
    )
    denial = _clone_access_event(
        operation_template, event_type=EventType.ACCESS_DENIED,
        event_id=_stable_id("event", randomness_reference, variant, 2),
        request_id=request_id, timestamp=times[1],
        previous_state=AuthState.ACCESS_REQUESTED, new_state=AuthState.ACCESS_DENIED,
        decision=AuthorizationDecision.DENIED.value,
        outcome=ResourceOperationOutcome.NOT_EXECUTED.value, reason=reason,
    )
    request.delay_since_previous_event = round(times[0] - events[anchor_index].timestamp, 6)
    denial.delay_since_previous_event = round(times[1] - times[0], 6)
    for event in (request, denial):
        event.authorization_applicable = True
        event.authorization_evidence_complete = True
        event.authorization_authoritative = True
        event.authentication_required = True
        event.credential_required = True
        event.active_session_required = True
    if variant == "missing_auth_context_enforced":
        for event in (request, denial):
            event.authenticated_context_active = False
            event.authenticated_identity = None
            # The current binding validator treats any token-bearing event
            # without active authenticated context as underdetermined.  This
            # rejected attempt therefore does not claim credential use; the
            # untouched base retains the valid credential/session lineage.
            event.token_id = None
            event.token_context_active = False
            event.token_validation_result = "not_evaluated"
            event.token_bound_identity = None
            event.token_bound_protected_session_id = None
    elif variant == "invalid_credential_enforced":
        for event in (request, denial):
            event.token_context_active = False
            event.token_validation_result = "rejected"
    elif variant == "inactive_session_enforced":
        for event in (request, denial):
            event.protected_session_active = False
    events[anchor_index + 1:anchor_index + 1] = [request, denial]
    context = replace(
        context,
        access_request_ids=tuple(context.access_request_ids) + (request_id,),
        authorization_decisions=tuple(context.authorization_decisions) + (AuthorizationDecision.DENIED.value,),
        resource_operation_outcomes=tuple(context.resource_operation_outcomes) + (ResourceOperationOutcome.NOT_EXECUTED.value,),
        authorization_decision=AuthorizationDecision.DENIED.value,
        resource_operation_outcome=ResourceOperationOutcome.NOT_EXECUTED.value,
        protected_session_active=(False if variant == "inactive_session_enforced" else context.protected_session_active),
        authenticated_context_active=(False if variant == "inactive_session_enforced" else context.authenticated_context_active),
        token_context_active=(False if variant == "inactive_session_enforced" else context.token_context_active),
        protected_session_ended_at=(close_timestamp if variant == "inactive_session_enforced" else context.protected_session_ended_at),
        trace_ended_at=max(context.trace_ended_at or times[-1], times[-1]),
    )
    provenance = {
        "provenance_version": ACCESS_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": ACCESS_HN_ADAPTER_VERSION,
        "resolver_version": ACCESS_EVIDENCE_RESOLVER_VERSION,
        "insertion_policy_version": ACCESS_HN_INSERTION_POLICY_VERSION,
        "role_local_timing_version": ACCESS_ROLE_LOCAL_TIMING_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": variant,
        "base_successful_operation_count": len(operations),
        "role_added_rejected_operation_count": 1,
        "derived_successful_operation_count": len(operations),
        "total_derived_access_attempts": len(operations) + 1,
        "insertion_anchor_event_id": events[anchor_index].event_id,
        "semantic_predecessor_event_id": events[anchor_index].event_id,
        "added_request_id": request_id,
        "added_request_event_id": request.event_id,
        "added_operation_event_id": denial.event_id,
        "role_local_semantic_timestamps": list(times),
        "role_local_close_timestamp": close_timestamp if added_close else None,
        "transformation_randomness_reference": randomness_reference,
        "base_timestamps_rewritten": False,
        "ordinary_base_timing_provenance_preserved": True,
        "base_timing_provenance_reference": stable_reference(base.timing_provenance),
        "role_local_close_added": added_close,
        "configuration_reference": configuration.configuration_reference,
        "applied": True,
    }
    return events, context, provenance, (denial.event_id,), (request.event_id, denial.event_id)


def _transform_anomaly(
    base: ValidBaseTrace, variant: str, randomness_reference: str,
    configuration: AccessRoleTransformationConfiguration,
) -> tuple[list[AuthEvent], Any, dict[str, Any], AccessOperationEvidence]:
    events, context = materialize_base(base)
    target = select_access_target(base)
    if target is None:
        raise ValueError("no complete successful access relation")
    request = events[target.request_ordinal - 1]
    operation = events[target.operation_ordinal - 1]
    original = {"request": request.to_dict(), "operation": operation.to_dict()}
    affected = (request, operation)
    if variant in {"access_success_without_auth_context", "access_success_without_security_prerequisites"}:
        for event in affected:
            event.authenticated_context_active = False
            event.authenticated_identity = None
    if variant in {"access_success_without_valid_credential", "access_success_without_security_prerequisites"}:
        for event in affected:
            event.token_context_active = False
            event.token_validation_result = "rejected"
    if variant in {"access_success_without_active_session", "access_success_without_security_prerequisites"}:
        for event in affected:
            event.protected_session_active = False
    if variant == "access_success_without_applicable_grant":
        for event in affected:
            event.authorization_decision = AuthorizationDecision.NO_APPLICABLE_GRANT.value
            event.authorization_evidence_complete = True
            event.authorization_authoritative = True
    elif variant == "access_success_despite_explicit_deny":
        for event in affected:
            event.authorization_decision = AuthorizationDecision.DENIED.value
            event.authorization_evidence_complete = True
            event.authorization_authoritative = True
    capability = VARIANT_CAPABILITIES[variant]
    injection_id = _stable_id("injection", randomness_reference, variant, 0)
    operation.injection_id = injection_id
    operation.anomaly_variant_name = variant
    operation.invariant_families = capability.invariant_families
    operation.source_context = "controlled_injection"
    injection = InjectionRecord(
        injection_id=injection_id, historical_scenario_name="access_without_auth",
        anomaly_variant_name=variant, capability_status=capability.status.value,
        mapped_semantic_concept=capability.semantic_concept,
        invariant_families=capability.invariant_families,
        canonical_invariant_ids=capability.invariant_ids,
        targeted_event_type=EventType.ACCESS_GRANTED.value,
        targeted_action_or_context="request-scoped protected resource operation",
        source_evidence_event_ids=(request.event_id, operation.event_id),
        source_material_identifier=target.access_request_id,
        source_trace_id=context.trace_id, source_session_id=context.legacy_session_id,
        source_auth_attempt_id=operation.auth_attempt_id,
        original_value="valid_reference_access_context", injected_value=variant,
        injection_parameters={
            "adapter_version": ACCESS_ANOMALY_ADAPTER_VERSION,
            "resolver_version": ACCESS_EVIDENCE_RESOLVER_VERSION,
            "target_selection_policy_version": ACCESS_TARGET_SELECTION_POLICY_VERSION,
            "configuration_reference": configuration.configuration_reference,
            "transformation_randomness_reference": randomness_reference,
        },
        trace_id=context.trace_id, device_id=context.device_id,
        session_id=context.legacy_session_id, auth_attempt_id=operation.auth_attempt_id,
        declared_history_scope=capability.history_scope,
        intended_observable_violation=capability.semantic_concept,
        transformation_applied=True,
    )
    context = replace(
        context, attack_type=variant, attack_phase=EventType.ACCESS_GRANTED.value,
        severity="high", attack_scenario_intent=True, observable_anomaly=False,
        observable_violation_candidate=False, injection_records=(injection.to_dict(),),
    )
    provenance = {
        "provenance_version": ACCESS_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": ACCESS_ANOMALY_ADAPTER_VERSION,
        "resolver_version": ACCESS_EVIDENCE_RESOLVER_VERSION,
        "target_selection_policy_version": ACCESS_TARGET_SELECTION_POLICY_VERSION,
        "base_trace_reference": base.base_trace_reference, "variant_id": variant,
        "selected_request_reference": f"event:{target.request_ordinal}",
        "selected_operation_reference": f"event:{target.operation_ordinal}",
        "selected_request_id": target.access_request_id,
        "original_request_evidence": original,
        "transformed_request_evidence": {
            "request": request.to_dict(), "operation": operation.to_dict(),
        },
        "base_successful_operation_count": len(resolve_access_operations(base)),
        "derived_successful_operation_count": len(resolve_access_operations(base)),
        "total_derived_access_attempts": len(resolve_access_operations(base)),
        "transformation_randomness_reference": randomness_reference,
        "configuration_reference": configuration.configuration_reference,
        "event_count_changed": False, "base_timestamps_rewritten": False,
        "applied": True,
    }
    return events, context, provenance, target


def _empty(
    base: ValidBaseTrace, request: AccessRoleTransformationRequest,
    status: TransformationStatus, reason: str, randomness_reference: str,
) -> DerivedRoleTrace:
    provenance_json = canonical_json({
        "provenance_version": ACCESS_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": (
            ACCESS_HN_ADAPTER_VERSION
            if request.variant_id in HN_VARIANTS else ACCESS_ANOMALY_ADAPTER_VERSION
        ),
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "status": status.value,
        "transformation_randomness_reference": randomness_reference,
        "configuration_reference": request.configuration.configuration_reference,
        "applied": False,
        "reason": reason,
    })
    derived_reference = _derived_reference(
        base, request.variant_id, randomness_reference,
        base.event_records_json, base.session_context_json, provenance_json,
    )
    return DerivedRoleTrace(
        DERIVED_TRACE_REFERENCE_VERSION, derived_reference,
        base.base_trace_reference, request.variant_id, status,
        base.event_records_json, base.session_context_json, provenance_json,
        (), None, False, None, reason,
    )


def integrate_access_role(
    base: ValidBaseTrace, request: AccessRoleTransformationRequest,
) -> DerivedRoleTrace:
    """Construct once, independently validate, then qualify/consolidate."""
    if request.base_trace_reference != base.base_trace_reference:
        raise ValueError("request/base reference mismatch")
    randomness_reference = derive_future_transformation_reference(
        transformation_seed=request.transformation_seed,
        base_trace_reference=base.base_trace_reference,
        transformation_key=f"{request.variant_id}|{request.configuration.configuration_reference}",
    )
    if request.variant_id not in SUPPORTED_ACCESS_VARIANTS:
        return _empty(base, request, TransformationStatus.UNSUPPORTED,
                      "variant is outside the frozen 4D.3.3 integration scope",
                      randomness_reference)
    applicability = _predicate(base, ApplicabilityPredicate.PROTECTED_ACCESS)
    if not applicability.applicable:
        return _empty(base, request, TransformationStatus.NOT_APPLICABLE,
                      applicability.reason, randomness_reference)
    if not resolve_access_operations(base):
        return _empty(base, request, TransformationStatus.UNSUPPORTED,
                      "represented access evidence is not request-scoped and complete",
                      randomness_reference)
    try:
        if request.variant_id in HN_VARIANTS:
            events, context, provenance, target_ids, evidence_ids = _append_hn_attempt(
                base, request.variant_id, randomness_reference, request.configuration,
            )
            report = _validation(events, context)
            provenance.update({
                "applicability": applicability.to_record(),
                "matched_boundary_reference": matched_boundary_reference(
                    base.base_trace_reference, request.variant_id,
                ),
            })
            provenance_json = canonical_json(provenance)
            event_json = tuple(canonical_json(event.to_dict()) for event in events)
            context_json = canonical_json(context.to_dict())
            derived_reference = _derived_reference(
                base, request.variant_id, randomness_reference,
                event_json, context_json, provenance_json,
            )
            candidate = _hn_candidate(
                context=context, variant_id=request.variant_id,
                derived_reference=derived_reference, target_ids=target_ids,
                evidence_ids=evidence_ids,
                boundary_parameters={"relation": "request_scoped_authoritative_enforcement"},
                construction_parameters={
                    "adapter_version": ACCESS_HN_ADAPTER_VERSION,
                    "base_trace_reference": base.base_trace_reference,
                    "derived_trace_reference": derived_reference,
                    "transformation_randomness_reference": randomness_reference,
                },
                counterfactual_reference=provenance.get("matched_boundary_reference"),
            )
            qualified = qualify_hard_negative(events, context, candidate, report)
            provenance.update({
                "independent_validation_status": report.trace_ground_truth_statuses.get(context.trace_id),
                "independent_validated_invariant_ids": list(
                    report.trace_validated_invariant_ids.get(context.trace_id, ())
                ),
                "qualification_status": qualified.qualification_status.value,
                "qualification_reason": qualified.qualification_reason,
            })
            provenance_json = canonical_json(provenance)
            derived_reference = _derived_reference(
                base, request.variant_id, randomness_reference,
                event_json, context_json, provenance_json,
            )
            qualified = replace(
                qualified, hard_negative_candidate_id=derived_reference,
            )
            final_report = _validation(events, context, qualification_records=[qualified])
            status = (
                TransformationStatus.APPLIED
                if qualified.qualification_status is QualificationStatus.QUALIFIED
                and final_report.build_gate_passed
                and final_report.trace_ground_truth_statuses.get(context.trace_id) == "validated_normal"
                and not final_report.trace_validated_invariant_ids.get(context.trace_id, ())
                else TransformationStatus.CONSTRUCTION_ERROR
            )
            return DerivedRoleTrace(
                DERIVED_TRACE_REFERENCE_VERSION, derived_reference,
                base.base_trace_reference, request.variant_id, status,
                event_json, context_json, provenance_json,
                tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ())),
                final_report.trace_ground_truth_statuses.get(context.trace_id),
                final_report.build_gate_passed, qualified,
                "one request-scoped rejected attempt added to immutable base activity",
            )

        events, context, provenance, target = _transform_anomaly(
            base, request.variant_id, randomness_reference, request.configuration,
        )
        evidence_report = _validation(events, context)
        capability = VARIANT_CAPABILITIES[request.variant_id]
        expected = set(capability.invariant_ids)
        confirmed = set(evidence_report.trace_validated_invariant_ids.get(context.trace_id, ()))
        if not expected.issubset(confirmed):
            return _empty(base, request, TransformationStatus.CONSTRUCTION_ERROR,
                          "independent validation did not confirm intended access invariant(s)",
                          randomness_reference)
        for event in events:
            if event.event_id in evidence_report.validated_positive_event_ids:
                event.anomaly_label = request.variant_id
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
        final_report = _validation(events, context)
        validated = tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ()))
        provenance.update({
            "applicability": applicability.to_record(),
            "injection_id": final_injection["injection_id"],
            "validated_invariant_ids": list(validated),
            "ground_truth_consolidation": "independent_evidence_pass_then_final_validation",
            "matched_boundary_reference": next((
                matched_boundary_reference(base.base_trace_reference, hn)
                for hn, anomaly in MATCHED_VARIANT.items() if anomaly == request.variant_id
            ), None),
        })
        provenance_json = canonical_json(provenance)
        event_json = tuple(canonical_json(event.to_dict()) for event in events)
        context_json = canonical_json(context.to_dict())
        derived_reference = _derived_reference(
            base, request.variant_id, randomness_reference,
            event_json, context_json, provenance_json,
        )
        status = (
            TransformationStatus.APPLIED
            if final_report.build_gate_passed
            and final_report.trace_ground_truth_statuses.get(context.trace_id) == "validated_positive"
            and expected.issubset(set(validated))
            else TransformationStatus.CONSTRUCTION_ERROR
        )
        return DerivedRoleTrace(
            DERIVED_TRACE_REFERENCE_VERSION, derived_reference,
            base.base_trace_reference, request.variant_id, status,
            event_json, context_json, provenance_json, validated,
            final_report.trace_ground_truth_statuses.get(context.trace_id),
            final_report.build_gate_passed, None,
            f"canonical request {target.access_request_id} transformed once",
        )
    except (KeyError, TypeError, ValueError) as error:
        return _empty(base, request, TransformationStatus.CONSTRUCTION_ERROR,
                      str(error), randomness_reference)
