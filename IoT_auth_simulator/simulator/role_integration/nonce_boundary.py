"""4D.3.4 cross-attempt nonce boundary integration over immutable bases."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import InjectionRecord, VARIANT_CAPABILITIES
from simulator.config.nonce_role_transformation import NonceRoleTransformationConfiguration
from simulator.event_model import AuthEvent, EventResult, EventType
from simulator.hard_negative import QualificationStatus
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.direct_timing import (
    DerivedRoleTrace, _hn_candidate, _predicate, _validation, materialize_base,
)
from simulator.role_integration.models import (
    ValidBaseTrace, canonical_json, normalized_event_records, stable_reference,
)
from simulator.role_integration.pool import derive_future_transformation_reference
from simulator.role_integration.transformation import TransformationStatus


NONCE_EVIDENCE_RESOLVER_VERSION = "4d3.4-nonce-evidence-resolver-v1"
NONCE_ATTEMPT_PAIR_POLICY_VERSION = "4d3.4-first-eligible-ordered-attempt-pair-v1"
NONCE_BOUNDARY_ADAPTER_VERSION = "4d3.4-cross-attempt-nonce-adapter-v1"
NONCE_TRANSFORMATION_PROVENANCE_VERSION = "4d3.4-nonce-provenance-v1"
NONCE_DERIVED_REFERENCE_VERSION = "4d3.4-nonce-derived-reference-sha256-v1"
NONCE_OPPORTUNITY_SOURCE_CLASSIFIER_VERSION = "4d3.4-attempt-source-classifier-v1"
NONCE_MATCHED_BOUNDARY_REFERENCE_VERSION = "4d3.4-matched-nonce-boundary-v1"

FRESH_NONCE_HN = "fresh_nonce_cross_attempt"
NONCE_REUSE_ANOMALY = "nonce_reuse"
SUPPORTED_NONCE_VARIANTS = (FRESH_NONCE_HN, NONCE_REUSE_ANOMALY)


@dataclass(frozen=True)
class NonceRoleTransformationRequest:
    base_trace_reference: str
    variant_id: str
    transformation_seed: int
    configuration: NonceRoleTransformationConfiguration = (
        NonceRoleTransformationConfiguration()
    )


@dataclass(frozen=True)
class AuthenticationAttemptEvidence:
    attempt_id: str
    first_ordinal: int
    last_ordinal: int
    nonce_received_ordinal: int
    nonce_received_event_id: str
    response_ordinals: tuple[int, ...]
    response_event_ids: tuple[str, ...]
    nonce_bearing_ordinals: tuple[int, ...]
    nonce_bearing_event_ids: tuple[str, ...]
    nonce: str
    outcome: str
    outcome_event_id: Optional[str]
    source_kind: str
    link_event_id: Optional[str]
    semantic_order_valid: bool
    observed_order_valid: bool


@dataclass(frozen=True)
class NonceAttemptPair:
    earlier: AuthenticationAttemptEvidence
    later: AuthenticationAttemptEvidence
    relationship: str


def _observed(record: Mapping[str, Any]) -> float:
    value = record.get("observed_timestamp")
    return float(record["timestamp"] if value is None else value)


def resolve_authentication_attempts(
    base: ValidBaseTrace,
) -> tuple[AuthenticationAttemptEvidence, ...]:
    """Reconstruct explicit attempt-linked nonce evidence in event order."""
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for ordinal, record in enumerate(base.event_records, 1):
        attempt_id = record.get("auth_attempt_id")
        if attempt_id:
            groups.setdefault(str(attempt_id), []).append((ordinal, record))
    attempts: list[AuthenticationAttemptEvidence] = []
    for attempt_id, entries in groups.items():
        nonce_received = [
            item for item in entries
            if item[1].get("event_type") == EventType.NONCE_RECEIVED.value
            and item[1].get("nonce")
        ]
        responses = [
            item for item in entries
            if item[1].get("event_type") == EventType.RESPONSE_SENT.value
            and item[1].get("nonce")
        ]
        nonce_bearing = [item for item in entries if item[1].get("nonce")]
        if len(nonce_received) != 1 or not responses or not nonce_bearing:
            continue
        nonce = str(nonce_received[0][1]["nonce"])
        if any(str(record["nonce"]) != nonce for _, record in nonce_bearing):
            continue
        links = [
            item for item in entries
            if item[1].get("event_type") in {
                EventType.RETRY.value, EventType.RENEWAL_REQUEST.value,
            }
            and item[1].get("result") == EventResult.SUCCESS.value
        ]
        source_kind = "initial"
        if links:
            source_kind = (
                "retry" if links[0][1]["event_type"] == EventType.RETRY.value
                else "renewal"
            )
        outcomes = [
            item for item in entries
            if item[1].get("event_type") in {
                EventType.AUTHENTICATION_SUCCESS.value,
                EventType.AUTHENTICATION_FAILURE.value,
            }
        ]
        ordered = sorted(entries)
        semantic = all(
            float(left[1]["timestamp"]) < float(right[1]["timestamp"])
            for left, right in zip(ordered, ordered[1:])
        )
        observed = all(
            _observed(left[1]) < _observed(right[1])
            for left, right in zip(ordered, ordered[1:])
        )
        attempts.append(AuthenticationAttemptEvidence(
            attempt_id=attempt_id, first_ordinal=entries[0][0],
            last_ordinal=entries[-1][0],
            nonce_received_ordinal=nonce_received[0][0],
            nonce_received_event_id=str(nonce_received[0][1]["event_id"]),
            response_ordinals=tuple(item[0] for item in responses),
            response_event_ids=tuple(str(item[1]["event_id"]) for item in responses),
            nonce_bearing_ordinals=tuple(item[0] for item in nonce_bearing),
            nonce_bearing_event_ids=tuple(str(item[1]["event_id"]) for item in nonce_bearing),
            nonce=nonce,
            outcome=(str(outcomes[-1][1]["event_type"]) if outcomes else "not_represented"),
            outcome_event_id=(str(outcomes[-1][1]["event_id"]) if outcomes else None),
            source_kind=source_kind,
            link_event_id=(str(links[0][1]["event_id"]) if links else None),
            semantic_order_valid=semantic, observed_order_valid=observed,
        ))
    return tuple(sorted(attempts, key=lambda item: item.first_ordinal))


def select_nonce_attempt_pair(base: ValidBaseTrace) -> Optional[NonceAttemptPair]:
    """Select the first eligible ordered pair before role construction."""
    attempts = resolve_authentication_attempts(base)
    for earlier_index, earlier in enumerate(attempts):
        for later in attempts[earlier_index + 1:]:
            if (
                earlier.attempt_id != later.attempt_id
                and earlier.semantic_order_valid and later.semantic_order_valid
                and earlier.observed_order_valid and later.observed_order_valid
                and later.source_kind in {"retry", "renewal"}
                and later.link_event_id
                and earlier.last_ordinal < later.nonce_received_ordinal
            ):
                return NonceAttemptPair(earlier, later, later.source_kind)
    return None


def classify_multi_attempt_source(base: ValidBaseTrace) -> str:
    """Classify represented multi-attempt causes for privileged diagnostics."""
    types = {record.get("event_type") for record in base.event_records}
    retry = EventType.RETRY.value in types
    renewal = EventType.RENEWAL_REQUEST.value in types
    if retry and renewal:
        return "retry_and_renewal"
    if retry:
        return "retry_only"
    if renewal:
        return "renewal_only"
    return "other"


def nonce_matched_boundary_reference(base_reference: str) -> str:
    return stable_reference({
        "version": NONCE_MATCHED_BOUNDARY_REFERENCE_VERSION,
        "base_trace_reference": base_reference,
        "variants": SUPPORTED_NONCE_VARIANTS,
        "pair_policy": NONCE_ATTEMPT_PAIR_POLICY_VERSION,
    })


def _derived_reference(
    base: ValidBaseTrace, request: NonceRoleTransformationRequest,
    pair: Optional[NonceAttemptPair], randomness_reference: str,
    event_json: tuple[str, ...], context_json: str, provenance_json: str,
) -> str:
    records = tuple(json.loads(item) for item in event_json)
    provenance = json.loads(provenance_json)
    return stable_reference({
        "reference_version": NONCE_DERIVED_REFERENCE_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "pair_policy_version": NONCE_ATTEMPT_PAIR_POLICY_VERSION,
        "selected_attempt_references": (
            (
                f"attempt-start:event:{pair.earlier.first_ordinal}",
                f"attempt-start:event:{pair.later.first_ordinal}",
            ) if pair else None
        ),
        "configuration_reference": request.configuration.configuration_reference,
        "transformation_randomness_reference": randomness_reference,
        "normalized_derived_evidence": normalized_event_records(records),
        "construction_mode": provenance.get("construction_mode"),
        "transformed_event_references": [
            f"event:{index + 1}" for index, record in enumerate(records)
            if record.get("event_id") in set(provenance.get("transformed_nonce_event_ids", ()))
        ],
        "validated_invariant_ids": provenance.get("validated_invariant_ids", ()),
    })


def _empty(
    base: ValidBaseTrace, request: NonceRoleTransformationRequest,
    status: TransformationStatus, reason: str, randomness_reference: str,
) -> DerivedRoleTrace:
    provenance_json = canonical_json({
        "provenance_version": NONCE_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": NONCE_BOUNDARY_ADAPTER_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id, "status": status.value,
        "transformation_randomness_reference": randomness_reference,
        "configuration_reference": request.configuration.configuration_reference,
        "applied": False, "reason": reason,
    })
    reference = _derived_reference(
        base, request, None, randomness_reference,
        base.event_records_json, base.session_context_json, provenance_json,
    )
    return DerivedRoleTrace(
        NONCE_DERIVED_REFERENCE_VERSION, reference,
        base.base_trace_reference, request.variant_id, status,
        base.event_records_json, base.session_context_json, provenance_json,
        (), None, False, None, reason,
    )


def integrate_nonce_role(
    base: ValidBaseTrace, request: NonceRoleTransformationRequest,
) -> DerivedRoleTrace:
    """Derive one frozen nonce role without changing multi-attempt structure."""
    if request.base_trace_reference != base.base_trace_reference:
        raise ValueError("request/base reference mismatch")
    randomness_reference = derive_future_transformation_reference(
        transformation_seed=request.transformation_seed,
        base_trace_reference=base.base_trace_reference,
        transformation_key=f"{request.variant_id}|{request.configuration.configuration_reference}",
    )
    if request.variant_id not in SUPPORTED_NONCE_VARIANTS:
        return _empty(base, request, TransformationStatus.UNSUPPORTED,
                      "variant is outside the frozen 4D.3.4 integration scope",
                      randomness_reference)
    broad = _predicate(base, ApplicabilityPredicate.DISTINCT_AUTHENTICATION_ATTEMPTS)
    if not broad.applicable:
        return _empty(base, request, TransformationStatus.NOT_APPLICABLE,
                      broad.reason, randomness_reference)
    pair = select_nonce_attempt_pair(base)
    if pair is None:
        return _empty(base, request, TransformationStatus.UNSUPPORTED,
                      "distinct attempts lack a complete supported ordered nonce pair",
                      randomness_reference)
    events, context = materialize_base(base)
    original_nonce_evidence = {
        "earlier_attempt_id": pair.earlier.attempt_id,
        "later_attempt_id": pair.later.attempt_id,
        "earlier_nonce": pair.earlier.nonce,
        "later_nonce": pair.later.nonce,
        "earlier_nonce_event_ids": list(pair.earlier.nonce_bearing_event_ids),
        "later_nonce_event_ids": list(pair.later.nonce_bearing_event_ids),
    }
    provenance: dict[str, Any] = {
        "provenance_version": NONCE_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": NONCE_BOUNDARY_ADAPTER_VERSION,
        "resolver_version": NONCE_EVIDENCE_RESOLVER_VERSION,
        "attempt_pair_policy_version": NONCE_ATTEMPT_PAIR_POLICY_VERSION,
        "opportunity_source_classifier_version": NONCE_OPPORTUNITY_SOURCE_CLASSIFIER_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "broad_applicability": broad.to_record(),
        "opportunity_source": classify_multi_attempt_source(base),
        "selected_attempt_relationship": pair.relationship,
        "selected_earlier_attempt_id": pair.earlier.attempt_id,
        "selected_later_attempt_id": pair.later.attempt_id,
        "link_event_id": pair.later.link_event_id,
        "original_nonce_evidence": original_nonce_evidence,
        "transformation_randomness_reference": randomness_reference,
        "configuration_reference": request.configuration.configuration_reference,
        "matched_boundary_reference": nonce_matched_boundary_reference(
            base.base_trace_reference
        ),
        "event_count_changed": False,
        "event_order_changed": False,
        "timestamps_rewritten": False,
        "ordinary_base_timing_provenance_preserved": True,
        "base_timing_provenance_reference": stable_reference(base.timing_provenance),
        "applied": True,
    }

    if request.variant_id == FRESH_NONCE_HN:
        if pair.earlier.nonce == pair.later.nonce:
            return _empty(base, request, TransformationStatus.CONSTRUCTION_ERROR,
                          "selected base pair does not establish fresh nonce material",
                          randomness_reference)
        provenance.update({
            "construction_mode": "reference_only",
            "transformed_nonce_event_ids": [],
            "derived_nonce_evidence": original_nonce_evidence,
        })
        report = _validation(events, context)
        event_json = base.event_records_json
        context_json = base.session_context_json
        provisional_json = canonical_json(provenance)
        provisional_reference = _derived_reference(
            base, request, pair, randomness_reference,
            event_json, context_json, provisional_json,
        )
        candidate = _hn_candidate(
            context=context, variant_id=FRESH_NONCE_HN,
            derived_reference=provisional_reference,
            target_ids=(pair.later.nonce_received_event_id,),
            evidence_ids=(
                pair.earlier.nonce_received_event_id,
                str(pair.later.link_event_id),
                pair.later.nonce_received_event_id,
            ),
            boundary_parameters={"attempt_relationship": pair.relationship},
            construction_parameters={
                "adapter_version": NONCE_BOUNDARY_ADAPTER_VERSION,
                "base_trace_reference": base.base_trace_reference,
                "transformation_randomness_reference": randomness_reference,
                "construction_mode": "reference_only",
            },
            counterfactual_reference=provenance["matched_boundary_reference"],
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
        reference = _derived_reference(
            base, request, pair, randomness_reference,
            event_json, context_json, provenance_json,
        )
        qualified = replace(qualified, hard_negative_candidate_id=reference)
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
            NONCE_DERIVED_REFERENCE_VERSION, reference,
            base.base_trace_reference, request.variant_id, status,
            event_json, context_json, provenance_json,
            tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ())),
            final_report.trace_ground_truth_statuses.get(context.trace_id),
            final_report.build_gate_passed, qualified,
            "fresh cross-attempt nonce evidence referenced without mutation",
        )

    # Anomaly: copy the earlier nonce into every nonce-bearing event of the
    # selected later attempt so within-attempt recurrence remains coherent.
    transformed_ids: list[str] = []
    target: Optional[AuthEvent] = None
    for ordinal in pair.later.nonce_bearing_ordinals:
        event = events[ordinal - 1]
        event.nonce = pair.earlier.nonce
        transformed_ids.append(event.event_id)
        if ordinal == pair.later.nonce_received_ordinal:
            target = event
    if target is None:
        return _empty(base, request, TransformationStatus.CONSTRUCTION_ERROR,
                      "selected later nonce-received event did not materialize",
                      randomness_reference)
    capability = VARIANT_CAPABILITIES[NONCE_REUSE_ANOMALY]
    injection_id = "4d3.4-injection-" + stable_reference({
        "base": base.base_trace_reference,
        "variant": NONCE_REUSE_ANOMALY,
        "pair": (pair.earlier.attempt_id, pair.later.attempt_id),
        "randomness": randomness_reference,
    })[7:31]
    target.injection_id = injection_id
    target.anomaly_variant_name = NONCE_REUSE_ANOMALY
    target.invariant_families = capability.invariant_families
    target.source_context = "controlled_injection"
    derived_nonce_evidence = {
        **original_nonce_evidence,
        "later_nonce": pair.earlier.nonce,
        "transformed_later_nonce_event_ids": transformed_ids,
    }
    injection = InjectionRecord(
        injection_id=injection_id,
        historical_scenario_name=NONCE_REUSE_ANOMALY,
        anomaly_variant_name=NONCE_REUSE_ANOMALY,
        capability_status=capability.status.value,
        mapped_semantic_concept=capability.semantic_concept,
        invariant_families=capability.invariant_families,
        canonical_invariant_ids=capability.invariant_ids,
        targeted_event_type=EventType.NONCE_RECEIVED.value,
        targeted_action_or_context="later selected authentication attempt nonce evidence",
        source_evidence_event_ids=(pair.earlier.nonce_received_event_id,),
        source_material_identifier=pair.earlier.nonce,
        source_trace_id=context.trace_id,
        source_session_id=context.legacy_session_id,
        source_auth_attempt_id=pair.earlier.attempt_id,
        original_value=pair.later.nonce, injected_value=pair.earlier.nonce,
        injection_parameters={
            "adapter_version": NONCE_BOUNDARY_ADAPTER_VERSION,
            "resolver_version": NONCE_EVIDENCE_RESOLVER_VERSION,
            "attempt_pair_policy_version": NONCE_ATTEMPT_PAIR_POLICY_VERSION,
            "transformed_event_ids": transformed_ids,
            "configuration_reference": request.configuration.configuration_reference,
            "transformation_randomness_reference": randomness_reference,
        },
        trace_id=context.trace_id, device_id=context.device_id,
        session_id=context.legacy_session_id,
        auth_attempt_id=pair.later.attempt_id,
        declared_history_scope=capability.history_scope,
        intended_observable_violation=capability.semantic_concept,
        transformation_applied=True,
    )
    context = replace(
        context, attack_type=NONCE_REUSE_ANOMALY,
        attack_phase=EventType.NONCE_RECEIVED.value, severity="medium",
        attack_scenario_intent=True, observable_anomaly=False,
        observable_violation_candidate=False,
        nonce_age_at_reuse=round(target.timestamp - events[pair.earlier.nonce_received_ordinal - 1].timestamp, 6),
        injection_records=(injection.to_dict(),),
    )
    provenance.update({
        "construction_mode": "nonce_local_reuse_transformation",
        "transformed_nonce_event_ids": transformed_ids,
        "derived_nonce_evidence": derived_nonce_evidence,
        "injection_id": injection_id,
    })
    evidence_report = _validation(events, context)
    expected = set(capability.invariant_ids)
    confirmed = set(evidence_report.trace_validated_invariant_ids.get(context.trace_id, ()))
    if not expected.issubset(confirmed):
        return _empty(base, request, TransformationStatus.CONSTRUCTION_ERROR,
                      "independent validation did not confirm cross-attempt nonce reuse",
                      randomness_reference)
    for event in events:
        if event.event_id in evidence_report.validated_positive_event_ids:
            event.anomaly_label = NONCE_REUSE_ANOMALY
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
        "validated_invariant_ids": list(validated),
        "ground_truth_consolidation": "independent_evidence_pass_then_final_validation",
    })
    provenance_json = canonical_json(provenance)
    event_json = tuple(canonical_json(event.to_dict()) for event in events)
    context_json = canonical_json(context.to_dict())
    reference = _derived_reference(
        base, request, pair, randomness_reference,
        event_json, context_json, provenance_json,
    )
    status = (
        TransformationStatus.APPLIED
        if final_report.build_gate_passed
        and final_report.trace_ground_truth_statuses.get(context.trace_id) == "validated_positive"
        and set(validated) == expected
        else TransformationStatus.CONSTRUCTION_ERROR
    )
    return DerivedRoleTrace(
        NONCE_DERIVED_REFERENCE_VERSION, reference,
        base.base_trace_reference, request.variant_id, status,
        event_json, context_json, provenance_json, validated,
        final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, None,
        "later attempt nonce evidence replaced coherently with earlier material",
    )
