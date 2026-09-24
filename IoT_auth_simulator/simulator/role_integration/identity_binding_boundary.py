"""4D.3.6 identity-consistency and token-binding boundary integration.

The adapters derive roles from immutable 4D.3.1 valid bases.  Selection is
evidence-first and role-independent; validation and HN qualification remain
owned by the existing independent authorities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import InjectionRecord, VARIANT_CAPABILITIES
from simulator.config.identity_binding_role_transformation import (
    IdentityBindingTransformationConfiguration,
)
from simulator.event_model import AuthEvent, EventResult, EventType
from simulator.hard_negative import QualificationStatus
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.direct_timing import (
    DerivedRoleTrace,
    _hn_candidate,
    _predicate,
    _validation,
    materialize_base,
)
from simulator.role_integration.models import (
    ValidBaseTrace,
    canonical_json,
    normalized_event_records,
    stable_reference,
)
from simulator.role_integration.pool import derive_future_transformation_reference
from simulator.role_integration.transformation import TransformationStatus


AUTHENTICATED_IDENTITY_RESOLVER_VERSION = "4d3.6-authenticated-identity-resolver-v1"
AUTHENTICATION_TARGET_POLICY_VERSION = "4d3.6-first-complete-successful-auth-v1"
TOKEN_BINDING_RESOLVER_VERSION = "4d3.6-token-binding-resolver-v1"
TOKEN_BINDING_TARGET_POLICY_VERSION = "4d3.6-first-current-token-use-v1"
PERSISTENT_IDENTITY_ANOMALY_ADAPTER_VERSION = "4d3.6-persistent-identity-anomaly-v1"
BINDING_HN_ADAPTER_VERSION = "4d3.6-binding-hn-v1"
TOKEN_BINDING_ANOMALY_ADAPTER_VERSION = "4d3.6-token-binding-anomaly-v1"
IDENTITY_BINDING_PROVENANCE_VERSION = "4d3.6-identity-binding-provenance-v1"
IDENTITY_BINDING_DERIVED_REFERENCE_VERSION = (
    "4d3.6-identity-binding-derived-reference-sha256-v1"
)
IDENTITY_BINDING_MATCHED_REFERENCE_VERSION = "4d3.6-matched-boundary-v1"

TOKEN_IDENTITY_COHERENT_HN = "token_identity_binding_coherent"
TOKEN_CONTEXT_COHERENT_HN = "token_protected_context_binding_coherent"
PERSISTENT_IDENTITY_ANOMALY = "authenticated_identity_mismatch_persistent_device"
TOKEN_IDENTITY_ANOMALY = "token_bound_identity_mismatch_authenticated_context"
SUPPORTED_IDENTITY_BINDING_VARIANTS = (
    TOKEN_IDENTITY_COHERENT_HN,
    TOKEN_CONTEXT_COHERENT_HN,
    PERSISTENT_IDENTITY_ANOMALY,
    TOKEN_IDENTITY_ANOMALY,
)

_AUTH_INTERACTION_TYPES = {
    EventType.AUTHENTICATION_REQUEST.value,
    EventType.RENEWAL_REQUEST.value,
    EventType.RETRY.value,
    EventType.CHALLENGE_SENT.value,
}
_TOKEN_USE_TYPES = {
    EventType.TOKEN_PRESENTED.value,
    EventType.TOKEN_VALIDATED.value,
    EventType.SESSION_OPENED.value,
    EventType.ACCESS_REQUEST.value,
    EventType.ACCESS_GRANTED.value,
    EventType.ACCESS_DENIED.value,
}


@dataclass(frozen=True)
class IdentityBindingRoleTransformationRequest:
    base_trace_reference: str
    variant_id: str
    transformation_seed: int
    configuration: IdentityBindingTransformationConfiguration = (
        IdentityBindingTransformationConfiguration()
    )


@dataclass(frozen=True)
class AuthenticatedIdentityEvidence:
    association_ordinal: int
    association_event_id: str
    authentication_ordinal: int
    authentication_event_id: str
    auth_attempt_id: str
    persistent_identity: str
    authenticated_identity: str


@dataclass(frozen=True)
class TokenBindingEvidence:
    issuance_ordinal: int
    issuance_event_id: str
    use_ordinal: int
    use_event_id: str
    use_event_type: str
    token_id: str
    auth_attempt_id: Optional[str]
    authenticated_identity: str
    token_bound_identity: str
    protected_session_id: Optional[str]
    token_bound_protected_session_id: Optional[str]


def resolve_authenticated_identity(
    base: ValidBaseTrace,
) -> Optional[AuthenticatedIdentityEvidence]:
    """Select the first complete successful persistent/auth identity relation."""
    context = json.loads(base.session_context_json)
    declared_attempts = set(context.get("auth_attempt_ids", ()))
    for index, target in enumerate(base.event_records):
        attempt = target.get("auth_attempt_id")
        if not (
            target.get("event_type") == EventType.AUTHENTICATION_SUCCESS.value
            and target.get("result") == EventResult.SUCCESS.value
            and target.get("authenticated_context_active") is True
            and target.get("authenticated_identity") == base.device_id
            and target.get("device_id") == base.device_id
            and attempt in declared_attempts
        ):
            continue
        association = next((
            (earlier_index, earlier)
            for earlier_index, earlier in enumerate(base.event_records[:index])
            if earlier.get("event_type") in _AUTH_INTERACTION_TYPES
            and earlier.get("auth_attempt_id") == attempt
            and earlier.get("device_id") == base.device_id
            and float(earlier["timestamp"]) < float(target["timestamp"])
        ), None)
        if association is None:
            continue
        earlier_index, earlier = association
        return AuthenticatedIdentityEvidence(
            association_ordinal=earlier_index + 1,
            association_event_id=str(earlier["event_id"]),
            authentication_ordinal=index + 1,
            authentication_event_id=str(target["event_id"]),
            auth_attempt_id=str(attempt),
            persistent_identity=base.device_id,
            authenticated_identity=str(target["authenticated_identity"]),
        )
    return None


def _issued_by_token(base: ValidBaseTrace) -> dict[str, tuple[int, Mapping[str, Any]]]:
    return {
        str(record["token_id"]): (index, record)
        for index, record in enumerate(base.event_records)
        if record.get("event_type") == EventType.TOKEN_ISSUED.value
        and record.get("result") == EventResult.SUCCESS.value
        and record.get("token_id")
    }


def _resolve_token_binding(
    base: ValidBaseTrace, *, require_protected_context: bool,
) -> Optional[TokenBindingEvidence]:
    issued = _issued_by_token(base)
    for index, use in enumerate(base.event_records):
        if use.get("event_type") not in _TOKEN_USE_TYPES or not use.get("token_id"):
            continue
        if not (
            use.get("result") == EventResult.SUCCESS.value
            or use.get("event_type") == EventType.ACCESS_DENIED.value
        ):
            continue
        if (
            use.get("token_id") != use.get("current_token_id")
            or use.get("token_lineage_status") != "current"
        ):
            continue
        origin_pair = issued.get(str(use["token_id"]))
        if origin_pair is None:
            continue
        origin_index, origin = origin_pair
        identity_complete = bool(
            origin_index < index
            and origin.get("trace_id") == use.get("trace_id")
            and origin.get("device_id") == use.get("device_id") == base.device_id
            and origin.get("token_bound_identity")
            and use.get("token_bound_identity")
            and origin.get("token_bound_identity") == use.get("token_bound_identity")
            and use.get("authenticated_context_active") is True
            and use.get("authenticated_identity")
            and use.get("token_bound_identity") == use.get("authenticated_identity")
        )
        context_complete = bool(
            use.get("protected_session_active") is True
            and use.get("protected_session_id")
            and use.get("token_bound_protected_session_id")
            and use.get("token_bound_protected_session_id")
            == use.get("protected_session_id")
            and (
                not origin.get("token_bound_protected_session_id")
                or origin.get("token_bound_protected_session_id")
                == use.get("token_bound_protected_session_id")
            )
        )
        if not identity_complete or (require_protected_context and not context_complete):
            continue
        return TokenBindingEvidence(
            issuance_ordinal=origin_index + 1,
            issuance_event_id=str(origin["event_id"]),
            use_ordinal=index + 1,
            use_event_id=str(use["event_id"]),
            use_event_type=str(use["event_type"]),
            token_id=str(use["token_id"]),
            auth_attempt_id=(
                str(use["auth_attempt_id"])
                if use.get("auth_attempt_id") is not None else None
            ),
            authenticated_identity=str(use["authenticated_identity"]),
            token_bound_identity=str(use["token_bound_identity"]),
            protected_session_id=(
                str(use["protected_session_id"])
                if use.get("protected_session_id") is not None else None
            ),
            token_bound_protected_session_id=(
                str(use["token_bound_protected_session_id"])
                if use.get("token_bound_protected_session_id") is not None else None
            ),
        )
    return None


def resolve_token_identity_binding(base: ValidBaseTrace) -> Optional[TokenBindingEvidence]:
    """Select the first complete current token/authenticated-identity relation."""
    return _resolve_token_binding(base, require_protected_context=False)


def resolve_token_protected_context_binding(
    base: ValidBaseTrace,
) -> Optional[TokenBindingEvidence]:
    """Select the first complete token/active-protected-context relation."""
    return _resolve_token_binding(base, require_protected_context=True)


def identity_matched_boundary_reference(base_reference: str) -> str:
    return stable_reference({
        "version": IDENTITY_BINDING_MATCHED_REFERENCE_VERSION,
        "base_trace_reference": base_reference,
        "variants": (
            "persistent_authenticated_identity_coherent",
            PERSISTENT_IDENTITY_ANOMALY,
        ),
        "target_policy": AUTHENTICATION_TARGET_POLICY_VERSION,
    })


def binding_matched_boundary_reference(base_reference: str) -> str:
    return stable_reference({
        "version": IDENTITY_BINDING_MATCHED_REFERENCE_VERSION,
        "base_trace_reference": base_reference,
        "variants": (TOKEN_IDENTITY_COHERENT_HN, TOKEN_IDENTITY_ANOMALY),
        "target_policy": TOKEN_BINDING_TARGET_POLICY_VERSION,
    })


def _derived_reference(
    base: ValidBaseTrace,
    request: IdentityBindingRoleTransformationRequest,
    randomness_reference: str,
    event_json: tuple[str, ...],
    provenance_json: str,
) -> str:
    provenance = json.loads(provenance_json)
    return stable_reference({
        "reference_version": IDENTITY_BINDING_DERIVED_REFERENCE_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "configuration_reference": request.configuration.configuration_reference,
        "transformation_randomness_reference": randomness_reference,
        "normalized_derived_evidence": normalized_event_records(
            tuple(json.loads(item) for item in event_json)
        ),
        "construction_mode": provenance.get("construction_mode"),
        "selected_event_references": provenance.get("selected_event_references", ()),
        "transformed_event_references": provenance.get("transformed_event_references", ()),
        "validated_invariant_ids": provenance.get("validated_invariant_ids", ()),
    })


def _empty(
    base: ValidBaseTrace,
    request: IdentityBindingRoleTransformationRequest,
    status: TransformationStatus,
    reason: str,
    randomness_reference: str,
) -> DerivedRoleTrace:
    provenance_json = canonical_json({
        "provenance_version": IDENTITY_BINDING_PROVENANCE_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "status": status.value,
        "configuration_reference": request.configuration.configuration_reference,
        "transformation_randomness_reference": randomness_reference,
        "applied": False,
        "reason": reason,
    })
    reference = _derived_reference(
        base, request, randomness_reference, base.event_records_json, provenance_json,
    )
    return DerivedRoleTrace(
        IDENTITY_BINDING_DERIVED_REFERENCE_VERSION, reference,
        base.base_trace_reference, request.variant_id, status,
        base.event_records_json, base.session_context_json, provenance_json,
        (), None, False, None, reason,
    )


def _qualify_reference_hn(
    base: ValidBaseTrace,
    request: IdentityBindingRoleTransformationRequest,
    randomness_reference: str,
    provenance: dict[str, Any],
    evidence: TokenBindingEvidence,
) -> DerivedRoleTrace:
    events, context = materialize_base(base)
    report = _validation(events, context)
    provisional_json = canonical_json(provenance)
    provisional_reference = _derived_reference(
        base, request, randomness_reference, base.event_records_json, provisional_json,
    )
    candidate = _hn_candidate(
        context=context,
        variant_id=request.variant_id,
        derived_reference=provisional_reference,
        target_ids=(evidence.use_event_id,),
        evidence_ids=(evidence.issuance_event_id, evidence.use_event_id),
        boundary_parameters={
            "boundary_targeted": True,
            "selection_policy": TOKEN_BINDING_TARGET_POLICY_VERSION,
        },
        construction_parameters={
            "adapter_version": BINDING_HN_ADAPTER_VERSION,
            "base_trace_reference": base.base_trace_reference,
            "construction_mode": "reference_only",
            "transformation_randomness_reference": randomness_reference,
        },
        counterfactual_reference=(
            binding_matched_boundary_reference(base.base_trace_reference)
            if request.variant_id == TOKEN_IDENTITY_COHERENT_HN else None
        ),
    )
    qualified = qualify_hard_negative(events, context, candidate, report)
    provenance.update({
        "independent_validation_status": report.trace_ground_truth_statuses.get(
            context.trace_id
        ),
        "independent_validated_invariant_ids": list(
            report.trace_validated_invariant_ids.get(context.trace_id, ())
        ),
        "qualification_status": qualified.qualification_status.value,
        "qualification_reason": qualified.qualification_reason,
    })
    provenance_json = canonical_json(provenance)
    reference = _derived_reference(
        base, request, randomness_reference, base.event_records_json, provenance_json,
    )
    qualified = replace(qualified, hard_negative_candidate_id=reference)
    final_report = _validation(events, context, qualification_records=[qualified])
    status = (
        TransformationStatus.APPLIED
        if qualified.qualification_status is QualificationStatus.QUALIFIED
        and final_report.build_gate_passed
        and final_report.trace_ground_truth_statuses.get(context.trace_id)
        == "validated_normal"
        and not final_report.trace_validated_invariant_ids.get(context.trace_id, ())
        else TransformationStatus.CONSTRUCTION_ERROR
    )
    return DerivedRoleTrace(
        IDENTITY_BINDING_DERIVED_REFERENCE_VERSION, reference,
        base.base_trace_reference, request.variant_id, status,
        base.event_records_json, base.session_context_json, provenance_json,
        tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ())),
        final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, qualified,
        "represented binding evidence referenced without mutation",
    )


def _deterministic_distinct_identity(
    *, base: ValidBaseTrace, variant_id: str, original: str, target_event_id: str,
) -> str:
    digest = stable_reference({
        "version": IDENTITY_BINDING_DERIVED_REFERENCE_VERSION,
        "base": base.base_trace_reference,
        "variant": variant_id,
        "target": target_event_id,
    })[7:31]
    prefix = "derived-authenticated" if variant_id == PERSISTENT_IDENTITY_ANOMALY else "derived-token-owner"
    value = f"{prefix}-{digest}"
    if value == original:  # defensive; current namespace makes this unreachable
        value += "-distinct"
    return value


def _finish_anomaly(
    base: ValidBaseTrace,
    request: IdentityBindingRoleTransformationRequest,
    randomness_reference: str,
    provenance: dict[str, Any],
    events: list[AuthEvent],
    context: Any,
    expected: set[str],
    reason: str,
) -> DerivedRoleTrace:
    evidence_report = _validation(events, context)
    confirmed = set(
        evidence_report.trace_validated_invariant_ids.get(context.trace_id, ())
    )
    if not expected.issubset(confirmed):
        return _empty(
            base, request, TransformationStatus.CONSTRUCTION_ERROR,
            "independent validation did not confirm the intended invariant",
            randomness_reference,
        )
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
    validated = tuple(
        final_report.trace_validated_invariant_ids.get(context.trace_id, ())
    )
    # The frozen validator deliberately treats an access performed under an
    # authenticated identity different from the persistent device as an
    # additional access-precondition violation.  It is not part of this
    # variant's mapping, but it is genuine independently reconstructed evidence
    # and must be reported rather than hidden or used to reselect a base.
    atomic_collision_ids = sorted({
        record.invariant_identifier
        for record in final_report.records
        if record.trace_id == context.trace_id
        and record.invariant_identifier
        and record.result.value == "PASS"
        and record.validation_id.startswith("c2.")
        and record.invariant_identifier not in expected
    })
    complete_validated = tuple(sorted(set(validated) | set(atomic_collision_ids)))
    provenance.update({
        "validated_invariant_ids": list(complete_validated),
        "intended_validated_invariant_ids": sorted(expected & set(validated)),
        "additional_invariant_collisions": atomic_collision_ids,
        "aggregate_build_gate_passed": final_report.build_gate_passed,
        "ground_truth_consolidation": (
            "independent_evidence_pass_then_final_validation"
        ),
    })
    provenance_json = canonical_json(provenance)
    event_json = tuple(canonical_json(event.to_dict()) for event in events)
    context_json = canonical_json(context.to_dict())
    reference = _derived_reference(
        base, request, randomness_reference, event_json, provenance_json,
    )
    status = (
        TransformationStatus.APPLIED
        if not final_report.gt_consistency_failures
        and expected.issubset(set(validated))
        else TransformationStatus.CONSTRUCTION_ERROR
    )
    return DerivedRoleTrace(
        IDENTITY_BINDING_DERIVED_REFERENCE_VERSION, reference,
        base.base_trace_reference, request.variant_id, status,
        event_json, context_json, provenance_json, complete_validated,
        final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, None, reason,
    )


def integrate_identity_binding_role(
    base: ValidBaseTrace, request: IdentityBindingRoleTransformationRequest,
) -> DerivedRoleTrace:
    """Derive one supported identity/binding role from an immutable base."""
    if request.base_trace_reference != base.base_trace_reference:
        raise ValueError("request/base reference mismatch")
    randomness_reference = derive_future_transformation_reference(
        transformation_seed=request.transformation_seed,
        base_trace_reference=base.base_trace_reference,
        transformation_key=(
            f"{request.variant_id}|{request.configuration.configuration_reference}"
        ),
    )
    if request.variant_id not in SUPPORTED_IDENTITY_BINDING_VARIANTS:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "variant is outside the frozen 4D.3.6 integration scope",
            randomness_reference,
        )

    identity = resolve_authenticated_identity(base)
    identity_binding = resolve_token_identity_binding(base)
    context_binding = resolve_token_protected_context_binding(base)
    broad_predicate = {
        PERSISTENT_IDENTITY_ANOMALY: ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION,
        TOKEN_IDENTITY_COHERENT_HN: ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING,
        TOKEN_CONTEXT_COHERENT_HN: ApplicabilityPredicate.TOKEN_PROTECTED_CONTEXT_BINDING,
        TOKEN_IDENTITY_ANOMALY: ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING,
    }[request.variant_id]
    broad = _predicate(base, broad_predicate)
    if not broad.applicable:
        return _empty(
            base, request, TransformationStatus.NOT_APPLICABLE, broad.reason,
            randomness_reference,
        )
    precise: Optional[object] = (
        identity if request.variant_id == PERSISTENT_IDENTITY_ANOMALY
        else context_binding if request.variant_id == TOKEN_CONTEXT_COHERENT_HN
        else identity_binding
    )
    if precise is None:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "broad applicability lacks the complete canonical represented relation",
            randomness_reference,
        )

    selected_refs: dict[str, str]
    if isinstance(precise, AuthenticatedIdentityEvidence):
        selected_refs = {
            "association": f"event:{precise.association_ordinal}",
            "authentication_success": f"event:{precise.authentication_ordinal}",
        }
    else:
        assert isinstance(precise, TokenBindingEvidence)
        selected_refs = {
            "token_issuance": f"event:{precise.issuance_ordinal}",
            "token_use": f"event:{precise.use_ordinal}",
        }
    provenance: dict[str, Any] = {
        "provenance_version": IDENTITY_BINDING_PROVENANCE_VERSION,
        "authenticated_identity_resolver_version": AUTHENTICATED_IDENTITY_RESOLVER_VERSION,
        "authentication_target_policy_version": AUTHENTICATION_TARGET_POLICY_VERSION,
        "token_binding_resolver_version": TOKEN_BINDING_RESOLVER_VERSION,
        "token_binding_target_policy_version": TOKEN_BINDING_TARGET_POLICY_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "broad_applicability": broad.to_record(),
        "selected_event_references": selected_refs,
        "configuration_reference": request.configuration.configuration_reference,
        "transformation_randomness_reference": randomness_reference,
        "event_count_changed": False,
        "event_order_changed": False,
        "timestamps_rewritten": False,
        "ordinary_base_timing_provenance_preserved": True,
        "base_timing_provenance_reference": stable_reference(base.timing_provenance),
        "applied": True,
    }

    if request.variant_id in {TOKEN_IDENTITY_COHERENT_HN, TOKEN_CONTEXT_COHERENT_HN}:
        evidence = (
            identity_binding
            if request.variant_id == TOKEN_IDENTITY_COHERENT_HN else context_binding
        )
        assert evidence is not None
        provenance.update({
            "adapter_version": BINDING_HN_ADAPTER_VERSION,
            "construction_mode": "reference_only",
            "transformed_event_references": [],
            "selected_token_use_type": evidence.use_event_type,
            "selected_token_reference": stable_reference({
                "base": base.base_trace_reference, "token_id": evidence.token_id,
            }),
            "matched_boundary_reference": (
                binding_matched_boundary_reference(base.base_trace_reference)
                if request.variant_id == TOKEN_IDENTITY_COHERENT_HN else None
            ),
        })
        return _qualify_reference_hn(
            base, request, randomness_reference, provenance, evidence,
        )

    events, context = materialize_base(base)
    capability = VARIANT_CAPABILITIES[request.variant_id]
    if request.variant_id == PERSISTENT_IDENTITY_ANOMALY:
        assert identity is not None
        original = identity.authenticated_identity
        injected = _deterministic_distinct_identity(
            base=base, variant_id=request.variant_id, original=original,
            target_event_id=identity.authentication_event_id,
        )
        transformed: list[str] = []
        for ordinal, event in enumerate(events, 1):
            if ordinal < identity.authentication_ordinal:
                continue
            changed = False
            if event.authenticated_identity == original:
                event.authenticated_identity = injected
                changed = True
            if event.token_bound_identity == original:
                event.token_bound_identity = injected
                changed = True
            if changed:
                transformed.append(f"event:{ordinal}")
        target = events[identity.authentication_ordinal - 1]
        context = replace(context, authenticated_identity=injected)
        matched = identity_matched_boundary_reference(base.base_trace_reference)
        adapter_version = PERSISTENT_IDENTITY_ANOMALY_ADAPTER_VERSION
        construction_mode = "coherent_authenticated_context_identity_substitution"
        source_ids = (identity.association_event_id,)
        targeted_context = "first complete successful authentication context"
        source_material = identity.auth_attempt_id
        protected_binding_preserved = all(
            event.token_bound_protected_session_id == original_event.get(
                "token_bound_protected_session_id"
            )
            for event, original_event in zip(events, base.event_records)
        )
    else:
        assert identity_binding is not None
        original = identity_binding.token_bound_identity
        injected = _deterministic_distinct_identity(
            base=base, variant_id=request.variant_id, original=original,
            target_event_id=identity_binding.use_event_id,
        )
        transformed = []
        for ordinal, event in enumerate(events, 1):
            if event.token_id == identity_binding.token_id and event.token_bound_identity == original:
                event.token_bound_identity = injected
                transformed.append(f"event:{ordinal}")
        target = events[identity_binding.use_ordinal - 1]
        matched = binding_matched_boundary_reference(base.base_trace_reference)
        adapter_version = TOKEN_BINDING_ANOMALY_ADAPTER_VERSION
        construction_mode = "token_lineage_bound_identity_substitution"
        source_ids = (identity_binding.issuance_event_id,)
        targeted_context = "first complete current-token authenticated-identity use"
        source_material = identity_binding.token_id
        protected_binding_preserved = all(
            event.token_bound_protected_session_id == original_event.get(
                "token_bound_protected_session_id"
            )
            for event, original_event in zip(events, base.event_records)
        )

    injection_id = "4d3.6-injection-" + stable_reference({
        "base": base.base_trace_reference,
        "variant": request.variant_id,
        "target": target.event_id,
        "randomness": randomness_reference,
    })[7:31]
    target.injection_id = injection_id
    target.anomaly_variant_name = request.variant_id
    target.invariant_families = capability.invariant_families
    target.source_context = "controlled_injection"
    injection = InjectionRecord(
        injection_id=injection_id,
        historical_scenario_name=capability.historical_name,
        anomaly_variant_name=request.variant_id,
        capability_status=capability.status.value,
        mapped_semantic_concept=capability.semantic_concept,
        invariant_families=capability.invariant_families,
        canonical_invariant_ids=capability.invariant_ids,
        targeted_event_type=target.event_type.value,
        targeted_action_or_context=targeted_context,
        source_evidence_event_ids=source_ids,
        source_material_identifier=source_material,
        source_trace_id=context.trace_id,
        source_session_id=context.legacy_session_id,
        source_auth_attempt_id=target.auth_attempt_id,
        original_value=original,
        injected_value=injected,
        injection_parameters={
            "adapter_version": adapter_version,
            "authenticated_identity_resolver_version": AUTHENTICATED_IDENTITY_RESOLVER_VERSION,
            "authentication_target_policy_version": AUTHENTICATION_TARGET_POLICY_VERSION,
            "token_binding_resolver_version": TOKEN_BINDING_RESOLVER_VERSION,
            "token_binding_target_policy_version": TOKEN_BINDING_TARGET_POLICY_VERSION,
            "transformed_event_references": transformed,
            "persistent_context_mutated": False,
            "protected_context_binding_preserved": protected_binding_preserved,
            "configuration_reference": request.configuration.configuration_reference,
            "transformation_randomness_reference": randomness_reference,
        },
        trace_id=context.trace_id,
        device_id=context.device_id,
        session_id=context.legacy_session_id,
        auth_attempt_id=target.auth_attempt_id,
        declared_history_scope=capability.history_scope,
        intended_observable_violation=capability.semantic_concept,
        transformation_applied=True,
    )
    context = replace(
        context, attack_type=request.variant_id,
        attack_phase=target.event_type.value, severity="high",
        attack_scenario_intent=True, observable_anomaly=False,
        observable_violation_candidate=False,
        injection_records=(injection.to_dict(),),
    )
    provenance.update({
        "adapter_version": adapter_version,
        "construction_mode": construction_mode,
        "transformed_event_references": transformed,
        "transformed_identity_fields": (
            ["authenticated_identity", "token_bound_identity", "SessionContext.authenticated_identity"]
            if request.variant_id == PERSISTENT_IDENTITY_ANOMALY
            else ["token_bound_identity"]
        ),
        "persistent_context_mutated": False,
        "authenticated_identity_preserved": request.variant_id == TOKEN_IDENTITY_ANOMALY,
        "token_lineage_identity_propagated": True,
        "protected_context_binding_preserved": protected_binding_preserved,
        "matched_boundary_reference": matched,
    })
    return _finish_anomaly(
        base, request, randomness_reference, provenance, events, context,
        set(capability.invariant_ids),
        (
            "authenticated context identity propagated while persistent identity remained fixed"
            if request.variant_id == PERSISTENT_IDENTITY_ANOMALY
            else "selected token lineage bound identity changed while authenticated context remained fixed"
        ),
    )
