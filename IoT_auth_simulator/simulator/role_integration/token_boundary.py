"""4D.3.5 token-currentness boundaries over immutable valid bases."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import InjectionRecord, VARIANT_CAPABILITIES
from simulator.config.token_role_transformation import TokenRoleTransformationConfiguration
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


TOKEN_EVIDENCE_RESOLVER_VERSION = "4d3.5-token-evidence-resolver-v1"
REPEATED_CURRENT_SELECTION_POLICY_VERSION = "4d3.5-first-current-token-multi-use-v1"
TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION = "4d3.5-first-complete-replacement-lineage-v1"
PRE_REPLACEMENT_T1_SELECTION_POLICY_VERSION = "4d3.5-last-valid-t1-use-before-replacement-v1"
POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION = "4d3.5-first-current-t2-presentation-after-replacement-v1"
TOKEN_BOUNDARY_ADAPTER_VERSION = "4d3.5-token-currentness-adapter-v1"
TOKEN_TRANSFORMATION_PROVENANCE_VERSION = "4d3.5-token-provenance-v1"
TOKEN_DERIVED_REFERENCE_VERSION = "4d3.5-token-derived-reference-sha256-v1"
TOKEN_MATCHED_BOUNDARY_REFERENCE_VERSION = "4d3.5-matched-token-boundary-v1"
TOKEN_OPPORTUNITY_CLASSIFIER_VERSION = "4d3.5-token-opportunity-classifier-v1"

REPEATED_CURRENT_TOKEN_HN = "repeated_current_token_use"
LAST_VALID_T1_HN = "last_valid_use_before_replacement"
CORRECT_REPLACEMENT_T2_HN = "correct_replacement_token_use"
SUPERSEDED_TOKEN_REPLAY_ANOMALY = "token_replay_after_replacement"
SUPPORTED_TOKEN_VARIANTS = (
    REPEATED_CURRENT_TOKEN_HN,
    LAST_VALID_T1_HN,
    CORRECT_REPLACEMENT_T2_HN,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY,
)


@dataclass(frozen=True)
class TokenRoleTransformationRequest:
    base_trace_reference: str
    variant_id: str
    transformation_seed: int
    configuration: TokenRoleTransformationConfiguration = (
        TokenRoleTransformationConfiguration()
    )


@dataclass(frozen=True)
class TokenUseEvidence:
    ordinal: int
    event_id: str
    event_type: str
    token_id: str
    current_token_id: str
    token_generation: int
    protected_session_id: str
    auth_attempt_id: Optional[str]
    timestamp: float
    observed_timestamp: float
    token_expiry: float
    token_lineage_status: str

    @property
    def expired_at_use(self) -> bool:
        return self.timestamp > self.token_expiry


@dataclass(frozen=True)
class RepeatedCurrentTokenEvidence:
    token_id: str
    protected_session_id: str
    uses: tuple[TokenUseEvidence, ...]


@dataclass(frozen=True)
class TokenReplacementEvidence:
    source_issuance_ordinal: int
    source_issuance_event_id: str
    session_open_ordinal: int
    session_open_event_id: str
    renewal_success_ordinal: int
    renewal_success_event_id: str
    replacement_ordinal: int
    replacement_event_id: str
    t1_token_id: str
    t2_token_id: str
    t1_generation: int
    t2_generation: int
    protected_session_id: str
    replacement_auth_attempt_id: str
    replacement_timestamp: float
    t1_expiry: float
    t1_scope: Optional[str]
    t1_bound_identity: Optional[str]
    t1_bound_protected_session_id: Optional[str]
    pre_replacement_use: Optional[TokenUseEvidence]
    post_replacement_use: Optional[TokenUseEvidence]
    post_replacement_validation: Optional[TokenUseEvidence]
    semantic_order_valid: bool
    observed_order_valid: bool


def _observed(record: Mapping[str, Any]) -> float:
    value = record.get("observed_timestamp")
    return float(record["timestamp"] if value is None else value)


def _use(record: Mapping[str, Any], ordinal: int) -> Optional[TokenUseEvidence]:
    required = (
        "event_id", "event_type", "token_id", "current_token_id",
        "token_generation", "protected_session_id", "timestamp", "token_expiry",
        "token_lineage_status",
    )
    if any(record.get(name) is None for name in required):
        return None
    return TokenUseEvidence(
        ordinal=ordinal,
        event_id=str(record["event_id"]),
        event_type=str(record["event_type"]),
        token_id=str(record["token_id"]),
        current_token_id=str(record["current_token_id"]),
        token_generation=int(record["token_generation"]),
        protected_session_id=str(record["protected_session_id"]),
        auth_attempt_id=(
            str(record["auth_attempt_id"])
            if record.get("auth_attempt_id") is not None else None
        ),
        timestamp=float(record["timestamp"]),
        observed_timestamp=_observed(record),
        token_expiry=float(record["token_expiry"]),
        token_lineage_status=str(record["token_lineage_status"]),
    )


def resolve_repeated_current_token_use(
    base: ValidBaseTrace,
) -> Optional[RepeatedCurrentTokenEvidence]:
    """Select the first token/session with two legitimate protected uses."""
    grouped: dict[tuple[str, str], list[TokenUseEvidence]] = {}
    for ordinal, record in enumerate(base.event_records, 1):
        if (
            record.get("event_type") != EventType.ACCESS_REQUEST.value
            or record.get("result") != EventResult.SUCCESS.value
            or record.get("token_id") != record.get("current_token_id")
            or record.get("token_lineage_status") != "current"
        ):
            continue
        evidence = _use(record, ordinal)
        if evidence is None or evidence.expired_at_use:
            continue
        grouped.setdefault(
            (evidence.token_id, evidence.protected_session_id), []
        ).append(evidence)
    candidates = [
        RepeatedCurrentTokenEvidence(token, session, tuple(uses))
        for (token, session), uses in grouped.items() if len(uses) >= 2
    ]
    return min(candidates, key=lambda item: item.uses[0].ordinal) if candidates else None


def resolve_token_replacements(base: ValidBaseTrace) -> tuple[TokenReplacementEvidence, ...]:
    """Reconstruct explicit T1 -> T2 replacement in one continuing session."""
    events = base.event_records
    context = json.loads(base.session_context_json)
    renewal_attempts = set(context.get("renewal_auth_attempt_ids", ()))
    continuing_session = context.get("protected_session_id")
    replacements: list[TokenReplacementEvidence] = []
    for replacement_index, replacement in enumerate(events):
        if (
            replacement.get("event_type") != EventType.TOKEN_ISSUED.value
            or replacement.get("result") != EventResult.SUCCESS.value
            or not replacement.get("token_id")
            or not replacement.get("supersedes_token_id")
            or not replacement.get("auth_attempt_id")
            or not replacement.get("protected_session_id")
        ):
            continue
        t1 = str(replacement["supersedes_token_id"])
        t2 = str(replacement["token_id"])
        attempt = str(replacement["auth_attempt_id"])
        session = str(replacement["protected_session_id"])
        if t1 == t2 or attempt not in renewal_attempts or session != continuing_session:
            continue
        source_matches = [
            (index, event) for index, event in enumerate(events[:replacement_index])
            if event.get("event_type") == EventType.TOKEN_ISSUED.value
            and event.get("result") == EventResult.SUCCESS.value
            and event.get("token_id") == t1
            and event.get("current_token_id") == t1
            and event.get("token_lineage_status") == "current"
        ]
        session_matches = [
            (index, event) for index, event in enumerate(events[:replacement_index])
            if event.get("event_type") == EventType.SESSION_OPENED.value
            and event.get("result") == EventResult.SUCCESS.value
            and event.get("token_id") == t1
            and event.get("protected_session_id") == session
        ]
        renewal_matches = [
            (index, event) for index, event in enumerate(events[:replacement_index])
            if event.get("event_type") == EventType.AUTHENTICATION_SUCCESS.value
            and event.get("result") == EventResult.SUCCESS.value
            and event.get("auth_attempt_id") == attempt
        ]
        if not source_matches or not session_matches or not renewal_matches:
            continue
        source_index, source = source_matches[-1]
        session_index, session_open = session_matches[-1]
        renewal_index, renewal_success = renewal_matches[-1]
        if (
            replacement.get("current_token_id") != t2
            or replacement.get("token_lineage_status") != "current"
            or replacement.get("token_generation") is None
            or source.get("token_generation") is None
            or int(replacement["token_generation"]) != int(source["token_generation"]) + 1
        ):
            continue

        pre_candidates: list[TokenUseEvidence] = []
        for index, event in enumerate(events[:replacement_index]):
            if (
                event.get("event_type") in {
                    EventType.ACCESS_REQUEST.value, EventType.TOKEN_PRESENTED.value,
                }
                and event.get("result") == EventResult.SUCCESS.value
                and event.get("token_id") == t1
                and event.get("current_token_id") == t1
                and event.get("token_lineage_status") == "current"
                and event.get("protected_session_id") == session
            ):
                evidence = _use(event, index + 1)
                if evidence is not None and not evidence.expired_at_use:
                    pre_candidates.append(evidence)
        protected_pre = [
            item for item in pre_candidates
            if item.event_type == EventType.ACCESS_REQUEST.value
        ]
        pre_use = (protected_pre or pre_candidates)[-1] if pre_candidates else None

        post_candidates: list[TokenUseEvidence] = []
        for index, event in enumerate(events[replacement_index + 1:], replacement_index + 1):
            if (
                event.get("event_type") == EventType.TOKEN_PRESENTED.value
                and event.get("result") == EventResult.SUCCESS.value
                and event.get("token_id") == t2
                and event.get("current_token_id") == t2
                and event.get("token_lineage_status") == "current"
                and event.get("protected_session_id") == session
            ):
                evidence = _use(event, index + 1)
                if evidence is not None:
                    post_candidates.append(evidence)
        post_use = post_candidates[0] if post_candidates else None
        validation_candidates: list[TokenUseEvidence] = []
        if post_use is not None:
            for index, event in enumerate(
                events[post_use.ordinal:], post_use.ordinal
            ):
                if (
                    event.get("event_type") == EventType.TOKEN_VALIDATED.value
                    and event.get("result") == EventResult.SUCCESS.value
                    and event.get("token_id") == t2
                    and event.get("current_token_id") == t2
                    and event.get("token_lineage_status") == "current"
                    and event.get("protected_session_id") == session
                    and event.get("auth_attempt_id") == post_use.auth_attempt_id
                ):
                    evidence = _use(event, index + 1)
                    if evidence is not None:
                        validation_candidates.append(evidence)
        post_validation = validation_candidates[0] if validation_candidates else None
        ordered_records = [source, session_open, renewal_success, replacement]
        if pre_use:
            ordered_records.append(events[pre_use.ordinal - 1])
        if post_use:
            ordered_records.append(events[post_use.ordinal - 1])
        semantic_order = bool(
            float(source["timestamp"]) < float(session_open["timestamp"])
            and float(renewal_success["timestamp"]) < float(replacement["timestamp"])
            and (pre_use is None or pre_use.timestamp < float(replacement["timestamp"]))
            and (post_use is None or float(replacement["timestamp"]) < post_use.timestamp)
        )
        observed_order = bool(
            _observed(source) < _observed(session_open)
            and _observed(renewal_success) < _observed(replacement)
            and (pre_use is None or pre_use.observed_timestamp < _observed(replacement))
            and (post_use is None or _observed(replacement) < post_use.observed_timestamp)
        )
        replacements.append(TokenReplacementEvidence(
            source_issuance_ordinal=source_index + 1,
            source_issuance_event_id=str(source["event_id"]),
            session_open_ordinal=session_index + 1,
            session_open_event_id=str(session_open["event_id"]),
            renewal_success_ordinal=renewal_index + 1,
            renewal_success_event_id=str(renewal_success["event_id"]),
            replacement_ordinal=replacement_index + 1,
            replacement_event_id=str(replacement["event_id"]),
            t1_token_id=t1,
            t2_token_id=t2,
            t1_generation=int(source["token_generation"]),
            t2_generation=int(replacement["token_generation"]),
            protected_session_id=session,
            replacement_auth_attempt_id=attempt,
            replacement_timestamp=float(replacement["timestamp"]),
            t1_expiry=float(source["token_expiry"]),
            t1_scope=(str(source["token_scope"]) if source.get("token_scope") else None),
            t1_bound_identity=(
                str(source["token_bound_identity"])
                if source.get("token_bound_identity") else None
            ),
            t1_bound_protected_session_id=(
                str(source["token_bound_protected_session_id"])
                if source.get("token_bound_protected_session_id") else None
            ),
            pre_replacement_use=pre_use,
            post_replacement_use=post_use,
            post_replacement_validation=post_validation,
            semantic_order_valid=semantic_order,
            observed_order_valid=observed_order,
        ))
    return tuple(sorted(replacements, key=lambda item: item.replacement_ordinal))


def select_token_replacement(base: ValidBaseTrace) -> Optional[TokenReplacementEvidence]:
    """Select the first complete represented replacement lineage in event order."""
    return next(
        (
            item for item in resolve_token_replacements(base)
            if item.semantic_order_valid and item.observed_order_valid
        ),
        None,
    )


def classify_token_opportunity(base: ValidBaseTrace) -> str:
    repeated = resolve_repeated_current_token_use(base) is not None
    replacement = select_token_replacement(base) is not None
    if repeated and replacement:
        return "repeated_and_replacement"
    if repeated:
        return "repeated_only"
    if replacement:
        return "replacement_only"
    return "neither"


def token_matched_boundary_reference(base_reference: str) -> str:
    return stable_reference({
        "version": TOKEN_MATCHED_BOUNDARY_REFERENCE_VERSION,
        "base_trace_reference": base_reference,
        "variants": (CORRECT_REPLACEMENT_T2_HN, SUPERSEDED_TOKEN_REPLAY_ANOMALY),
        "replacement_policy": TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION,
        "post_use_policy": POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION,
    })


def _derived_reference(
    base: ValidBaseTrace,
    request: TokenRoleTransformationRequest,
    randomness_reference: str,
    event_json: tuple[str, ...],
    provenance_json: str,
) -> str:
    provenance = json.loads(provenance_json)
    return stable_reference({
        "reference_version": TOKEN_DERIVED_REFERENCE_VERSION,
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
    request: TokenRoleTransformationRequest,
    status: TransformationStatus,
    reason: str,
    randomness_reference: str,
) -> DerivedRoleTrace:
    provenance_json = canonical_json({
        "provenance_version": TOKEN_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": TOKEN_BOUNDARY_ADAPTER_VERSION,
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
        TOKEN_DERIVED_REFERENCE_VERSION, reference, base.base_trace_reference,
        request.variant_id, status, base.event_records_json,
        base.session_context_json, provenance_json, (), None, False, None, reason,
    )


def _qualify_reference_hn(
    base: ValidBaseTrace,
    request: TokenRoleTransformationRequest,
    randomness_reference: str,
    provenance: dict[str, Any],
    *,
    target_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    boundary_parameters: Mapping[str, Any],
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
        target_ids=target_ids,
        evidence_ids=evidence_ids,
        boundary_parameters=boundary_parameters,
        construction_parameters={
            "adapter_version": TOKEN_BOUNDARY_ADAPTER_VERSION,
            "base_trace_reference": base.base_trace_reference,
            "transformation_randomness_reference": randomness_reference,
            "construction_mode": "reference_only",
        },
        counterfactual_reference=(
            token_matched_boundary_reference(base.base_trace_reference)
            if request.variant_id == CORRECT_REPLACEMENT_T2_HN else None
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
        TOKEN_DERIVED_REFERENCE_VERSION, reference, base.base_trace_reference,
        request.variant_id, status, base.event_records_json,
        base.session_context_json, provenance_json,
        tuple(final_report.trace_validated_invariant_ids.get(context.trace_id, ())),
        final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, qualified,
        "represented token-currentness evidence referenced without mutation",
    )


def integrate_token_role(
    base: ValidBaseTrace, request: TokenRoleTransformationRequest,
) -> DerivedRoleTrace:
    """Derive one frozen token role without changing base trace structure."""
    if request.base_trace_reference != base.base_trace_reference:
        raise ValueError("request/base reference mismatch")
    randomness_reference = derive_future_transformation_reference(
        transformation_seed=request.transformation_seed,
        base_trace_reference=base.base_trace_reference,
        transformation_key=(
            f"{request.variant_id}|{request.configuration.configuration_reference}"
        ),
    )
    if request.variant_id not in SUPPORTED_TOKEN_VARIANTS:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "variant is outside the frozen 4D.3.5 integration scope",
            randomness_reference,
        )

    repeated = resolve_repeated_current_token_use(base)
    replacement = select_token_replacement(base)
    predicate_by_variant = {
        REPEATED_CURRENT_TOKEN_HN: ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES,
        LAST_VALID_T1_HN: ApplicabilityPredicate.CURRENT_T1_USE_BEFORE_REPLACEMENT,
        CORRECT_REPLACEMENT_T2_HN: ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT,
        SUPERSEDED_TOKEN_REPLAY_ANOMALY: ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT,
    }
    broad = _predicate(base, predicate_by_variant[request.variant_id])
    if not broad.applicable:
        return _empty(
            base, request, TransformationStatus.NOT_APPLICABLE, broad.reason,
            randomness_reference,
        )
    if request.variant_id == REPEATED_CURRENT_TOKEN_HN and repeated is None:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "broad repeated-use evidence lacks two valid protected current-token uses",
            randomness_reference,
        )
    if request.variant_id != REPEATED_CURRENT_TOKEN_HN and replacement is None:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "broad replacement evidence lacks complete continuing-session lineage",
            randomness_reference,
        )
    if request.variant_id == LAST_VALID_T1_HN and replacement.pre_replacement_use is None:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "replacement lineage lacks a valid represented pre-replacement T1 use",
            randomness_reference,
        )
    if request.variant_id in {
        CORRECT_REPLACEMENT_T2_HN, SUPERSEDED_TOKEN_REPLAY_ANOMALY,
    } and replacement.post_replacement_use is None:
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "replacement lineage lacks a represented post-replacement T2 presentation",
            randomness_reference,
        )
    if (
        request.variant_id == SUPERSEDED_TOKEN_REPLAY_ANOMALY
        and replacement.post_replacement_validation is None
    ):
        return _empty(
            base, request, TransformationStatus.UNSUPPORTED,
            "post-replacement presentation lacks its represented validation outcome",
            randomness_reference,
        )

    selected_refs: dict[str, str] = {}
    if repeated is not None:
        selected_refs["first_repeated_use"] = f"event:{repeated.uses[0].ordinal}"
        selected_refs["second_repeated_use"] = f"event:{repeated.uses[1].ordinal}"
    if replacement is not None:
        selected_refs.update({
            "source_t1_issuance": f"event:{replacement.source_issuance_ordinal}",
            "continuing_session_open": f"event:{replacement.session_open_ordinal}",
            "renewal_authentication_success": f"event:{replacement.renewal_success_ordinal}",
            "replacement_t2_issuance": f"event:{replacement.replacement_ordinal}",
        })
        if replacement.pre_replacement_use:
            selected_refs["pre_replacement_t1_use"] = (
                f"event:{replacement.pre_replacement_use.ordinal}"
            )
        if replacement.post_replacement_use:
            selected_refs["post_replacement_t2_use"] = (
                f"event:{replacement.post_replacement_use.ordinal}"
            )
        if replacement.post_replacement_validation:
            selected_refs["post_replacement_t2_validation"] = (
                f"event:{replacement.post_replacement_validation.ordinal}"
            )
    provenance: dict[str, Any] = {
        "provenance_version": TOKEN_TRANSFORMATION_PROVENANCE_VERSION,
        "adapter_version": TOKEN_BOUNDARY_ADAPTER_VERSION,
        "token_evidence_resolver_version": TOKEN_EVIDENCE_RESOLVER_VERSION,
        "repeated_current_selection_policy_version": (
            REPEATED_CURRENT_SELECTION_POLICY_VERSION
        ),
        "replacement_lineage_resolver_version": (
            TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION
        ),
        "pre_replacement_selection_policy_version": (
            PRE_REPLACEMENT_T1_SELECTION_POLICY_VERSION
        ),
        "post_replacement_selection_policy_version": (
            POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION
        ),
        "opportunity_classifier_version": TOKEN_OPPORTUNITY_CLASSIFIER_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "variant_id": request.variant_id,
        "broad_applicability": broad.to_record(),
        "opportunity_class": classify_token_opportunity(base),
        "selected_event_references": selected_refs,
        "transformation_randomness_reference": randomness_reference,
        "configuration_reference": request.configuration.configuration_reference,
        "event_count_changed": False,
        "event_order_changed": False,
        "timestamps_rewritten": False,
        "ordinary_base_timing_provenance_preserved": True,
        "base_timing_provenance_reference": stable_reference(base.timing_provenance),
        "applied": True,
    }

    if request.variant_id == REPEATED_CURRENT_TOKEN_HN:
        assert repeated is not None
        uses = repeated.uses
        provenance.update({
            "construction_mode": "reference_only",
            "transformed_event_references": [],
            "represented_use_count": len(uses),
            "protected_use_event_type": EventType.ACCESS_REQUEST.value,
        })
        return _qualify_reference_hn(
            base, request, randomness_reference, provenance,
            target_ids=(uses[1].event_id,),
            evidence_ids=tuple(item.event_id for item in uses),
            boundary_parameters={"represented_protected_use_count": len(uses)},
        )

    assert replacement is not None
    replacement_common = {
        "t1_token_id": replacement.t1_token_id,
        "t2_token_id": replacement.t2_token_id,
        "protected_session_id": replacement.protected_session_id,
        "replacement_timestamp": replacement.replacement_timestamp,
        "same_continuing_protected_session": True,
        "replacement_established_by_renewal": True,
        "matched_boundary_reference": token_matched_boundary_reference(
            base.base_trace_reference
        ),
    }
    provenance.update(replacement_common)
    if request.variant_id == LAST_VALID_T1_HN:
        use = replacement.pre_replacement_use
        assert use is not None
        provenance.update({
            "construction_mode": "reference_only",
            "transformed_event_references": [],
            "selected_t1_use_type": use.event_type,
            "selected_t1_expired_at_use": use.expired_at_use,
            "selected_t1_use_to_replacement_s": round(
                replacement.replacement_timestamp - use.timestamp, 6
            ),
        })
        return _qualify_reference_hn(
            base, request, randomness_reference, provenance,
            target_ids=(use.event_id,),
            evidence_ids=(use.event_id, replacement.replacement_event_id),
            boundary_parameters={"selected_use_policy": PRE_REPLACEMENT_T1_SELECTION_POLICY_VERSION},
        )
    if request.variant_id == CORRECT_REPLACEMENT_T2_HN:
        use = replacement.post_replacement_use
        assert use is not None
        provenance.update({
            "construction_mode": "reference_only",
            "transformed_event_references": [],
            "selected_t2_use_type": use.event_type,
            "selected_t2_expired_at_use": use.expired_at_use,
            "selected_t2_use_after_replacement_s": round(
                use.timestamp - replacement.replacement_timestamp, 6
            ),
        })
        return _qualify_reference_hn(
            base, request, randomness_reference, provenance,
            target_ids=(use.event_id,),
            evidence_ids=(replacement.replacement_event_id, use.event_id),
            boundary_parameters={"selected_use_policy": POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION},
        )

    # Anomaly: reuse T1 at the same canonical post-replacement presentation
    # opportunity while preserving T2 establishment and the continuing session.
    post_use = replacement.post_replacement_use
    assert post_use is not None
    events, context = materialize_base(base)
    target = events[post_use.ordinal - 1]
    original_t2 = target.token_id
    target.token_id = replacement.t1_token_id
    target.current_token_id = replacement.t2_token_id
    target.token_generation = replacement.t1_generation
    target.token_lineage_status = "superseded"
    target.supersedes_token_id = None
    target.superseded_by_token_id = replacement.t2_token_id
    target.token_replacement_timestamp = replacement.replacement_timestamp
    target.token_expiry = replacement.t1_expiry
    target.token_scope = replacement.t1_scope
    target.token_bound_identity = replacement.t1_bound_identity
    target.token_bound_protected_session_id = (
        replacement.t1_bound_protected_session_id
        or replacement.protected_session_id
    )
    validation_evidence = replacement.post_replacement_validation
    assert validation_evidence is not None
    enforcement = events[validation_evidence.ordinal - 1]
    enforcement.event_type = EventType.TOKEN_REJECTED
    enforcement.result = EventResult.FAILURE
    enforcement.failure_reason = "superseded_token_rejected"
    enforcement.new_state = enforcement.previous_state
    enforcement.token_id = replacement.t1_token_id
    enforcement.current_token_id = replacement.t2_token_id
    enforcement.token_generation = replacement.t1_generation
    enforcement.token_lineage_status = "superseded"
    enforcement.supersedes_token_id = None
    enforcement.superseded_by_token_id = replacement.t2_token_id
    enforcement.token_replacement_timestamp = replacement.replacement_timestamp
    enforcement.token_expiry = replacement.t1_expiry
    enforcement.token_scope = replacement.t1_scope
    enforcement.token_bound_identity = replacement.t1_bound_identity
    enforcement.token_bound_protected_session_id = (
        replacement.t1_bound_protected_session_id
        or replacement.protected_session_id
    )
    enforcement.token_validation_result = "rejected"
    capability = VARIANT_CAPABILITIES[SUPERSEDED_TOKEN_REPLAY_ANOMALY]
    injection_id = "4d3.5-injection-" + stable_reference({
        "base": base.base_trace_reference,
        "variant": SUPERSEDED_TOKEN_REPLAY_ANOMALY,
        "replacement": replacement.replacement_ordinal,
        "post_use": post_use.ordinal,
        "randomness": randomness_reference,
    })[7:31]
    target.injection_id = injection_id
    target.anomaly_variant_name = SUPERSEDED_TOKEN_REPLAY_ANOMALY
    target.invariant_families = capability.invariant_families
    target.source_context = "controlled_injection"
    enforcement.injection_id = injection_id
    enforcement.anomaly_variant_name = SUPERSEDED_TOKEN_REPLAY_ANOMALY
    enforcement.invariant_families = capability.invariant_families
    enforcement.source_context = "controlled_enforcement"
    injection = InjectionRecord(
        injection_id=injection_id,
        historical_scenario_name="replay_token",
        anomaly_variant_name=SUPERSEDED_TOKEN_REPLAY_ANOMALY,
        capability_status=capability.status.value,
        mapped_semantic_concept=capability.semantic_concept,
        invariant_families=capability.invariant_families,
        canonical_invariant_ids=capability.invariant_ids,
        targeted_event_type=EventType.TOKEN_PRESENTED.value,
        targeted_action_or_context=(
            "canonical post-replacement presentation in continuing protected session"
        ),
        source_evidence_event_ids=(
            replacement.source_issuance_event_id,
            replacement.session_open_event_id,
            replacement.renewal_success_event_id,
            replacement.replacement_event_id,
        ),
        source_material_identifier=replacement.t1_token_id,
        source_trace_id=context.trace_id,
        source_session_id=context.legacy_session_id,
        source_auth_attempt_id=events[replacement.source_issuance_ordinal - 1].auth_attempt_id,
        original_value=original_t2,
        injected_value=replacement.t1_token_id,
        injection_parameters={
            "adapter_version": TOKEN_BOUNDARY_ADAPTER_VERSION,
            "token_evidence_resolver_version": TOKEN_EVIDENCE_RESOLVER_VERSION,
            "replacement_lineage_resolver_version": (
                TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION
            ),
            "post_replacement_selection_policy_version": (
                POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION
            ),
            "transformed_event_ids": [target.event_id, enforcement.event_id],
            "replacement_token_id": replacement.t2_token_id,
            "protected_session_id": replacement.protected_session_id,
            "replay_before_t1_expiry": target.timestamp <= replacement.t1_expiry,
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
        context,
        attack_type=SUPERSEDED_TOKEN_REPLAY_ANOMALY,
        attack_phase=EventType.TOKEN_PRESENTED.value,
        severity="medium",
        attack_scenario_intent=True,
        observable_anomaly=False,
        observable_violation_candidate=False,
        injection_records=(injection.to_dict(),),
    )
    provenance.update({
        "construction_mode": "post_replacement_token_local_substitution",
        "transformed_event_references": [
            f"event:{post_use.ordinal}", f"event:{validation_evidence.ordinal}",
        ],
        "transformed_token_event_ids": [target.event_id, enforcement.event_id],
        "enforcement_outcome": "superseded_token_rejected",
        "event_type_changed_for_coherent_enforcement": True,
        "original_post_replacement_token_id": original_t2,
        "derived_post_replacement_token_id": replacement.t1_token_id,
        "replay_before_t1_expiry": target.timestamp <= replacement.t1_expiry,
        "t1_expiry": replacement.t1_expiry,
        "replay_timestamp": target.timestamp,
    })
    evidence_report = _validation(events, context)
    expected = set(capability.invariant_ids)
    confirmed = set(
        evidence_report.trace_validated_invariant_ids.get(context.trace_id, ())
    )
    if not expected.issubset(confirmed):
        return _empty(
            base, request, TransformationStatus.CONSTRUCTION_ERROR,
            "independent validation did not confirm superseded-token replay",
            randomness_reference,
        )
    for event in events:
        if event.event_id in evidence_report.validated_positive_event_ids:
            event.anomaly_label = SUPERSEDED_TOKEN_REPLAY_ANOMALY
    final_injection = dict(context.injection_records[0])
    final_injection.update({
        "evidence_check_passed": True,
        "observable_violation_candidate": True,
        "validation_status": "validated_positive",
    })
    context = replace(
        context,
        observable_anomaly=True,
        observable_violation_candidate=True,
        injection_records=(final_injection,),
    )
    final_report = _validation(events, context)
    validated = tuple(
        final_report.trace_validated_invariant_ids.get(context.trace_id, ())
    )
    provenance.update({
        "validated_invariant_ids": list(validated),
        "token_validity_collision": "INV-TIME-TOKEN-VALIDITY" in validated,
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
        if final_report.build_gate_passed
        and final_report.trace_ground_truth_statuses.get(context.trace_id)
        == "validated_positive"
        and expected.issubset(set(validated))
        else TransformationStatus.CONSTRUCTION_ERROR
    )
    return DerivedRoleTrace(
        TOKEN_DERIVED_REFERENCE_VERSION, reference, base.base_trace_reference,
        request.variant_id, status, event_json, context_json, provenance_json,
        validated, final_report.trace_ground_truth_statuses.get(context.trace_id),
        final_report.build_gate_passed, None,
        "canonical post-replacement T2 presentation substituted coherently with T1",
    )
