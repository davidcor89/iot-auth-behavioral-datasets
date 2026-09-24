"""Deterministic positive-normal qualification for explicit HN candidates.

This module never constructs or selects behavior. Candidate intent establishes
which boundary to assess; emitted semantic evidence establishes normality. The
existing invariant validator remains authoritative for violation and GT status.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from simulator.event_model import AuthEvent, EventResult, EventType
from simulator.hard_negative import (
    ConstructionStatus,
    HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
    HN_CHECK_BY_ID,
    HN_VARIANT_BY_ID,
    HardNegativeQualificationRecord,
    QualificationStatus,
)


@dataclass(frozen=True)
class _Decision:
    status: QualificationStatus
    reason: str
    predicates: Mapping[str, Any]
    boundary: Mapping[str, Any]
    target_event_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    disqualifying_invariant_ids: tuple[str, ...] = ()


def _proof(satisfied: bool, event_ids: Iterable[str], **details: Any) -> dict[str, Any]:
    return {
        "satisfied": satisfied,
        "evidence_event_ids": list(dict.fromkeys(event_ids)),
        **details,
    }


def _decision(
    status: QualificationStatus,
    reason: str,
    *,
    predicates: Optional[Mapping[str, Any]] = None,
    boundary: Optional[Mapping[str, Any]] = None,
    targets: Iterable[str] = (),
    evidence: Iterable[str] = (),
    disqualifying: Iterable[str] = (),
) -> _Decision:
    return _Decision(
        status,
        reason,
        predicates or {},
        boundary or {},
        tuple(dict.fromkeys(targets)),
        tuple(dict.fromkeys(evidence)),
        tuple(dict.fromkeys(disqualifying)),
    )


def _observed_time(event: AuthEvent) -> Optional[float]:
    return event.observed_timestamp if event.observed_timestamp is not None else event.timestamp


def _referenced_events(
    events: Sequence[AuthEvent],
    candidate: HardNegativeQualificationRecord,
) -> tuple[list[AuthEvent], list[AuthEvent], list[str]]:
    by_id = {event.event_id: event for event in events}
    missing = [
        event_id
        for event_id in (*candidate.target_event_ids, *candidate.evidence_event_ids)
        if event_id not in by_id
    ]
    targets = [by_id[event_id] for event_id in candidate.target_event_ids if event_id in by_id]
    evidence = [by_id[event_id] for event_id in candidate.evidence_event_ids if event_id in by_id]
    return targets, evidence, missing


def _nonce_fresh_cross_attempt(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    if missing or not targets or not evidence:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "nonce relation lacks referenced events")
    target = next((event for event in targets if event.nonce is not None), None)
    source = next((event for event in evidence if event.event_id != getattr(target, "event_id", None) and event.nonce is not None), None)
    if target is None or source is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "both nonce occurrences must expose nonce material")
    refs = (source.event_id, target.event_id)
    if not source.auth_attempt_id or not target.auth_attempt_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "nonce occurrence has unknown authentication-attempt membership", targets=(target.event_id,), evidence=refs)
    if source.auth_attempt_id == target.auth_attempt_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "nonce occurrences are not associated with distinct attempts", targets=(target.event_id,), evidence=refs)
    if source.nonce == target.nonce:
        return _decision(
            QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION,
            "the same nonce occurs across distinct attempts",
            predicates={"attempts_distinct": _proof(True, refs)},
            targets=(target.event_id,), evidence=refs,
            disqualifying=("INV-REPLAY-NONCE-CROSS-ATTEMPT",),
        )
    link = next((event for event in events if (
        event.event_type in {EventType.RETRY, EventType.RENEWAL_REQUEST}
        and event.result is EventResult.SUCCESS
        and event.auth_attempt_id == target.auth_attempt_id
        and source.timestamp < event.timestamp < target.timestamp
        and event.trace_id == candidate.trace_id
        and event.device_id == candidate.device_id
    )), None)
    if link is None:
        return _decision(
            QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED,
            "distinct attempts are normal but no linked retry/re-auth event establishes the intended boundary",
            predicates={"attempts_distinct": _proof(True, refs), "nonces_distinct": _proof(True, refs)},
            targets=(target.event_id,), evidence=refs,
        )
    if link.event_id not in candidate.evidence_event_ids:
        return _decision(
            QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE,
            "related retry/re-auth event is not included in candidate evidence references",
            targets=(target.event_id,), evidence=refs,
        )
    relationship = "retry" if link.event_type is EventType.RETRY else "renewal"
    declared = candidate.boundary_parameters.get("attempt_relationship")
    if declared is not None and declared != relationship:
        return _decision(
            QualificationStatus.QUALIFICATION_ERROR,
            "declared attempt relationship contradicts the represented lifecycle event",
            targets=(target.event_id,), evidence=(*refs, link.event_id),
        )
    return _decision(
        QualificationStatus.QUALIFIED,
        "fresh nonce material is positively established across related distinct attempts",
        predicates={
            "attempts_distinct": _proof(True, refs, source_attempt_id=source.auth_attempt_id, target_attempt_id=target.auth_attempt_id),
            "nonces_distinct": _proof(True, refs),
            "within_trace_scope": _proof(source.trace_id == target.trace_id == candidate.trace_id, refs),
        },
        boundary={"boundary_reached": _proof(True, (*refs, link.event_id), attempt_relationship=relationship)},
        targets=(target.event_id,), evidence=(*refs, link.event_id),
    )


def _nonce_same_attempt(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    occurrences = list({event.event_id: event for event in (*targets, *evidence)}.values())
    if missing or len(occurrences) < 2 or any(event.nonce is None for event in occurrences):
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "same-attempt recurrence requires two referenced nonce-bearing events")
    if any(not event.auth_attempt_id for event in occurrences):
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "nonce attempt membership is unknown", targets=[event.event_id for event in targets], evidence=[event.event_id for event in occurrences])
    attempts = {event.auth_attempt_id for event in occurrences}
    nonces = {event.nonce for event in occurrences}
    refs = tuple(event.event_id for event in occurrences)
    if len(attempts) > 1 and len(nonces) == 1:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "nonce recurs across attempts", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-REPLAY-NONCE-CROSS-ATTEMPT",))
    if len(attempts) != 1 or len(nonces) != 1:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "the represented relation is not a same-nonce same-attempt recurrence", targets=[event.event_id for event in targets], evidence=refs)
    allowed = {EventType.CHALLENGE_SENT, EventType.NONCE_RECEIVED, EventType.RESPONSE_SENT}
    if any(event.event_type not in allowed for event in occurrences):
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "event relationship does not establish a supported within-attempt nonce recurrence", targets=[event.event_id for event in targets], evidence=refs)
    return _decision(
        QualificationStatus.QUALIFIED,
        "represented protocol events carry the same nonce in one authentication attempt",
        predicates={
            "nonce_material_equal": _proof(True, refs),
            "attempt_membership_equal": _proof(True, refs, auth_attempt_id=next(iter(attempts))),
            "represented_relation_permits_recurrence": _proof(True, refs),
        },
        boundary={"boundary_reached": _proof(True, refs, relation="same_attempt_protocol_recurrence")},
        targets=[event.event_id for event in targets], evidence=refs,
    )


def _token_valid_at(event: AuthEvent) -> Optional[bool]:
    if event.token_expiry is None:
        return None
    return event.timestamp <= event.token_expiry


def _token_repeated_current(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    uses = [
        event for event in {item.event_id: item for item in (*targets, *evidence)}.values()
        if event.event_type in {EventType.TOKEN_PRESENTED, EventType.ACCESS_REQUEST, EventType.ACCESS_GRANTED, EventType.ACCESS_DENIED}
    ]
    if missing or not uses or any(not event.token_id for event in uses):
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "current-token use requires referenced token identities")
    if len(uses) < 2:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "token is presented or used only once", targets=[event.event_id for event in targets], evidence=[event.event_id for event in uses])
    refs = tuple(event.event_id for event in uses)
    if len({event.token_id for event in uses}) != 1 or len({event.protected_session_id for event in uses}) != 1 or not uses[0].protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "stable token or continuing protected-session linkage cannot be established", targets=[event.event_id for event in targets], evidence=refs)
    if any(event.current_token_id is None or event.token_lineage_status is None for event in uses):
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "token replacement/currentness evidence is incomplete", targets=[event.event_id for event in targets], evidence=refs)
    assessed_token = uses[0].token_id
    last_use_time = max(event.timestamp for event in uses)
    replacement_before_reuse = next(
        (
            event for event in events
            if event.event_type is EventType.TOKEN_ISSUED
            and event.result is EventResult.SUCCESS
            and event.supersedes_token_id == assessed_token
            and event.timestamp < last_use_time
            and event.protected_session_id == uses[0].protected_session_id
        ),
        None,
    )
    if replacement_before_reuse is not None:
        replacement_refs = (*refs, replacement_before_reuse.event_id)
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "successful replacement occurred before the assessed reuse", targets=[event.event_id for event in targets], evidence=replacement_refs, disqualifying=("INV-REPLAY-TOKEN-SUPERSEDED",))
    if any(event.token_id != event.current_token_id or event.token_lineage_status == "superseded" for event in uses):
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "a superseded token is reused", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-REPLAY-TOKEN-SUPERSEDED",))
    validity = [_token_valid_at(event) for event in uses]
    if any(value is None for value in validity):
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "token validity interval is absent", targets=[event.event_id for event in targets], evidence=refs)
    if not all(validity):
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "token use is outside its represented validity interval", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-TIME-TOKEN-VALIDITY",))
    return _decision(
        QualificationStatus.QUALIFIED,
        "the same current valid token is used repeatedly in one continuing protected session",
        predicates={
            "stable_token_identity": _proof(True, refs),
            "presented_token_is_current": _proof(True, refs),
            "replacement_not_yet_effective": _proof(True, refs),
            "token_validity_satisfied": _proof(True, refs),
            "continuing_protected_session": _proof(True, refs),
        },
        boundary={"boundary_reached": _proof(True, refs, repeated_use_count=len(uses))},
        targets=[event.event_id for event in targets], evidence=refs,
    )


def _replacement_pair(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord
) -> tuple[Optional[AuthEvent], Optional[AuthEvent], list[AuthEvent], list[str]]:
    targets, evidence, missing = _referenced_events(events, candidate)
    referenced = list({item.event_id: item for item in (*targets, *evidence)}.values())
    replacements = [
        event for event in referenced
        if event.event_type == EventType.TOKEN_ISSUED and event.supersedes_token_id
    ]
    replacement = replacements[0] if replacements else None
    # A protected access request is an actual use/presentation of its token in
    # the continuing session; the initial TOKEN_PRESENTED event precedes open.
    use_types = (
        {EventType.TOKEN_PRESENTED, EventType.ACCESS_REQUEST}
        if candidate.intended_variant_id == "last_valid_use_before_replacement"
        else {EventType.TOKEN_PRESENTED}
    )
    presentations = [event for event in referenced if event.event_type in use_types]
    return replacement, presentations[0] if presentations else None, targets, missing


def _token_last_before_replacement(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    replacement, presentation, targets, missing = _replacement_pair(events, candidate)
    if missing or presentation is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "T1 presentation evidence is absent")
    if replacement is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "no represented replacement establishes the intended boundary", targets=[event.event_id for event in targets], evidence=(presentation.event_id,))
    refs = (presentation.event_id, replacement.event_id)
    if replacement.result is not EventResult.SUCCESS:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "the represented replacement was not successful", targets=[event.event_id for event in targets], evidence=refs)
    if not presentation.token_id or not replacement.token_id or not replacement.supersedes_token_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "token lineage identities are absent", targets=[event.event_id for event in targets], evidence=refs)
    if replacement.token_id == presentation.token_id or replacement.supersedes_token_id != presentation.token_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "T1-to-distinct-T2 replacement lineage is ambiguous", targets=[event.event_id for event in targets], evidence=refs)
    if not presentation.protected_session_id or presentation.protected_session_id != replacement.protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "continuing protected-session linkage is not established", targets=[event.event_id for event in targets], evidence=refs)
    if presentation.current_token_id is None or presentation.token_lineage_status is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "T1 currentness is not reconstructable", targets=[event.event_id for event in targets], evidence=refs)
    if presentation.current_token_id != presentation.token_id or presentation.token_lineage_status == "superseded" or presentation.timestamp >= replacement.timestamp:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "T1 presentation is not a current use before successful replacement", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-REPLAY-TOKEN-SUPERSEDED",))
    validity = _token_valid_at(presentation)
    if validity is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "T1 validity evidence is absent", targets=[event.event_id for event in targets], evidence=refs)
    if not validity:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "T1 is invalid at its assessed presentation", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-TIME-TOKEN-VALIDITY",))
    return _decision(
        QualificationStatus.QUALIFIED,
        "current T1 is presented before successful distinct T2 replacement",
        predicates={
            "t1_current_at_presentation": _proof(True, (presentation.event_id,)),
            "replacement_successful_and_distinct": _proof(True, (replacement.event_id,)),
            "presentation_precedes_replacement": _proof(True, refs),
            "continuing_protected_session": _proof(True, refs),
        },
        boundary={"boundary_reached": _proof(True, refs, relation="last_valid_use_before_replacement")},
        targets=[event.event_id for event in targets], evidence=refs,
    )


def _token_replacement_use(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    replacement, presentation, targets, missing = _replacement_pair(events, candidate)
    if missing or replacement is None or presentation is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "successful replacement and actual T2 presentation are both required")
    refs = (replacement.event_id, presentation.event_id)
    if replacement.result is not EventResult.SUCCESS:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "replacement success is unresolved", targets=[event.event_id for event in targets], evidence=refs)
    if not replacement.token_id or not replacement.supersedes_token_id or not presentation.token_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "replacement lineage or presentation identity is absent", targets=[event.event_id for event in targets], evidence=refs)
    if replacement.token_id == replacement.supersedes_token_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "replacement token is not distinct from its predecessor", targets=[event.event_id for event in targets], evidence=refs)
    if presentation.timestamp <= replacement.timestamp or presentation.protected_session_id != replacement.protected_session_id or not presentation.protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "ordered continuing-session replacement use is not established", targets=[event.event_id for event in targets], evidence=refs)
    if presentation.token_id != replacement.token_id or presentation.current_token_id != replacement.token_id or presentation.token_lineage_status != "current":
        if presentation.token_id == replacement.supersedes_token_id:
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "superseded T1 is presented after replacement", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-REPLAY-TOKEN-SUPERSEDED",))
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "actual presented token is not established as current T2", targets=[event.event_id for event in targets], evidence=refs)
    validity = _token_valid_at(presentation)
    if validity is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "T2 validity evidence is absent", targets=[event.event_id for event in targets], evidence=refs)
    if not validity:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "T2 is used outside its validity interval", targets=[event.event_id for event in targets], evidence=refs, disqualifying=("INV-TIME-TOKEN-VALIDITY",))
    return _decision(
        QualificationStatus.QUALIFIED,
        "successfully established replacement T2 is actually presented as current",
        predicates={
            "replacement_successful_and_distinct": _proof(True, (replacement.event_id,)),
            "replacement_precedes_presentation": _proof(True, refs),
            "presented_token_is_current_replacement": _proof(True, (presentation.event_id,)),
            "continuing_protected_session": _proof(True, refs),
        },
        boundary={"boundary_reached": _proof(True, refs, relation="first_correct_replacement_use")},
        targets=[event.event_id for event in targets], evidence=refs,
    )


def _positive_causal_margin(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, _context: Any, _report: Any
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    if missing or not targets:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "causal target event is absent")
    target = targets[0]
    predecessor = next((event for event in evidence if event.event_id != target.event_id), None)
    if predecessor is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "explicit causal predecessor is absent")
    refs = (predecessor.event_id, target.event_id)
    predecessor_observed, target_observed = _observed_time(predecessor), _observed_time(target)
    if predecessor_observed is None or target_observed is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "comparable observed timestamps are absent", targets=(target.event_id,), evidence=refs)
    if target.timestamp < predecessor.timestamp or target_observed < predecessor_observed:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "causal timestamp order is inverted", targets=(target.event_id,), evidence=refs, disqualifying=("INV-TIME-OBSERVED-CAUSAL",))
    epsilon = target_observed - predecessor_observed
    if target.timestamp == predecessor.timestamp or epsilon == 0:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "timestamp equality remains unresolved for the first HN programme", targets=(target.event_id,), evidence=refs)
    epsilon_hn = candidate.boundary_parameters.get("epsilon_hn")
    if (not isinstance(epsilon_hn, (int, float)) or isinstance(epsilon_hn, bool)
            or not math.isfinite(epsilon_hn) or epsilon_hn <= 0):
        return _decision(QualificationStatus.QUALIFICATION_ERROR, "a positive candidate-specific epsilon_hn parameter is required", targets=(target.event_id,), evidence=refs)
    predicates = {
        "semantic_predecessor_order": _proof(predecessor.timestamp < target.timestamp, refs),
        "strict_observed_order": _proof(epsilon > 0, refs, epsilon=epsilon),
    }
    if epsilon > epsilon_hn:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "positive observed margin exceeds the declared HN construction margin", predicates=predicates, targets=(target.event_id,), evidence=refs)
    return _decision(
        QualificationStatus.QUALIFIED,
        "strict positive observed causal margin falls within the configured HN margin",
        predicates=predicates,
        boundary={"boundary_reached": _proof(True, refs, epsilon=epsilon, epsilon_hn=epsilon_hn)},
        targets=(target.event_id,), evidence=refs,
    )


def _access_relation(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord
) -> tuple[Optional[AuthEvent], Optional[AuthEvent], list[str]]:
    targets, evidence, missing = _referenced_events(events, candidate)
    operation = next(
        (event for event in targets if event.event_type in {EventType.ACCESS_GRANTED, EventType.ACCESS_DENIED}),
        targets[0] if targets else None,
    )
    request = next(
        (
            event for event in evidence
            if operation is not None
            and event.event_type == EventType.ACCESS_REQUEST
            and event.access_request_id
            and event.access_request_id == operation.access_request_id
            and event.timestamp < operation.timestamp
        ),
        None,
    )
    return operation, request, missing


def _access_check(
    check_id: str,
    events: Sequence[AuthEvent],
    candidate: HardNegativeQualificationRecord,
) -> _Decision:
    operation, request, missing = _access_relation(events, candidate)
    if missing or operation is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "protected operation evidence is absent")
    if operation.resource_operation_outcome is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "authoritative operation outcome is absent", targets=(operation.event_id,))
    if request is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "request-to-operation linkage is not reconstructable", targets=(operation.event_id,), evidence=(operation.event_id,))
    refs = (request.event_id, operation.event_id)
    success = operation.result is EventResult.SUCCESS and operation.resource_operation_outcome == "success"
    non_success = operation.resource_operation_outcome in {"denied", "not_executed"}
    auth_valid = bool(operation.authenticated_context_active and operation.authenticated_identity == operation.device_id)
    credential_valid = bool(operation.token_id and operation.token_context_active and operation.token_validation_result == "validated")
    session_active = bool(operation.protected_session_id and operation.protected_session_active)
    grant = operation.authorization_decision == "granted"
    deny = operation.authorization_decision == "denied"
    common_boundary = {"boundary_reached": _proof(True, refs, relation="same_request_authoritative_enforcement")}

    requirements: dict[str, tuple[Optional[bool], bool, str]] = {
        "hn.v1.access.missing_auth_enforced": (operation.authentication_required, not auth_valid, "authentication context"),
        "hn.v1.access.invalid_credential_enforced": (operation.credential_required, not credential_valid, "credential context"),
        "hn.v1.access.inactive_session_enforced": (operation.active_session_required, not session_active, "active protected session"),
    }
    if check_id in requirements:
        applicable, missing_prerequisite, label = requirements[check_id]
        if applicable is None:
            return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, f"{label} applicability is unknown", targets=(operation.event_id,), evidence=refs)
        if applicable is not True or not missing_prerequisite:
            return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, f"the intended missing {label} boundary is not present", targets=(operation.event_id,), evidence=refs)
        invariant_id = HN_VARIANT_BY_ID[candidate.intended_variant_id].invariant_ids[0]
        if success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, f"protected operation succeeds despite missing {label}", targets=(operation.event_id,), evidence=refs, disqualifying=(invariant_id,))
        if not non_success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "non-success must be positively represented as denied or not executed", targets=(operation.event_id,), evidence=refs)
        return _decision(
            QualificationStatus.QUALIFIED,
            f"missing {label} is positively represented and correctly enforced",
            predicates={f"{label.replace(' ', '_')}_invalid_or_absent": _proof(True, refs), "operation_non_success": _proof(True, refs, outcome=operation.resource_operation_outcome)},
            boundary=common_boundary, targets=(operation.event_id,), evidence=refs,
        )

    if operation.authorization_applicable is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "authorization applicability is unknown", targets=(operation.event_id,), evidence=refs)
    if not operation.authorization_evidence_complete or not operation.authorization_authoritative:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "authorization decision evidence is incomplete or non-authoritative", targets=(operation.event_id,), evidence=refs)

    if check_id == "hn.v1.access.no_grant_enforced":
        if operation.authorization_applicable is not True or grant:
            return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "the intended no-grant boundary is not present", targets=(operation.event_id,), evidence=refs)
        if success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "operation succeeds without an applicable grant", targets=(operation.event_id,), evidence=refs, disqualifying=("INV-ACCESS-AUTHZ-GRANT",))
        if not non_success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "no authoritative non-success outcome is represented", targets=(operation.event_id,), evidence=refs)
        return _decision(QualificationStatus.QUALIFIED, "absence of an applicable grant is correctly enforced", predicates={"applicable_grant_absent": _proof(True, refs), "operation_non_success": _proof(True, refs)}, boundary=common_boundary, targets=(operation.event_id,), evidence=refs)

    if check_id == "hn.v1.access.explicit_deny_enforced":
        if not deny:
            return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "no explicit authoritative denial is represented", targets=(operation.event_id,), evidence=refs)
        if success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "operation succeeds despite explicit denial", targets=(operation.event_id,), evidence=refs, disqualifying=("INV-ACCESS-DENY-ENFORCED",))
        if not non_success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "denial lacks a positive non-success operation outcome", targets=(operation.event_id,), evidence=refs)
        return _decision(QualificationStatus.QUALIFIED, "explicit authoritative denial prevents execution", predicates={"decision_is_deny": _proof(True, refs), "operation_non_success": _proof(True, refs)}, boundary=common_boundary, targets=(operation.event_id,), evidence=refs)

    if check_id == "hn.v1.authz.valid_context_legitimate_denial":
        if not (auth_valid and credential_valid and session_active):
            return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "authentication, credential, and active-session context are not all valid", targets=(operation.event_id,), evidence=refs)
        if grant and not deny:
            return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "neither a missing grant nor an explicit denial is represented", targets=(operation.event_id,), evidence=refs)
        if success:
            disqualifying = "INV-ACCESS-DENY-ENFORCED" if deny else "INV-ACCESS-AUTHZ-GRANT"
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "authorization boundary is ignored by a successful operation", targets=(operation.event_id,), evidence=refs, disqualifying=(disqualifying,))
        if not non_success:
            return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "legitimate denial lacks a positive non-success outcome", targets=(operation.event_id,), evidence=refs)
        return _decision(
            QualificationStatus.QUALIFIED,
            "valid authentication, credential, and session context is correctly separated from authorization denial",
            predicates={
                "authentication_context_valid": _proof(True, refs),
                "credential_context_valid": _proof(True, refs),
                "protected_session_active": _proof(True, refs),
                "authorization_not_granted": _proof(True, refs, decision=operation.authorization_decision),
                "operation_non_success": _proof(True, refs),
            },
            boundary=common_boundary, targets=(operation.event_id,), evidence=refs,
        )
    return _decision(QualificationStatus.QUALIFICATION_ERROR, f"unknown access qualification check: {check_id}")


def _identity_coherent(
    events: Sequence[AuthEvent], candidate: HardNegativeQualificationRecord, context: Any, report: Any
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    target = next((event for event in targets if event.event_type == EventType.AUTHENTICATION_SUCCESS), None)
    if missing or target is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "successful authentication target is absent")
    persistent_identity = getattr(context, "device_id", None)
    if not persistent_identity or not target.authenticated_identity:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "persistent or authenticated identity evidence is absent", targets=(target.event_id,))
    if not target.auth_attempt_id or target.auth_attempt_id not in set(getattr(context, "auth_attempt_ids", ()) or ()):
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "authentication-context linkage is ambiguous", targets=(target.event_id,))
    association = next((event for event in evidence if event.auth_attempt_id == target.auth_attempt_id and event.timestamp < target.timestamp and event.device_id == persistent_identity), None)
    if association is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "device-to-authentication-context association is not independently evidenced", targets=(target.event_id,), evidence=(target.event_id,))
    validator_record = next((item for item in getattr(report, "records", ()) if getattr(item, "validation_id", None) == "c2.3.identity.persistent_authenticated" and getattr(item, "trace_id", None) == candidate.trace_id), None)
    comparisons = getattr(validator_record, "observed_evidence", {}).get("comparisons", ()) if validator_record else ()
    independent = next((item for item in comparisons if item.get("authentication_event_id") == target.event_id and item.get("context_device_association_reliable") and item.get("persistent_identity_source") and item.get("authenticated_identity_source") and item.get("persistent_identity_source") != item.get("authenticated_identity_source")), None)
    refs = (association.event_id, target.event_id)
    if independent is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "identity equality is not backed by independent represented sources", targets=(target.event_id,), evidence=refs)
    if persistent_identity != target.authenticated_identity:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "persistent and authenticated identities differ", targets=(target.event_id,), evidence=refs, disqualifying=("INV-IDENT-AUTH-ACTOR-CONSISTENCY",))
    if candidate.boundary_parameters.get("boundary_targeted") is not True:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "ordinary identity equality alone does not establish HN proximity", predicates={"persistent_authenticated_identity_equal": _proof(True, refs)}, targets=(target.event_id,), evidence=refs)
    return _decision(
        QualificationStatus.QUALIFIED,
        "independently represented persistent and authenticated identities are coherent at the targeted boundary",
        predicates={
            "identity_sources_independent": _proof(True, refs),
            "authentication_context_successful": _proof(True, (target.event_id,)),
            "persistent_authenticated_identity_equal": _proof(True, refs),
            "context_device_association_reliable": _proof(True, refs),
        },
        boundary={"boundary_reached": _proof(True, refs, relation="supported_identity_equality_boundary")},
        targets=(target.event_id,), evidence=refs,
    )


def _binding_coherent(
    check_id: str,
    events: Sequence[AuthEvent],
    candidate: HardNegativeQualificationRecord,
) -> _Decision:
    targets, evidence, missing = _referenced_events(events, candidate)
    use = next((event for event in targets if event.token_id and event.event_type in {EventType.TOKEN_PRESENTED, EventType.TOKEN_VALIDATED, EventType.SESSION_OPENED, EventType.ACCESS_REQUEST, EventType.ACCESS_GRANTED, EventType.ACCESS_DENIED}), None)
    if missing or use is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "actual token presentation/use evidence is absent")
    issuance = next((event for event in evidence if event.event_type == EventType.TOKEN_ISSUED and event.result is EventResult.SUCCESS and event.token_id == use.token_id and event.timestamp < use.timestamp), None)
    if issuance is None:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "token use cannot be linked reliably to issuance", targets=(use.event_id,), evidence=(use.event_id,))
    refs = (issuance.event_id, use.event_id)
    if candidate.boundary_parameters.get("boundary_targeted") is not True:
        return _decision(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, "ordinary binding equality alone does not establish HN proximity", targets=(use.event_id,), evidence=refs)
    if check_id == "hn.v1.binding.identity_coherent":
        if not use.token_bound_identity or not use.authenticated_identity:
            return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "token-bound or authenticated identity is absent", targets=(use.event_id,), evidence=refs)
        if use.token_bound_identity != use.authenticated_identity:
            return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "token-bound identity differs from authenticated identity", targets=(use.event_id,), evidence=refs, disqualifying=("INV-BIND-TOKEN-CONTEXT",))
        if issuance.token_bound_identity != use.token_bound_identity:
            return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "token-bound identity is not coherent across issuance and use", targets=(use.event_id,), evidence=refs)
        return _decision(QualificationStatus.QUALIFIED, "explicit token-bound identity equals authenticated identity at actual use", predicates={"token_use_link_reliable": _proof(True, refs), "token_bound_identity_equal_authenticated_identity": _proof(True, refs)}, boundary={"boundary_reached": _proof(True, refs, relation="supported_token_identity_binding_boundary")}, targets=(use.event_id,), evidence=refs)
    if not use.token_bound_protected_session_id or not use.protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, "token-bound or active protected context is absent", targets=(use.event_id,), evidence=refs)
    if issuance.token_bound_protected_session_id and issuance.token_bound_protected_session_id != use.token_bound_protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, "represented context binding changes between issuance and use", targets=(use.event_id,), evidence=refs)
    if use.token_bound_protected_session_id != use.protected_session_id:
        return _decision(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "token-bound protected context differs from active protected context", targets=(use.event_id,), evidence=refs, disqualifying=("INV-BIND-TOKEN-CONTEXT",))
    return _decision(QualificationStatus.QUALIFIED, "explicit token-bound protected context equals the context of actual use", predicates={"token_use_link_reliable": _proof(True, refs), "token_bound_context_equal_active_context": _proof(True, refs)}, boundary={"boundary_reached": _proof(True, refs, relation="supported_token_context_binding_boundary")}, targets=(use.event_id,), evidence=refs)


_EVALUATORS: Mapping[str, Callable[[Sequence[AuthEvent], HardNegativeQualificationRecord, Any, Any], _Decision]] = {
    "hn.v1.nonce.fresh_cross_attempt": _nonce_fresh_cross_attempt,
    "hn.v1.nonce.same_attempt_recurrence": _nonce_same_attempt,
    "hn.v1.token.current_repeated_use": _token_repeated_current,
    "hn.v1.token.last_use_before_replacement": _token_last_before_replacement,
    "hn.v1.token.replacement_token_use": _token_replacement_use,
    "hn.v1.time.positive_causal_margin": _positive_causal_margin,
    "hn.v1.access.missing_auth_enforced": lambda e, c, _x, _r: _access_check("hn.v1.access.missing_auth_enforced", e, c),
    "hn.v1.access.invalid_credential_enforced": lambda e, c, _x, _r: _access_check("hn.v1.access.invalid_credential_enforced", e, c),
    "hn.v1.access.inactive_session_enforced": lambda e, c, _x, _r: _access_check("hn.v1.access.inactive_session_enforced", e, c),
    "hn.v1.access.no_grant_enforced": lambda e, c, _x, _r: _access_check("hn.v1.access.no_grant_enforced", e, c),
    "hn.v1.access.explicit_deny_enforced": lambda e, c, _x, _r: _access_check("hn.v1.access.explicit_deny_enforced", e, c),
    "hn.v1.authz.valid_context_legitimate_denial": lambda e, c, _x, _r: _access_check("hn.v1.authz.valid_context_legitimate_denial", e, c),
    "hn.v1.identity.persistent_authenticated_coherent": _identity_coherent,
    "hn.v1.binding.identity_coherent": lambda e, c, _x, _r: _binding_coherent("hn.v1.binding.identity_coherent", e, c),
    "hn.v1.binding.protected_context_coherent": lambda e, c, _x, _r: _binding_coherent("hn.v1.binding.protected_context_coherent", e, c),
}


def _validator_evidence(
    report: Any,
    trace_id: str,
    check_id: str,
) -> tuple[dict[str, str], Optional[QualificationStatus], tuple[str, ...], str]:
    check = HN_CHECK_BY_ID[check_id]
    outcomes: dict[str, str] = {}
    disqualifying: list[str] = []
    blocking: Optional[QualificationStatus] = None
    reason = ""
    records = tuple(getattr(report, "records", ()) or ())
    for validator_id in check.validator_check_ids:
        matches = [
            item for item in records
            if getattr(item, "validation_id", None) == validator_id
            and getattr(item, "trace_id", None) in {None, trace_id}
        ]
        if not matches:
            outcomes[validator_id] = "NOT_REPORTED_NO_RELEVANT_VALIDATOR_RECORD"
            continue
        values = [getattr(getattr(item, "result", None), "value", getattr(item, "result", None)) for item in matches]
        outcome = values[0] if len(set(values)) == 1 else "CONFLICTING_RESULTS"
        outcomes[validator_id] = str(outcome)
        invariant_ids = [getattr(item, "invariant_identifier", None) for item in matches]
        if "ERROR" in values or outcome == "CONFLICTING_RESULTS":
            blocking, reason = QualificationStatus.QUALIFICATION_ERROR, "relevant invariant validation errored or disagreed"
        elif "UNSUPPORTED" in values:
            blocking, reason = QualificationStatus.NOT_QUALIFIED_RELEVANT_UNSUPPORTED, "a required relation is unsupported by the validator"
        elif "FAIL" in values:
            blocking, reason = QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "a relevant invariant validation failed"
            disqualifying.extend(item for item in invariant_ids if item)
        elif "PASS" in values:
            # These mapped atomic validators use PASS for a positively reconstructed violation.
            blocking, reason = QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, "a relevant atomic validator positively reconstructed a violation"
            disqualifying.extend(item for item in invariant_ids if item)
    return outcomes, blocking, tuple(dict.fromkeys(disqualifying)), reason


def qualify_hard_negative(
    events: Sequence[AuthEvent],
    context: Any,
    candidate: HardNegativeQualificationRecord,
    validation_report: Any,
) -> HardNegativeQualificationRecord:
    """Qualify one explicitly constructed candidate without mutating its trace."""
    structural_errors = candidate.structural_errors()
    variant = HN_VARIANT_BY_ID.get(candidate.intended_variant_id or "")
    if structural_errors or variant is None or set(candidate.intended_boundary_invariant_ids) != set(variant.invariant_ids):
        reason = "; ".join(structural_errors) or "candidate invariant mapping does not exactly match its registered variant"
        return replace(
            candidate,
            qualification_status=QualificationStatus.QUALIFICATION_ERROR,
            qualified_family_ids=(), qualified_boundary_invariant_ids=(),
            qualification_check_ids=(), normal_side_predicates={}, boundary_proximity_evidence={},
            relevant_validation_check_ids=(), relevant_validation_outcomes={},
            disqualifying_invariant_ids=(), qualification_reason=reason,
            qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        )
    context_identity = (
        getattr(context, "trace_id", None),
        getattr(context, "scenario_id", None),
        getattr(context, "device_id", None),
    )
    candidate_identity = (candidate.trace_id, candidate.scenario_id, candidate.device_id)
    referenced_ids = set(candidate.target_event_ids) | set(candidate.evidence_event_ids)
    referenced = [event for event in events if event.event_id in referenced_ids]
    inconsistent_reference = any(
        (event.trace_id, event.scenario_id, event.device_id) != candidate_identity
        for event in referenced
    )
    if context_identity != candidate_identity or inconsistent_reference:
        return replace(
            candidate,
            qualification_status=QualificationStatus.QUALIFICATION_ERROR,
            qualified_family_ids=(), qualified_boundary_invariant_ids=(),
            qualification_check_ids=(), normal_side_predicates={}, boundary_proximity_evidence={},
            relevant_validation_check_ids=(), relevant_validation_outcomes={},
            disqualifying_invariant_ids=(),
            qualification_reason="candidate, context, and referenced semantic evidence identities disagree",
            qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        )
    if candidate.construction_status is ConstructionStatus.REQUESTED_NOT_APPLIED:
        return replace(candidate, qualification_status=QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, qualification_check_ids=variant.qualification_check_ids, qualification_reason="requested construction was not applied", qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION)
    if candidate.construction_status is ConstructionStatus.CONSTRUCTION_ERROR:
        return replace(candidate, qualification_status=QualificationStatus.QUALIFICATION_ERROR, qualification_check_ids=variant.qualification_check_ids, qualification_reason="candidate construction reported an implementation error", qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION)
    if candidate.construction_status is ConstructionStatus.PARTIALLY_APPLIED:
        return replace(candidate, qualification_status=QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, qualification_check_ids=variant.qualification_check_ids, qualification_reason="candidate construction was only partially applied", qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION)
    if candidate.construction_status is not ConstructionStatus.APPLIED:
        return replace(candidate, qualification_status=QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, qualification_check_ids=variant.qualification_check_ids, qualification_reason="candidate construction was not applied", qualifier_implementation_version=HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION)

    trace_status = getattr(validation_report, "trace_ground_truth_statuses", {}).get(candidate.trace_id)
    trace_invariants = tuple(getattr(validation_report, "trace_validated_invariant_ids", {}).get(candidate.trace_id, ()))
    observable_anomaly = bool(getattr(context, "observable_anomaly", False) or any(event.anomaly_label for event in events))
    check_id = variant.qualification_check_ids[0]
    outcomes, validator_block, validator_invariants, validator_reason = _validator_evidence(validation_report, candidate.trace_id, check_id)

    base_updates = {
        "qualification_check_ids": (check_id,),
        "relevant_validation_check_ids": HN_CHECK_BY_ID[check_id].validator_check_ids,
        "relevant_validation_outcomes": outcomes,
        "qualifier_implementation_version": HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        "trace_ground_truth_status": trace_status,
        "trace_observable_anomaly": observable_anomaly,
    }
    if observable_anomaly or trace_status == "validated_positive" or trace_invariants:
        return replace(candidate, qualification_status=QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, qualified_family_ids=(), qualified_boundary_invariant_ids=(), normal_side_predicates={}, boundary_proximity_evidence={}, disqualifying_invariant_ids=tuple(dict.fromkeys((*trace_invariants, *validator_invariants))), qualification_reason="trace anomaly GT or validated invariant evidence is positive", **base_updates)
    if trace_status == "validation_failed":
        return replace(candidate, qualification_status=QualificationStatus.QUALIFICATION_ERROR, qualified_family_ids=(), qualified_boundary_invariant_ids=(), normal_side_predicates={}, boundary_proximity_evidence={}, disqualifying_invariant_ids=validator_invariants, qualification_reason="trace validation failed", **base_updates)
    if trace_status is None:
        return replace(candidate, qualification_status=QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, qualified_family_ids=(), qualified_boundary_invariant_ids=(), normal_side_predicates={}, boundary_proximity_evidence={}, disqualifying_invariant_ids=(), qualification_reason="trace validation result is absent", **base_updates)

    decision = _EVALUATORS[check_id](events, candidate, context, validation_report)
    if decision.status is QualificationStatus.QUALIFIED and validator_block is not None:
        decision = replace(decision, status=validator_block, reason=validator_reason, disqualifying_invariant_ids=validator_invariants)
    qualified = decision.status is QualificationStatus.QUALIFIED
    return replace(
        candidate,
        qualification_status=decision.status,
        qualified_family_ids=(candidate.primary_intended_family_id,) if qualified and candidate.primary_intended_family_id else (),
        qualified_boundary_invariant_ids=variant.invariant_ids if qualified else (),
        normal_side_predicates=decision.predicates,
        boundary_proximity_evidence=decision.boundary,
        target_event_ids=decision.target_event_ids or candidate.target_event_ids,
        evidence_event_ids=decision.evidence_event_ids or candidate.evidence_event_ids,
        disqualifying_invariant_ids=decision.disqualifying_invariant_ids,
        qualification_reason=decision.reason,
        **base_updates,
    )


def qualification_engine_report() -> dict[str, Any]:
    """Return passive HN.2 capability without implying HN generation."""
    return {
        "qualifier_implementation_version": HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        "implemented_check_ids": sorted(_EVALUATORS),
        "generation_available": False,
        "sampling_available": False,
        "qualification_requires_explicit_candidate": True,
        "deterministic": True,
    }
