"""Evidence-based, non-canonical opportunity predicates for future roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from simulator.role_integration.models import ValidBaseTrace, stable_reference


APPLICABILITY_VOCABULARY_VERSION = "4d3.1-role-applicability-vocabulary-v1"
APPLICABILITY_EVALUATOR_VERSION = "4d3.1-evidence-applicability-evaluator-v1"


class ApplicabilityPredicate(str, Enum):
    SUCCESSFUL_AUTHENTICATION = "successful_authentication_exists"
    DISTINCT_AUTHENTICATION_ATTEMPTS = "distinct_authentication_attempts_exist"
    SAME_ATTEMPT_NONCE_EVIDENCE = "same_attempt_nonce_evidence_exists"
    ACTIVE_PROTECTED_SESSION = "active_protected_session_exists"
    PROTECTED_ACCESS = "protected_access_exists"
    MULTIPLE_CURRENT_TOKEN_USES = "multiple_current_token_uses_exist"
    AUTHORITATIVE_GRANT = "authoritative_grant_exists"
    AUTHORITATIVE_DENIAL = "authoritative_denial_exists"
    SUCCESSFUL_TOKEN_REPLACEMENT = "successful_token_replacement_exists"
    CURRENT_T1_USE_BEFORE_REPLACEMENT = "current_t1_use_before_replacement_exists"
    T2_PRESENTATION_AFTER_REPLACEMENT = "t2_presentation_after_replacement_exists"
    PERSISTENT_IDENTITY_ASSOCIATION = "persistent_identity_association_exists"
    TOKEN_AUTHENTICATED_IDENTITY_BINDING = "token_authenticated_identity_binding_evidence_exists"
    TOKEN_PROTECTED_CONTEXT_BINDING = "token_protected_context_binding_evidence_exists"


@dataclass(frozen=True)
class ApplicabilityResult:
    predicate_id: str
    vocabulary_version: str
    evaluator_version: str
    applicable: bool
    reason: str
    evidence_references: tuple[str, ...]
    evaluation_reference: str

    def to_record(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "vocabulary_version": self.vocabulary_version,
            "evaluator_version": self.evaluator_version,
            "applicable": self.applicable,
            "reason": self.reason,
            "evidence_references": list(self.evidence_references),
            "evaluation_reference": self.evaluation_reference,
        }


def _refs(indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(f"event:{index + 1}" for index in sorted(set(indices)))


def _result(
    base: ValidBaseTrace,
    predicate: ApplicabilityPredicate,
    indices: Iterable[int],
    *,
    positive_reason: str,
    negative_reason: str,
) -> ApplicabilityResult:
    evidence = _refs(indices)
    applicable = bool(evidence)
    reason = positive_reason if applicable else negative_reason
    evaluation_reference = stable_reference({
        "vocabulary_version": APPLICABILITY_VOCABULARY_VERSION,
        "evaluator_version": APPLICABILITY_EVALUATOR_VERSION,
        "base_trace_reference": base.base_trace_reference,
        "predicate_id": predicate.value,
        "applicable": applicable,
        "reason": reason,
        "evidence_references": evidence,
    })
    return ApplicabilityResult(
        predicate_id=predicate.value,
        vocabulary_version=APPLICABILITY_VOCABULARY_VERSION,
        evaluator_version=APPLICABILITY_EVALUATOR_VERSION,
        applicable=applicable,
        reason=reason,
        evidence_references=evidence,
        evaluation_reference=evaluation_reference,
    )


def _event_type(record: Mapping[str, Any]) -> str:
    return str(record.get("event_type") or "")


def evaluate_applicability(base: ValidBaseTrace) -> tuple[ApplicabilityResult, ...]:
    """Evaluate every supported predicate from represented immutable evidence."""
    events = base.event_records
    by_type: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        by_type.setdefault(_event_type(event), []).append(index)

    success_auth = [
        index for index in by_type.get("authentication_success", ())
        if events[index].get("result") == "success"
    ]
    attempt_groups: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        attempt = event.get("auth_attempt_id")
        if attempt:
            attempt_groups.setdefault(str(attempt), []).append(index)
    distinct_attempts = sorted(
        (indices[0] for indices in attempt_groups.values()),
    ) if len(attempt_groups) >= 2 else []

    nonce_groups: dict[tuple[str, str], list[int]] = {}
    for index, event in enumerate(events):
        attempt, nonce = event.get("auth_attempt_id"), event.get("nonce")
        if attempt and nonce:
            nonce_groups.setdefault((str(attempt), str(nonce)), []).append(index)
    same_nonce = next((indices for indices in nonce_groups.values() if len(indices) >= 2), [])

    active_session = [
        index for index in by_type.get("session_opened", ())
        if events[index].get("result") == "success"
        and events[index].get("protected_session_active") is True
    ]
    protected_access = [
        index for index in by_type.get("access_granted", ())
        if events[index].get("result") == "success"
        and events[index].get("resource_operation_outcome") == "success"
    ]
    grants = [
        index for index in protected_access
        if events[index].get("authorization_decision") == "granted"
        and events[index].get("authorization_authoritative") is True
        and events[index].get("authorization_evidence_complete") is True
    ]
    denials = [
        index for index in by_type.get("access_denied", ())
        if events[index].get("authorization_decision") == "denied"
        and events[index].get("authorization_authoritative") is True
        and events[index].get("authorization_evidence_complete") is True
    ]

    # Presentation/validation/grant are one credential-use chain, not distinct
    # protected uses. Count request evidence so A=1 is not mischaracterized as
    # multiple use and A>=2 is supported by distinct access-request records.
    use_types = {"access_request"}
    uses_by_token: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        token = event.get("token_id")
        if token and _event_type(event) in use_types and event.get("token_lineage_status") != "superseded":
            uses_by_token.setdefault(str(token), []).append(index)
    multiple_uses = next((indices for indices in uses_by_token.values() if len(indices) >= 2), [])

    replacements = [
        index for index in by_type.get("token_issued", ())
        if events[index].get("result") == "success"
        and events[index].get("supersedes_token_id")
    ]
    t1_before: list[int] = []
    t2_after: list[int] = []
    for replacement_index in replacements:
        replacement = events[replacement_index]
        t1 = replacement.get("supersedes_token_id")
        t2 = replacement.get("token_id")
        before = [
            index for index, event in enumerate(events[:replacement_index])
            if event.get("token_id") == t1
            and _event_type(event) in {"token_presented", "access_request"}
        ]
        after = [
            index for index, event in enumerate(events[replacement_index + 1:], replacement_index + 1)
            if event.get("token_id") == t2 and _event_type(event) == "token_presented"
        ]
        if before and not t1_before:
            t1_before = [before[-1], replacement_index]
        if after and not t2_after:
            t2_after = [replacement_index, after[0]]

    identity = [
        index for index in success_auth
        if events[index].get("authenticated_identity") == events[index].get("device_id")
    ]
    identity_binding = [
        index for index, event in enumerate(events)
        if event.get("token_id")
        and event.get("token_bound_identity")
        and event.get("authenticated_identity")
        and event.get("token_bound_identity") == event.get("authenticated_identity")
    ]
    context_binding = [
        index for index, event in enumerate(events)
        if event.get("token_id")
        and event.get("token_bound_protected_session_id")
        and event.get("protected_session_id")
        and event.get("token_bound_protected_session_id") == event.get("protected_session_id")
    ]

    specifications: tuple[tuple[ApplicabilityPredicate, Iterable[int], str, str], ...] = (
        (ApplicabilityPredicate.SUCCESSFUL_AUTHENTICATION, success_auth,
         "represented successful authentication event", "no represented successful authentication"),
        (ApplicabilityPredicate.DISTINCT_AUTHENTICATION_ATTEMPTS, distinct_attempts,
         "at least two represented authentication attempts", "fewer than two represented authentication attempts"),
        (ApplicabilityPredicate.SAME_ATTEMPT_NONCE_EVIDENCE, same_nonce,
         "nonce occurs in multiple events of one authentication attempt", "no resolvable same-attempt nonce chain"),
        (ApplicabilityPredicate.ACTIVE_PROTECTED_SESSION, active_session,
         "successful protected-session opening is represented", "no successful active protected session is represented"),
        (ApplicabilityPredicate.PROTECTED_ACCESS, protected_access,
         "successful protected operation is represented", "no successful protected operation is represented"),
        (ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES, multiple_uses,
         "current token has multiple represented uses", "no current token has multiple represented uses"),
        (ApplicabilityPredicate.AUTHORITATIVE_GRANT, grants,
         "complete authoritative grant evidence is represented", "no complete authoritative grant evidence is represented"),
        (ApplicabilityPredicate.AUTHORITATIVE_DENIAL, denials,
         "complete authoritative denial evidence is represented", "no complete authoritative denial evidence is represented"),
        (ApplicabilityPredicate.SUCCESSFUL_TOKEN_REPLACEMENT, replacements,
         "successful token lineage replacement is represented", "no successful token replacement is represented"),
        (ApplicabilityPredicate.CURRENT_T1_USE_BEFORE_REPLACEMENT, t1_before,
         "superseded token use precedes its replacement", "no current T1 use is represented before replacement"),
        (ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT, t2_after,
         "replacement token presentation follows replacement", "no T2 presentation is represented after replacement"),
        (ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION, identity,
         "authenticated identity matches persistent device association", "no represented successful persistent identity association"),
        (ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING, identity_binding,
         "token binding matches authenticated identity", "no resolvable token/authenticated-identity binding evidence"),
        (ApplicabilityPredicate.TOKEN_PROTECTED_CONTEXT_BINDING, context_binding,
         "token binding matches protected-session context", "no resolvable token/protected-context binding evidence"),
    )
    return tuple(
        _result(base, predicate, indices,
                positive_reason=positive, negative_reason=negative)
        for predicate, indices, positive, negative in specifications
    )


def validate_evidence_references(
    base: ValidBaseTrace, results: Iterable[ApplicabilityResult],
) -> None:
    maximum = len(base.event_records)
    for result in results:
        if result.applicable and not result.evidence_references:
            raise ValueError(f"positive predicate lacks evidence: {result.predicate_id}")
        if not result.applicable and result.evidence_references:
            raise ValueError(f"negative predicate unexpectedly references evidence: {result.predicate_id}")
        for reference in result.evidence_references:
            prefix, _, ordinal = reference.partition(":")
            if prefix != "event" or not ordinal.isdigit() or not 1 <= int(ordinal) <= maximum:
                raise ValueError(f"unresolvable applicability evidence reference: {reference}")
