"""Infrastructure contract for future hard-negative qualification.

Hard-negative qualification is privileged methodological annotation for
positively validated near-boundary normal behavior. It is independent of
semantic anomaly ground truth. HN.1 declares structures only: UNSUPPORTED is
not normal, underdetermined is not normal, and construction applied is not
qualification. No function in this module generates or qualifies behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from simulator.invariant_registry import INVARIANT_REGISTRY


HARD_NEGATIVE_SEMANTIC_CONTRACT_VERSION = "behavioral-model-v1.1-hn-contract"
HARD_NEGATIVE_FAMILY_REGISTRY_VERSION = "hn-families-v1"
HARD_NEGATIVE_QUALIFICATION_CONTRACT_VERSION = "hn-qualification-v1"
HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION = "hn-qualification-v1-schema"
HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION = "hn-qualification-v1-hn2"
HARD_NEGATIVE_GENERATOR_IMPLEMENTATION_VERSION = "hn-generation-v1"
HARD_NEGATIVE_RELEASE_VERSION = "hn-v1.0.0"


class CandidateStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"


class ConstructionStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED_NOT_APPLIED = "requested_not_applied"
    APPLIED = "applied"
    PARTIALLY_APPLIED = "partially_applied"
    CONSTRUCTION_ERROR = "construction_error"


class QualificationStatus(str, Enum):
    QUALIFIED = "qualified"
    NOT_QUALIFIED_NOT_APPLIED = "not_qualified_not_applied"
    NOT_QUALIFIED_INSUFFICIENT_EVIDENCE = "not_qualified_insufficient_evidence"
    NOT_QUALIFIED_UNDERDETERMINED = "not_qualified_underdetermined"
    NOT_QUALIFIED_RELEVANT_UNSUPPORTED = "not_qualified_relevant_unsupported"
    NOT_QUALIFIED_INVARIANT_VIOLATION = "not_qualified_invariant_violation"
    NOT_QUALIFIED_BOUNDARY_NOT_REACHED = "not_qualified_boundary_not_reached"
    QUALIFICATION_ERROR = "qualification_error"


@dataclass(frozen=True)
class QualificationCheckDefinition:
    check_id: str
    family_id: str
    variant_id: str
    validator_check_ids: tuple[str, ...]
    implementation_status: str = "frozen_hn_v1"


@dataclass(frozen=True)
class HardNegativeVariantDefinition:
    variant_id: str
    family_id: str
    invariant_ids: tuple[str, ...]
    qualification_check_ids: tuple[str, ...]
    scope: str
    status: str = "frozen_hn_v1"
    version: str = HARD_NEGATIVE_FAMILY_REGISTRY_VERSION


@dataclass(frozen=True)
class HardNegativeFamilyDefinition:
    family_id: str
    semantic_name: str
    description: str
    primary_invariant_ids: tuple[str, ...]
    approved_variant_ids: tuple[str, ...]
    qualification_check_ids: tuple[str, ...]
    scope: str
    composition_eligible: Optional[bool]
    status: str = "frozen_hn_v1"
    version: str = HARD_NEGATIVE_FAMILY_REGISTRY_VERSION


def _variant(
    variant_id: str,
    family_id: str,
    invariant_ids: tuple[str, ...],
    check_id: str,
    scope: str,
) -> HardNegativeVariantDefinition:
    return HardNegativeVariantDefinition(
        variant_id, family_id, invariant_ids, (check_id,), scope
    )


HN_VARIANTS: tuple[HardNegativeVariantDefinition, ...] = (
    _variant("fresh_nonce_cross_attempt", "HN-NONCE-FRESHNESS", ("INV-REPLAY-NONCE-CROSS-ATTEMPT",), "hn.v1.nonce.fresh_cross_attempt", "within_trace_cross_authentication_attempts"),
    _variant("legitimate_same_attempt_nonce_recurrence", "HN-NONCE-FRESHNESS", ("INV-REPLAY-NONCE-CROSS-ATTEMPT",), "hn.v1.nonce.same_attempt_recurrence", "within_single_authentication_attempt"),
    _variant("repeated_current_token_use", "HN-TOKEN-CURRENTNESS", ("INV-REPLAY-TOKEN-SUPERSEDED",), "hn.v1.token.current_repeated_use", "within_continuing_protected_session"),
    _variant("last_valid_use_before_replacement", "HN-TOKEN-CURRENTNESS", ("INV-REPLAY-TOKEN-SUPERSEDED",), "hn.v1.token.last_use_before_replacement", "within_continuing_protected_session"),
    _variant("correct_replacement_token_use", "HN-TOKEN-CURRENTNESS", ("INV-REPLAY-TOKEN-SUPERSEDED",), "hn.v1.token.replacement_token_use", "within_continuing_protected_session"),
    _variant("small_positive_causal_margin", "HN-TIME-CAUSAL-MARGIN", ("INV-TIME-OBSERVED-CAUSAL",), "hn.v1.time.positive_causal_margin", "declared_observed_causal_predecessor_relation_equality_unresolved"),
    _variant("missing_auth_context_enforced", "HN-ACCESS-ENFORCEMENT", ("INV-ACCESS-AUTH-CONTEXT",), "hn.v1.access.missing_auth_enforced", "single_access_request_reference_profile"),
    _variant("invalid_credential_enforced", "HN-ACCESS-ENFORCEMENT", ("INV-ACCESS-CREDENTIAL",), "hn.v1.access.invalid_credential_enforced", "single_access_request_reference_profile"),
    _variant("inactive_session_enforced", "HN-ACCESS-ENFORCEMENT", ("INV-ACCESS-ACTIVE-SESSION",), "hn.v1.access.inactive_session_enforced", "single_access_request_reference_profile"),
    _variant("no_grant_enforced", "HN-ACCESS-ENFORCEMENT", ("INV-ACCESS-AUTHZ-GRANT",), "hn.v1.access.no_grant_enforced", "single_access_request_reference_profile"),
    _variant("explicit_deny_enforced", "HN-ACCESS-ENFORCEMENT", ("INV-ACCESS-DENY-ENFORCED",), "hn.v1.access.explicit_deny_enforced", "single_access_request_reference_profile"),
    _variant("valid_context_legitimate_denial", "HN-AUTHZ-SEPARATION", ("INV-ACCESS-CREDENTIAL", "INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED"), "hn.v1.authz.valid_context_legitimate_denial", "single_access_request_reference_profile"),
    _variant("persistent_authenticated_identity_coherent", "HN-IDENTITY-COHERENCE", ("INV-IDENT-AUTH-ACTOR-CONSISTENCY",), "hn.v1.identity.persistent_authenticated_coherent", "persistent_identity_to_authenticated_identity"),
    _variant("token_identity_binding_coherent", "HN-BINDING-COHERENCE", ("INV-BIND-TOKEN-CONTEXT",), "hn.v1.binding.identity_coherent", "token_bound_identity_to_authenticated_identity"),
    _variant("token_protected_context_binding_coherent", "HN-BINDING-COHERENCE", ("INV-BIND-TOKEN-CONTEXT",), "hn.v1.binding.protected_context_coherent", "token_bound_context_to_protected_context_reference_profile"),
)


_VALIDATOR_CHECKS_BY_HN_CHECK: Mapping[str, tuple[str, ...]] = {
    "hn.v1.nonce.fresh_cross_attempt": ("c1.7.nonce_reuse.within_trace_attempts",),
    "hn.v1.nonce.same_attempt_recurrence": ("c1.7.nonce_reuse.within_trace_attempts",),
    "hn.v1.token.current_repeated_use": ("c2.1.replay_token.superseded_within_continuing_session",),
    "hn.v1.token.last_use_before_replacement": ("c2.1.replay_token.superseded_within_continuing_session",),
    "hn.v1.token.replacement_token_use": ("c2.1.replay_token.superseded_within_continuing_session",),
    "hn.v1.time.positive_causal_margin": ("c1.7.timestamp_inconsistency.causal_predecessor",),
    "hn.v1.access.missing_auth_enforced": ("c2.2.access.auth_context",),
    "hn.v1.access.invalid_credential_enforced": ("c2.2.access.credential",),
    "hn.v1.access.inactive_session_enforced": ("c2.2.access.active_session",),
    "hn.v1.access.no_grant_enforced": ("c2.2.access.authz_grant",),
    "hn.v1.access.explicit_deny_enforced": ("c2.2.access.deny_enforced",),
    "hn.v1.authz.valid_context_legitimate_denial": (
        "c2.2.access.credential",
        "c2.2.access.authz_grant",
        "c2.2.access.deny_enforced",
    ),
    "hn.v1.identity.persistent_authenticated_coherent": (
        "c2.3.identity.persistent_authenticated",
    ),
    "hn.v1.binding.identity_coherent": ("c2.4.binding.identity",),
    "hn.v1.binding.protected_context_coherent": (
        "c2.4.binding.protected_context",
    ),
}


HN_CHECKS: tuple[QualificationCheckDefinition, ...] = tuple(
    QualificationCheckDefinition(
        check_id,
        variant.family_id,
        variant.variant_id,
        _VALIDATOR_CHECKS_BY_HN_CHECK[check_id],
    )
    for variant in HN_VARIANTS
    for check_id in variant.qualification_check_ids
)


def _family(
    family_id: str,
    name: str,
    description: str,
    invariants: tuple[str, ...],
    scope: str,
) -> HardNegativeFamilyDefinition:
    variants = tuple(v.variant_id for v in HN_VARIANTS if v.family_id == family_id)
    checks = tuple(
        check_id
        for variant in HN_VARIANTS if variant.family_id == family_id
        for check_id in variant.qualification_check_ids
    )
    return HardNegativeFamilyDefinition(
        family_id, name, description, invariants, variants, checks, scope, None
    )


HN_FAMILIES: tuple[HardNegativeFamilyDefinition, ...] = (
    _family("HN-NONCE-FRESHNESS", "Nonce freshness boundary", "Legitimate nonce behavior near the adopted cross-attempt replay boundary.", ("INV-REPLAY-NONCE-CROSS-ATTEMPT",), "within_trace_cross_authentication_attempts"),
    _family("HN-TOKEN-CURRENTNESS", "Token currentness boundary", "Legitimate current/replacement-token behavior near supersession.", ("INV-REPLAY-TOKEN-SUPERSEDED",), "within_continuing_protected_session"),
    _family("HN-TIME-CAUSAL-MARGIN", "Observed causal-time margin", "Positive observed causal margins near the supported ordering boundary.", ("INV-TIME-OBSERVED-CAUSAL",), "declared_observed_causal_predecessor_relation_equality_unresolved"),
    _family("HN-ACCESS-ENFORCEMENT", "Access enforcement", "Correct enforcement when an applicable access prerequisite is absent.", ("INV-ACCESS-AUTH-CONTEXT", "INV-ACCESS-CREDENTIAL", "INV-ACCESS-ACTIVE-SESSION", "INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED"), "single_access_request_reference_profile"),
    _family("HN-AUTHZ-SEPARATION", "Credential/authorization separation", "Legitimate denial despite otherwise valid security context.", ("INV-ACCESS-CREDENTIAL", "INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED"), "single_access_request_reference_profile"),
    _family("HN-IDENTITY-COHERENCE", "Persistent/authenticated identity coherence", "Coherent supported identity relation without request-actor claims.", ("INV-IDENT-AUTH-ACTOR-CONSISTENCY",), "persistent_identity_to_authenticated_identity_request_actor_unsupported"),
    _family("HN-BINDING-COHERENCE", "Token binding coherence", "Coherent supported identity and reference-profile protected-context bindings.", ("INV-BIND-TOKEN-CONTEXT",), "supported_identity_and_protected_context_relations_issuer_unsupported_scope_partial"),
)

HN_FAMILY_BY_ID = {item.family_id: item for item in HN_FAMILIES}
HN_VARIANT_BY_ID = {item.variant_id: item for item in HN_VARIANTS}
HN_CHECK_BY_ID = {item.check_id: item for item in HN_CHECKS}


DEFERRED_HARD_NEGATIVE_BOUNDARIES = (
    "statistical_abnormal_failure_rate",
    "behavioral_or_statistical_abnormal_renewal",
    "cross_session_token_replay",
    "broader_nonce_reuse_domains",
    "duplicate_or_sequence_replay",
    "request_level_acting_identity_mismatch",
    "issuer_applicability",
    "universal_resource_action_binding",
    "compromised_device",
    "compromised_gateway",
    "concurrency_or_race_behavior",
    "operational_monitoring_point_effects",
    "empirical_device_behavior_deviation",
    "ip_or_location_identity_inference",
)


def validate_hard_negative_registry(
    families: Iterable[HardNegativeFamilyDefinition] = HN_FAMILIES,
    variants: Iterable[HardNegativeVariantDefinition] = HN_VARIANTS,
    checks: Iterable[QualificationCheckDefinition] = HN_CHECKS,
) -> tuple[str, ...]:
    """Return structural registry errors; never infer qualification."""
    family_items, variant_items, check_items = tuple(families), tuple(variants), tuple(checks)
    errors: list[str] = []
    family_counts = Counter(item.family_id for item in family_items)
    variant_counts = Counter(item.variant_id for item in variant_items)
    check_counts = Counter(item.check_id for item in check_items)
    errors.extend(f"duplicate family ID: {key}" for key, count in family_counts.items() if count > 1)
    errors.extend(f"duplicate variant ID: {key}" for key, count in variant_counts.items() if count > 1)
    errors.extend(f"duplicate qualification-check ID: {key}" for key, count in check_counts.items() if count > 1)
    family_map = {item.family_id: item for item in family_items}
    variant_map = {item.variant_id: item for item in variant_items}
    check_map = {item.check_id: item for item in check_items}
    for family in family_items:
        if not family.primary_invariant_ids or not family.approved_variant_ids or not family.qualification_check_ids:
            errors.append(f"family has illegal empty mapping: {family.family_id}")
        for invariant_id in family.primary_invariant_ids:
            if invariant_id not in INVARIANT_REGISTRY:
                errors.append(f"family references unknown invariant: {family.family_id}:{invariant_id}")
        for variant_id in family.approved_variant_ids:
            if variant_id not in variant_map:
                errors.append(f"family references unknown variant: {family.family_id}:{variant_id}")
            elif variant_map[variant_id].family_id != family.family_id:
                errors.append(f"invalid family/variant mapping: {family.family_id}:{variant_id}")
        for check_id in family.qualification_check_ids:
            if check_id not in check_map:
                errors.append(f"family references unknown qualification check: {family.family_id}:{check_id}")
    referenced_variants = {item for family in family_items for item in family.approved_variant_ids}
    referenced_checks = {item for family in family_items for item in family.qualification_check_ids}
    for variant in variant_items:
        family = family_map.get(variant.family_id)
        if family is None:
            errors.append(f"variant references unknown family: {variant.variant_id}:{variant.family_id}")
            continue
        if variant.variant_id not in referenced_variants:
            errors.append(f"orphan variant: {variant.variant_id}")
        if not variant.invariant_ids or not variant.qualification_check_ids:
            errors.append(f"variant has illegal empty mapping: {variant.variant_id}")
        for invariant_id in variant.invariant_ids:
            if invariant_id not in INVARIANT_REGISTRY:
                errors.append(f"variant references unknown invariant: {variant.variant_id}:{invariant_id}")
            elif invariant_id not in family.primary_invariant_ids:
                errors.append(f"variant invariant absent from family: {variant.variant_id}:{invariant_id}")
        for check_id in variant.qualification_check_ids:
            check = check_map.get(check_id)
            if check is None:
                errors.append(f"variant references unknown qualification check: {variant.variant_id}:{check_id}")
            elif check.variant_id != variant.variant_id or check.family_id != variant.family_id:
                errors.append(f"invalid variant/check mapping: {variant.variant_id}:{check_id}")
    for check in check_items:
        if check.check_id not in referenced_checks:
            errors.append(f"orphan qualification check: {check.check_id}")
        if check.family_id not in family_map:
            errors.append(f"qualification check references unknown family: {check.check_id}")
        if check.variant_id not in variant_map:
            errors.append(f"qualification check references unknown variant: {check.check_id}")
    return tuple(dict.fromkeys(errors))


@dataclass(frozen=True)
class HardNegativeQualificationRecord:
    hard_negative_candidate_id: str
    trace_id: str
    scenario_id: str
    device_id: str
    candidate_status: CandidateStatus = CandidateStatus.NOT_REQUESTED
    primary_intended_family_id: Optional[str] = None
    intended_variant_id: Optional[str] = None
    intended_boundary_invariant_ids: tuple[str, ...] = ()
    construction_status: ConstructionStatus = ConstructionStatus.NOT_REQUESTED
    construction_parameters: Mapping[str, Any] = field(default_factory=dict)
    boundary_parameters: Mapping[str, Any] = field(default_factory=dict)
    qualification_status: QualificationStatus = QualificationStatus.NOT_QUALIFIED_NOT_APPLIED
    qualified_family_ids: tuple[str, ...] = ()
    secondary_satisfied_family_ids: tuple[str, ...] = ()
    qualified_boundary_invariant_ids: tuple[str, ...] = ()
    qualification_check_ids: tuple[str, ...] = ()
    target_event_ids: tuple[str, ...] = ()
    evidence_event_ids: tuple[str, ...] = ()
    normal_side_predicates: Mapping[str, Any] = field(default_factory=dict)
    boundary_proximity_evidence: Mapping[str, Any] = field(default_factory=dict)
    relevant_validation_check_ids: tuple[str, ...] = ()
    relevant_validation_outcomes: Mapping[str, str] = field(default_factory=dict)
    disqualifying_invariant_ids: tuple[str, ...] = ()
    qualification_reason: str = "HN.1 structure only; qualification not evaluated"
    counterfactual_pair_id: Optional[str] = None
    qualifier_implementation_version: Optional[str] = None
    trace_ground_truth_status: Optional[str] = None
    trace_observable_anomaly: Optional[bool] = None

    def structural_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if len(set(self.target_event_ids)) != len(self.target_event_ids):
            errors.append("duplicate target event references")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            errors.append("duplicate evidence event references")
        if not all((self.hard_negative_candidate_id, self.trace_id, self.scenario_id, self.device_id)):
            errors.append("candidate and trace/scenario/device identifiers are required")
        family = HN_FAMILY_BY_ID.get(self.primary_intended_family_id) if self.primary_intended_family_id else None
        variant = HN_VARIANT_BY_ID.get(self.intended_variant_id) if self.intended_variant_id else None
        if self.primary_intended_family_id and family is None:
            errors.append(f"unknown family ID: {self.primary_intended_family_id}")
        if self.intended_variant_id and variant is None:
            errors.append(f"unknown variant ID: {self.intended_variant_id}")
        if variant and family is None:
            errors.append("intended variant requires a primary family")
        if family and variant and variant.family_id != family.family_id:
            errors.append("intended variant does not belong to primary family")
        for check_id in self.qualification_check_ids:
            if check_id not in HN_CHECK_BY_ID:
                errors.append(f"unknown qualification-check ID: {check_id}")
        all_invariants = set(self.intended_boundary_invariant_ids) | set(self.qualified_boundary_invariant_ids) | set(self.disqualifying_invariant_ids)
        for invariant_id in all_invariants:
            if invariant_id not in INVARIANT_REGISTRY:
                errors.append(f"unknown stable invariant ID: {invariant_id}")
        if variant and not set(self.intended_boundary_invariant_ids).issubset(variant.invariant_ids):
            errors.append("intended invariant is not mapped to intended variant")
        for family_id in set(self.qualified_family_ids) | set(self.secondary_satisfied_family_ids):
            if family_id not in HN_FAMILY_BY_ID:
                errors.append(f"unknown qualified family ID: {family_id}")
        if self.candidate_status is CandidateStatus.NOT_REQUESTED:
            if self.construction_status is not ConstructionStatus.NOT_REQUESTED:
                errors.append("non-requested candidate cannot have construction activity")
            if self.primary_intended_family_id or self.intended_variant_id:
                errors.append("non-requested candidate cannot declare HN intent")
        else:
            if not family or not variant or not self.intended_boundary_invariant_ids:
                errors.append("requested candidate requires family, variant, and boundary invariant mapping")
            if self.construction_status is ConstructionStatus.NOT_REQUESTED:
                errors.append("requested candidate requires an explicit construction outcome")
        if self.qualification_status is QualificationStatus.QUALIFIED:
            if self.candidate_status is not CandidateStatus.REQUESTED:
                errors.append("qualified candidate must have requested candidate status")
            if self.construction_status is not ConstructionStatus.APPLIED:
                errors.append("qualified candidate requires applied construction")
            if not self.qualified_family_ids:
                errors.append("qualified candidate requires at least one family ID")
            if not self.qualification_check_ids:
                errors.append("qualified candidate requires a qualification-check ID")
            if not self.qualified_boundary_invariant_ids:
                errors.append("qualified candidate requires a relevant invariant ID")
            if not self.target_event_ids or not self.evidence_event_ids:
                errors.append("qualified candidate requires target and evidence event references")
            if not self.normal_side_predicates:
                errors.append("qualified candidate requires positive normal-side predicate evidence")
            if not self.boundary_proximity_evidence:
                errors.append("qualified candidate requires boundary-proximity evidence")
            if not self.relevant_validation_check_ids or not self.relevant_validation_outcomes:
                errors.append("qualified candidate requires relevant validator evidence")
            if self.trace_ground_truth_status not in {"validated_normal", "unsupported"}:
                errors.append("qualified candidate requires non-positive, non-failed trace GT")
            if self.trace_observable_anomaly is not False:
                errors.append("qualified candidate cannot have positive or unknown anomaly GT")
            if self.qualifier_implementation_version != HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION:
                errors.append("qualified candidate requires the current qualifier version")
            if self.disqualifying_invariant_ids:
                errors.append("qualified candidate cannot carry disqualifying invariants")
            qualified_invariants = {
                invariant_id
                for family_id in self.qualified_family_ids
                if family_id in HN_FAMILY_BY_ID
                for invariant_id in HN_FAMILY_BY_ID[family_id].primary_invariant_ids
            }
            if not set(self.qualified_boundary_invariant_ids).issubset(qualified_invariants):
                errors.append("qualified invariant is not mapped to a qualified family")
            if any(
                HN_CHECK_BY_ID[check_id].family_id not in self.qualified_family_ids
                for check_id in self.qualification_check_ids
                if check_id in HN_CHECK_BY_ID
            ):
                errors.append("qualification check is not mapped to a qualified family")
            if variant and not set(variant.qualification_check_ids).issubset(
                self.qualification_check_ids
            ):
                errors.append("qualified candidate lacks its intended variant check")
            if any(
                not isinstance(item, Mapping) or item.get("satisfied") is not True
                for item in self.normal_side_predicates.values()
            ):
                errors.append("qualified candidate has an unproven normal-side predicate")
            boundary = self.boundary_proximity_evidence.get("boundary_reached")
            if not isinstance(boundary, Mapping) or boundary.get("satisfied") is not True:
                errors.append("qualified candidate has no positive boundary result")
            blocked_outcomes = {
                "PASS", "FAIL", "ERROR", "UNSUPPORTED", "CONFLICTING_RESULTS"
            }
            if any(
                outcome in blocked_outcomes
                for outcome in self.relevant_validation_outcomes.values()
            ):
                errors.append("qualified candidate carries a disqualifying validator outcome")
        if self.qualification_status is QualificationStatus.QUALIFICATION_ERROR:
            errors.append("qualification error is a hard-negative build-gate failure")
        return tuple(dict.fromkeys(errors))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_status"] = self.candidate_status.value
        value["construction_status"] = self.construction_status.value
        value["qualification_status"] = self.qualification_status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardNegativeQualificationRecord":
        data = dict(value)
        data["candidate_status"] = CandidateStatus(data["candidate_status"])
        data["construction_status"] = ConstructionStatus(data["construction_status"])
        data["qualification_status"] = QualificationStatus(data["qualification_status"])
        for name in (
            "intended_boundary_invariant_ids", "qualified_family_ids",
            "secondary_satisfied_family_ids", "qualified_boundary_invariant_ids",
            "qualification_check_ids", "target_event_ids", "evidence_event_ids",
            "relevant_validation_check_ids", "disqualifying_invariant_ids",
        ):
            data[name] = tuple(data.get(name, ()))
        return cls(**data)


@dataclass(frozen=True)
class HardNegativeStructuralGateReport:
    passed: bool
    errors: tuple[str, ...]
    records_checked: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HARD_NEGATIVE_PRIVILEGED_FIELD_NAMES = frozenset(HardNegativeQualificationRecord.__dataclass_fields__)


def validate_hard_negative_structure(
    records: Iterable[Any] = (),
    *,
    field_manifest: Optional[Mapping[str, Any]] = None,
    known_event_ids_by_trace: Optional[Mapping[str, set[str]]] = None,
    trace_anomaly_gt: Optional[Mapping[str, bool]] = None,
) -> HardNegativeStructuralGateReport:
    """Run HN.1 structural gates; semantic qualification remains unavailable."""
    raw_items = tuple(records)
    errors = list(validate_hard_negative_registry())
    items: list[HardNegativeQualificationRecord] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, HardNegativeQualificationRecord):
            errors.append(f"malformed qualification record at index {index}")
        else:
            items.append(item)
    counts = Counter(item.hard_negative_candidate_id for item in items)
    errors.extend(f"duplicate hard-negative candidate ID: {key}" for key, count in counts.items() if count > 1)
    for item in items:
        errors.extend(
            f"{item.hard_negative_candidate_id}: {error}"
            for error in item.structural_errors()
        )
        try:
            serialized = json.dumps(item.to_dict())
            restored = HardNegativeQualificationRecord.from_dict(json.loads(serialized))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{item.hard_negative_candidate_id}: qualification serialization error: {exc}")
        else:
            if restored != item:
                errors.append(f"{item.hard_negative_candidate_id}: qualification serialization mismatch")
        if known_event_ids_by_trace is not None:
            known = known_event_ids_by_trace.get(item.trace_id)
            if known is None:
                errors.append(f"{item.hard_negative_candidate_id}: qualification references unknown trace")
            else:
                unknown = (set(item.target_event_ids) | set(item.evidence_event_ids)) - known
                if unknown:
                    errors.append(
                        f"{item.hard_negative_candidate_id}: qualification references unknown events: "
                        + ", ".join(sorted(unknown))
                    )
        if (
            trace_anomaly_gt is not None
            and item.qualification_status is QualificationStatus.QUALIFIED
            and trace_anomaly_gt.get(item.trace_id) is not False
        ):
            errors.append(f"{item.hard_negative_candidate_id}: qualified record conflicts with trace anomaly GT")
    if field_manifest:
        forbidden_fragments = (
            "hard_negative", "candidate_id", "family_id", "variant_id",
            "qualification", "boundary_invariant", "boundary_parameter",
            "counterfactual",
        )
        leaked = {
            item.get("field_name")
            for item in field_manifest.get("fields", ())
            if item.get("export_view") == "detector_observations"
            and (
                item.get("field_name") in HARD_NEGATIVE_PRIVILEGED_FIELD_NAMES
                or any(fragment in str(item.get("field_name", "")) for fragment in forbidden_fragments)
            )
        }
        errors.extend(f"detector manifest exposes privileged HN field: {name}" for name in sorted(leaked))
    unique = tuple(dict.fromkeys(errors))
    return HardNegativeStructuralGateReport(not unique, unique, len(raw_items))


def construction_yield(applied: int, intended: int) -> Optional[float]:
    """Methodology diagnostic; return None when no candidates were intended."""
    return applied / intended if intended else None


def qualification_yield(qualified: int, applied: int) -> Optional[float]:
    """Methodology diagnostic; return None when no constructions were applied."""
    return qualified / applied if applied else None


@dataclass(frozen=True)
class HardNegativeRunSummary:
    total_traces: int
    ordinary_normal_traces: int
    hn_intended_traces: int
    hn_construction_applied: int
    hn_qualified: int
    hn_not_qualified: int
    qualification_status_counts: Mapping[str, int]
    qualification_status_by_family: Mapping[str, Mapping[str, int]]
    qualification_status_by_variant: Mapping[str, Mapping[str, int]]
    qualification_status_by_check: Mapping[str, Mapping[str, int]]
    intended_by_family: Mapping[str, int]
    applied_by_family: Mapping[str, int]
    qualified_by_family: Mapping[str, int]
    rejected_by_family: Mapping[str, int]
    intended_by_variant: Mapping[str, int]
    applied_by_variant: Mapping[str, int]
    qualified_by_variant: Mapping[str, int]
    construction_yield_by_family: Mapping[str, Optional[float]]
    qualification_yield_by_family: Mapping[str, Optional[float]]
    candidate_anomaly_collisions: int
    relevant_unsupported_cases: int
    underdetermined_cases: int
    qualification_errors: int
    primary_family_counts: Mapping[str, int]
    multi_family_qualified_traces: int
    build_gate_result: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_records(
        cls,
        *,
        total_traces: int,
        ordinary_normal_traces: int,
        records: Iterable[HardNegativeQualificationRecord] = (),
        candidate_anomaly_collisions: int = 0,
        build_gate_result: Optional[bool] = None,
    ) -> "HardNegativeRunSummary":
        items = tuple(records)
        gate = validate_hard_negative_structure(items)
        intended = Counter(item.primary_intended_family_id for item in items if item.candidate_status is CandidateStatus.REQUESTED and item.primary_intended_family_id)
        applied = Counter(item.primary_intended_family_id for item in items if item.construction_status is ConstructionStatus.APPLIED and item.primary_intended_family_id)
        qualified = Counter(family_id for item in items if item.qualification_status is QualificationStatus.QUALIFIED for family_id in item.qualified_family_ids)
        rejected = Counter(item.primary_intended_family_id for item in items if item.primary_intended_family_id and item.qualification_status is not QualificationStatus.QUALIFIED)
        intended_variants = Counter(item.intended_variant_id for item in items if item.candidate_status is CandidateStatus.REQUESTED and item.intended_variant_id)
        applied_variants = Counter(item.intended_variant_id for item in items if item.construction_status is ConstructionStatus.APPLIED and item.intended_variant_id)
        qualified_variants = Counter(item.intended_variant_id for item in items if item.qualification_status is QualificationStatus.QUALIFIED and item.intended_variant_id)
        status_counts = Counter(item.qualification_status.value for item in items)
        status_by_family: dict[str, Counter[str]] = {
            key: Counter() for key in HN_FAMILY_BY_ID
        }
        status_by_variant: dict[str, Counter[str]] = {
            key: Counter() for key in HN_VARIANT_BY_ID
        }
        status_by_check: dict[str, Counter[str]] = {
            key: Counter() for key in HN_CHECK_BY_ID
        }
        for item in items:
            status = item.qualification_status.value
            if item.primary_intended_family_id in status_by_family:
                status_by_family[item.primary_intended_family_id][status] += 1
            if item.intended_variant_id in status_by_variant:
                status_by_variant[item.intended_variant_id][status] += 1
            for check_id in item.qualification_check_ids:
                if check_id in status_by_check:
                    status_by_check[check_id][status] += 1
        return cls(
            total_traces=total_traces,
            ordinary_normal_traces=ordinary_normal_traces,
            hn_intended_traces=sum(intended.values()),
            hn_construction_applied=sum(applied.values()),
            hn_qualified=sum(item.qualification_status is QualificationStatus.QUALIFIED for item in items),
            hn_not_qualified=sum(item.qualification_status is not QualificationStatus.QUALIFIED for item in items),
            qualification_status_counts=dict(status_counts),
            qualification_status_by_family={
                key: dict(value) for key, value in status_by_family.items()
            },
            qualification_status_by_variant={
                key: dict(value) for key, value in status_by_variant.items()
            },
            qualification_status_by_check={
                key: dict(value) for key, value in status_by_check.items()
            },
            intended_by_family=dict(intended), applied_by_family=dict(applied),
            qualified_by_family=dict(qualified), rejected_by_family=dict(rejected),
            intended_by_variant=dict(intended_variants), applied_by_variant=dict(applied_variants),
            qualified_by_variant=dict(qualified_variants),
            construction_yield_by_family={key: construction_yield(applied[key], intended[key]) for key in HN_FAMILY_BY_ID},
            qualification_yield_by_family={key: qualification_yield(qualified[key], applied[key]) for key in HN_FAMILY_BY_ID},
            candidate_anomaly_collisions=candidate_anomaly_collisions,
            relevant_unsupported_cases=status_counts[QualificationStatus.NOT_QUALIFIED_RELEVANT_UNSUPPORTED.value],
            underdetermined_cases=status_counts[QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED.value],
            qualification_errors=status_counts[QualificationStatus.QUALIFICATION_ERROR.value],
            primary_family_counts=dict(intended),
            multi_family_qualified_traces=len({item.trace_id for item in items if item.qualification_status is QualificationStatus.QUALIFIED and len(item.qualified_family_ids) > 1}),
            build_gate_result=gate.passed if build_gate_result is None else build_gate_result,
        )


def hard_negative_capability_report() -> dict[str, Any]:
    """Machine-readable frozen HN implementation capability declaration."""
    return {
        "hard_negative_release_version": HARD_NEGATIVE_RELEASE_VERSION,
        "hard_negative_semantic_contract_version": HARD_NEGATIVE_SEMANTIC_CONTRACT_VERSION,
        "hard_negative_family_registry_version": HARD_NEGATIVE_FAMILY_REGISTRY_VERSION,
        "hard_negative_qualification_contract_version": HARD_NEGATIVE_QUALIFICATION_CONTRACT_VERSION,
        "hard_negative_qualification_schema_version": HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION,
        "hard_negative_qualifier_implementation_version": HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        "families": [asdict(item) for item in HN_FAMILIES],
        "variants": [asdict(item) for item in HN_VARIANTS],
        "qualification_checks": [asdict(item) for item in HN_CHECKS],
        "supported_family_ids": sorted(HN_FAMILY_BY_ID),
        "supported_variant_ids": sorted(HN_VARIANT_BY_ID),
        "explicitly_generatable_variant_ids": sorted(HN_VARIANT_BY_ID),
        "generation_blocked_variant_ids": [],
        "qualification_check_ids": sorted(HN_CHECK_BY_ID),
        "registered_family_count": len(HN_FAMILIES),
        "registered_variant_count": len(HN_VARIANTS),
        "deterministic_qualification_check_count": len(HN_CHECKS),
        "explicit_generation_mechanism_count": len(HN_VARIANTS),
        "deferred_or_unsupported_boundaries": list(DEFERRED_HARD_NEGATIVE_BOUNDARIES),
        "generation_available": True,
        "explicit_single_family_generation_available": True,
        "random_sampling_available": False,
        "multi_boundary_generation_available": False,
        "generator_implementation_version": HARD_NEGATIVE_GENERATOR_IMPLEMENTATION_VERSION,
        "semantic_qualification_available": True,
        "semantic_validation_available": True,
        "privileged_export_available": True,
        "scientific_sampling_prevalence_frozen": False,
        "controlled_device_profile_variability_available": False,
        "empirical_detector_difficulty_established": False,
        "operational_realism_established": False,
        "real_world_hn_prevalence_established": False,
        "qualification_requires_explicit_candidate": True,
        "ordinary_trace_default": "no qualification record",
        "qualification_principle": "positive_evidence_required_never_inferred_from_normal_gt_or_validator_silence",
        "registry_integrity_errors": list(validate_hard_negative_registry()),
    }
