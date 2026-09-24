"""Behavioral-model v1.1 stable invariant registry.

The identifiers in this module are owned by the authoritative behavioral
model.  Implementation status describes the current C2 development checkout;
it does not redefine an invariant, create an anomaly label, or make a field
detector-visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping


INVARIANT_REGISTRY_VERSION = "behavioral-model-v1.1-sync-v1.1.0-implementation-map"


IDENTITY_RELATION_CAPABILITIES = {
    "persistent_identity_to_authenticated_identity": {
        "status": "exact",
        "check": "c2.3.identity.persistent_authenticated",
        "scope": "single-device reference profile authentication context",
        "minimum_evidence": (
            "persistent device identity, successful authentication event, "
            "authenticated identity, authentication-attempt identifier, and "
            "reconstructable authentication-context-to-device association"
        ),
    },
    "authenticated_identity_to_request_acting_identity": {
        "status": "unsupported_by_current_evidence",
        "check": None,
        "scope": "protected request",
        "minimum_evidence": (
            "independent request-level acting identity distinct from authenticated "
            "context, token binding, persistent device identity, and network address"
        ),
        "reason": (
            "AccessRequestContext.authenticated_identity is copied from the "
            "authentication context and is not an independent ActingIdentity(R)."
        ),
    },
}


TOKEN_BINDING_RELATION_CAPABILITIES = {
    "token_bound_identity_to_authenticated_identity": {
        "status": "exact",
        "check": "c2.4.binding.identity",
        "scope": "token presentation/use in an authenticated context",
        "minimum_evidence": (
            "token ID, issuance, independently stored token-bound identity, "
            "presentation/use, authenticated identity, and reliable context linkage"
        ),
    },
    "token_bound_protected_context_to_active_context": {
        "status": "exact",
        "check": "c2.4.binding.protected_context",
        "scope": "reference-profile token lineage after protected-context binding is defined",
        "minimum_evidence": (
            "token ID, established token lineage, bound protected-session ID, "
            "and the active protected-session ID in which the token is used"
        ),
        "reason": (
            "This is limited to the existing TokenLineageContext protected-session "
            "association and is not a universal token/session-binding requirement."
        ),
    },
    "issuer_applicability": {
        "status": "unsupported_by_current_evidence",
        "check": None,
        "scope": "token issuance and validation",
        "minimum_evidence": "independently represented token issuer and applicable trusted-issuer relation",
        "reason": "The synchronized token evidence does not carry an independent issuer relationship.",
    },
    "resource_action_scope_applicability": {
        "status": "partial",
        "check": None,
        "scope": "reference authorization policy",
        "minimum_evidence": "token scope plus requested resource/action and declared applicability semantics",
        "reason": (
            "Scope is represented and checked by existing credential/access logic, "
            "but C2.4 does not redefine it as universal identity/context binding."
        ),
    },
}


class ImplementationStatus(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    ABSENT = "absent"
    UNSUPPORTED_BY_CURRENT_EVIDENCE = "unsupported_by_current_evidence"


@dataclass(frozen=True)
class InvariantDefinition:
    invariant_id: str
    short_name: str
    semantic_family: str
    scope: str
    decidability: str
    implementation_status: ImplementationStatus
    version_introduced: str
    minimum_evidence: str
    current_checks: tuple[str, ...] = ()
    implementation_delta: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["implementation_status"] = self.implementation_status.value
        return value


def _definition(
    invariant_id: str,
    short_name: str,
    family: str,
    scope: str,
    status: ImplementationStatus,
    evidence: str,
    *,
    checks: tuple[str, ...] = (),
    delta: str = "",
    notes: str = "",
) -> InvariantDefinition:
    return InvariantDefinition(
        invariant_id=invariant_id,
        short_name=short_name,
        semantic_family=family,
        scope=scope,
        decidability="rule-decidable when the declared minimum evidence is present",
        implementation_status=status,
        version_introduced="behavioral-model-v1.1",
        minimum_evidence=evidence,
        current_checks=checks,
        implementation_delta=delta,
        notes=notes,
    )


INVARIANT_REGISTRY: Mapping[str, InvariantDefinition] = {
    item.invariant_id: item
    for item in (
        _definition(
            "INV-AUTH-FAIL-NO-CONTEXT",
            "Authentication failure creates no authenticated context",
            "authentication",
            "authentication attempt",
            ImplementationStatus.PARTIAL,
            "authentication attempt, failure result, resulting authenticated-context evidence",
            checks=("c1.7.state_result.supported_subset",),
            delta="Emit an atomic validator result that distinguishes newly created context from a previously valid continuing context.",
        ),
        _definition(
            "INV-AUTH-CONTEXT-REQUIRES-SUCCESS",
            "Authenticated context requires successful authentication",
            "authentication",
            "authentication attempt and protected session",
            ImplementationStatus.PARTIAL,
            "successful authentication event and the context it establishes",
            checks=("c1.7.state_result.supported_subset",),
            delta="Separate this rule from the composite state/result validator and retain explicit supporting-success linkage.",
        ),
        _definition(
            "INV-LIFE-TOKEN-AFTER-AUTH",
            "Token establishment follows successful authentication",
            "lifecycle/causal",
            "authentication attempt",
            ImplementationStatus.PARTIAL,
            "token issuance and successful authentication linked to the same applicable attempt",
            checks=("c1.7.lifecycle_causal.supported_subset", "c1.7.credential_token.supported_subset"),
            delta="Expose an atomic result for the already represented same-attempt prerequisite.",
        ),
        _definition(
            "INV-LIFE-TOKEN-USE-AFTER-ESTABLISH",
            "Token use follows token establishment",
            "lifecycle/causal",
            "token context within a trace/session",
            ImplementationStatus.PARTIAL,
            "token identity on establishment, presentation, validation, and applicable use",
            checks=("c1.7.lifecycle_causal.supported_subset", "c1.7.credential_token.supported_subset"),
            delta="Define and validate each applicable token-use operation atomically rather than through the composite lifecycle check.",
        ),
        _definition(
            "INV-LIFE-SESSION-ESTABLISH-PRECONDITIONS",
            "Protected-session establishment requires security prerequisites",
            "lifecycle/causal",
            "protected-session establishment",
            ImplementationStatus.PARTIAL,
            "successful authentication, applicable token validation, and session-open evidence",
            checks=("c1.7.lifecycle_causal.supported_subset", "c1.7.state_result.supported_subset"),
            delta="Emit one atomic result for the complete v1.1 session-establishment prerequisite set.",
        ),
        _definition(
            "INV-TIME-CAUSAL-ORDER",
            "Authoritative semantic causal order is coherent",
            "temporal",
            "trace and retained per-device chronology",
            ImplementationStatus.PARTIAL,
            "authoritative semantic timestamps and causal predecessor relationships",
            checks=("c1.7.temporal.supported_subset",),
            delta="Represent causal relationships explicitly where total sequential order is not sufficient.",
        ),
        _definition(
            "INV-TIME-TOKEN-VALIDITY",
            "Token use respects its validity interval",
            "temporal",
            "token context",
            ImplementationStatus.PARTIAL,
            "issuance time, presentation/validation time, and explicit validity interval",
            checks=("c1.7.temporal.supported_subset",),
            delta="Emit an atomic validity result and reconcile the exact validity-boundary convention if 02 specifies it.",
        ),
        _definition(
            "INV-TIME-OBSERVED-CAUSAL",
            "Observed time respects declared causal ordering",
            "temporal",
            "within-trace causal predecessor",
            ImplementationStatus.EXACT,
            "linked predecessor/target events, semantic timestamps, observed timestamps, and targeted relation",
            checks=("c1.7.timestamp_inconsistency.causal_predecessor",),
            notes="The authoritative semantic clock remains coherent when this observed-evidence invariant is violated.",
        ),
        _definition(
            "INV-SESS-ENROLLMENT-PERSISTS",
            "Enrollment persists across ordinary session termination",
            "session",
            "cross-trace device history",
            ImplementationStatus.EXACT,
            "stable device identity and enrollment state before/after ordinary session termination",
            checks=("c1.7.persistence.cross_trace_enrollment",),
        ),
        _definition(
            "INV-SESS-ACTIVE-COHERENCE",
            "Active session context is internally coherent",
            "session",
            "protected session",
            ImplementationStatus.PARTIAL,
            "session identifier, establishment evidence, active flags, and termination evidence",
            checks=("c1.7.state_result.supported_subset", "c1.7.session_continuity.supported_subset"),
            delta="Split active-session establishment, ownership, and termination coherence into an atomic validator result.",
        ),
        _definition(
            "INV-SESS-RENEWAL-CONTINUITY",
            "Renewal remains linked to the continuing protected session",
            "session",
            "renewal attempt and protected session",
            ImplementationStatus.PARTIAL,
            "renewal attempt ID, refreshed protected-session ID, old/new token context, and renewal result",
            checks=("c1.7.session_continuity.supported_subset", "c1.7.temporal.supported_subset"),
            delta="Emit an atomic continuity result and preserve the unresolved re-authorization decision separately.",
        ),
        _definition(
            "INV-REPLAY-NONCE-CROSS-ATTEMPT",
            "Nonce is not reused across authentication attempts in the declared domain",
            "replay",
            "within one trace across authentication attempts",
            ImplementationStatus.EXACT,
            "same nonce material, distinct attempt IDs, source/reuse events, and injection lineage",
            checks=("c1.7.nonce_reuse.within_trace_attempts",),
        ),
        _definition(
            "INV-REPLAY-TOKEN-SUPERSEDED",
            "Superseded token is not reused where supersession forbids use",
            "replay",
            "token lineage and declared supersession domain",
            ImplementationStatus.EXACT,
            "token identity, predecessor/successor relationship, supersession event, reuse event, and declared forbidden-use scope",
            checks=("c2.1.replay_token.superseded_within_continuing_session",),
            notes="C2.1 support is limited to actual reuse after successful replacement in one continuing protected session; historical replay_token remains distinct.",
        ),
        _definition(
            "INV-ACCESS-AUTH-CONTEXT",
            "Protected access requires authenticated context",
            "access",
            "access request/decision",
            ImplementationStatus.EXACT,
            "access request and applicable authenticated identity/context",
            checks=("c2.2.access.auth_context",),
        ),
        _definition(
            "INV-ACCESS-CREDENTIAL",
            "Protected access requires applicable valid credential context",
            "access",
            "access request/decision",
            ImplementationStatus.EXACT,
            "access request, token identity, validation result, active token context, and applicable scope",
            checks=("c2.2.access.credential",),
        ),
        _definition(
            "INV-ACCESS-ACTIVE-SESSION",
            "Protected access requires active protected session",
            "access",
            "access request/decision",
            ImplementationStatus.EXACT,
            "access request and matching active protected-session context",
            checks=("c2.2.access.active_session",),
        ),
        _definition(
            "INV-ACCESS-AUTHZ-GRANT",
            "Successful protected operation requires authorization grant",
            "access",
            "access request/decision/outcome",
            ImplementationStatus.EXACT,
            "request ID, authorization decision, and resource-operation outcome",
            checks=("c2.2.access.authz_grant",),
        ),
        _definition(
            "INV-ACCESS-DENY-ENFORCED",
            "Authorization denial prevents successful protected operation",
            "access",
            "access request/decision/outcome",
            ImplementationStatus.EXACT,
            "same request ID, denial decision, and denied/not-executed operation outcome",
            checks=("c2.2.access.deny_enforced",),
        ),
        _definition(
            "INV-IDENT-AUTH-ACTOR-CONSISTENCY",
            "Authenticated and acting identity are consistent",
            "identity/binding",
            "authentication and protected operation context",
            ImplementationStatus.PARTIAL,
            "independent identity values plus reliable context/device association for each scoped comparison",
            checks=("c2.3.identity.persistent_authenticated",),
            delta="Represent an independent request-level acting identity before implementing the authenticated-to-request-actor comparison.",
            notes=(
                "Persistent-to-authenticated relation is supported/exact in the "
                "single-device reference profile; authenticated-to-request-actor "
                "relation remains unsupported because no independent request actor exists."
            ),
        ),
        _definition(
            "INV-BIND-TOKEN-CONTEXT",
            "Token remains bound to its applicable identity and context",
            "identity/binding",
            "token, authentication, session, and access context",
            ImplementationStatus.PARTIAL,
            "independent token-bound identity/context plus authenticated and protected-context use evidence",
            checks=("c2.4.binding.identity", "c2.4.binding.protected_context"),
            delta="Add issuer applicability only if an authoritative issuer relation is later represented; retain scope applicability separately.",
            notes=(
                "Identity binding and the existing reference-profile protected-context "
                "lineage relation are exact. Issuer applicability is unsupported and "
                "scope applicability remains partial; no universal credential architecture is claimed."
            ),
        ),
    )
}


EXACT_VALIDATOR_CHECK_TO_INVARIANT: Mapping[str, str] = {
    "c1.7.timestamp_inconsistency.causal_predecessor": "INV-TIME-OBSERVED-CAUSAL",
    "c1.7.persistence.cross_trace_enrollment": "INV-SESS-ENROLLMENT-PERSISTS",
    "c1.7.nonce_reuse.within_trace_attempts": "INV-REPLAY-NONCE-CROSS-ATTEMPT",
    "c2.1.replay_token.superseded_within_continuing_session": "INV-REPLAY-TOKEN-SUPERSEDED",
    "c2.2.access.auth_context": "INV-ACCESS-AUTH-CONTEXT",
    "c2.2.access.credential": "INV-ACCESS-CREDENTIAL",
    "c2.2.access.active_session": "INV-ACCESS-ACTIVE-SESSION",
    "c2.2.access.authz_grant": "INV-ACCESS-AUTHZ-GRANT",
    "c2.2.access.deny_enforced": "INV-ACCESS-DENY-ENFORCED",
    "c2.3.identity.persistent_authenticated": "INV-IDENT-AUTH-ACTOR-CONSISTENCY",
    "c2.4.binding.identity": "INV-BIND-TOKEN-CONTEXT",
    "c2.4.binding.protected_context": "INV-BIND-TOKEN-CONTEXT",
}


EXACT_VARIANT_TO_INVARIANTS: Mapping[str, tuple[str, ...]] = {
    "nonce_reuse": ("INV-REPLAY-NONCE-CROSS-ATTEMPT",),
    "timestamp_inconsistency": ("INV-TIME-OBSERVED-CAUSAL",),
    "token_replay_after_replacement": ("INV-REPLAY-TOKEN-SUPERSEDED",),
    "access_success_without_auth_context": ("INV-ACCESS-AUTH-CONTEXT",),
    "access_success_without_valid_credential": ("INV-ACCESS-CREDENTIAL",),
    "access_success_without_active_session": ("INV-ACCESS-ACTIVE-SESSION",),
    "access_success_without_applicable_grant": ("INV-ACCESS-AUTHZ-GRANT",),
    "access_success_despite_explicit_deny": (
        "INV-ACCESS-AUTHZ-GRANT",
        "INV-ACCESS-DENY-ENFORCED",
    ),
    "access_success_without_security_prerequisites": (
        "INV-ACCESS-AUTH-CONTEXT",
        "INV-ACCESS-CREDENTIAL",
        "INV-ACCESS-ACTIVE-SESSION",
    ),
    "authenticated_identity_mismatch_persistent_device": (
        "INV-IDENT-AUTH-ACTOR-CONSISTENCY",
    ),
    "token_bound_identity_mismatch_authenticated_context": (
        "INV-BIND-TOKEN-CONTEXT",
    ),
}


INVARIANT_IMPLEMENTATION_VERSIONS: Mapping[str, str] = {
    **{
        invariant_id: "c1.7"
        for invariant_id in (
            "INV-AUTH-FAIL-NO-CONTEXT",
            "INV-AUTH-CONTEXT-REQUIRES-SUCCESS",
            "INV-LIFE-TOKEN-AFTER-AUTH",
            "INV-LIFE-TOKEN-USE-AFTER-ESTABLISH",
            "INV-LIFE-SESSION-ESTABLISH-PRECONDITIONS",
            "INV-TIME-CAUSAL-ORDER",
            "INV-TIME-TOKEN-VALIDITY",
            "INV-TIME-OBSERVED-CAUSAL",
            "INV-SESS-ENROLLMENT-PERSISTS",
            "INV-SESS-ACTIVE-COHERENCE",
            "INV-SESS-RENEWAL-CONTINUITY",
            "INV-REPLAY-NONCE-CROSS-ATTEMPT",
        )
    },
    "INV-REPLAY-TOKEN-SUPERSEDED": "c2.1",
    "INV-ACCESS-AUTH-CONTEXT": "c2.2",
    "INV-ACCESS-CREDENTIAL": "c2.2",
    "INV-ACCESS-ACTIVE-SESSION": "c2.2",
    "INV-ACCESS-AUTHZ-GRANT": "c2.2",
    "INV-ACCESS-DENY-ENFORCED": "c2.2",
    "INV-IDENT-AUTH-ACTOR-CONSISTENCY": "c2.3",
    "INV-BIND-TOKEN-CONTEXT": "c2.4",
}


def invariant_registry_report() -> dict:
    variants_by_invariant = {
        invariant_id: sorted(
            variant
            for variant, invariant_ids in EXACT_VARIANT_TO_INVARIANTS.items()
            if invariant_id in invariant_ids
        )
        for invariant_id in INVARIANT_REGISTRY
    }
    invariants = []
    for key in sorted(INVARIANT_REGISTRY):
        item = INVARIANT_REGISTRY[key].to_dict()
        item["implementation_version"] = INVARIANT_IMPLEMENTATION_VERSIONS[key]
        item["synchronized_anomaly_variants"] = variants_by_invariant[key]
        item["scope_caveat"] = item["notes"] or item["implementation_delta"] or (
            "No caveat beyond the declared scope and minimum-evidence requirement."
        )
        if key == "INV-IDENT-AUTH-ACTOR-CONSISTENCY":
            item["relation_capabilities"] = IDENTITY_RELATION_CAPABILITIES
        elif key == "INV-BIND-TOKEN-CONTEXT":
            item["relation_capabilities"] = TOKEN_BINDING_RELATION_CAPABILITIES
        else:
            item["relation_capabilities"] = {}
        invariants.append(item)
    return {
        "registry_version": INVARIANT_REGISTRY_VERSION,
        "behavior_model_version": "behavioral-model-v1.1",
        "identifier_convention": "INV-<DOMAIN>-<CONCEPT>",
        "canonical_identifiers": sorted(INVARIANT_REGISTRY),
        "invariants": invariants,
        "exact_validator_check_mappings": dict(EXACT_VALIDATOR_CHECK_TO_INVARIANT),
        "exact_variant_mappings": {
            name: list(ids) for name, ids in EXACT_VARIANT_TO_INVARIANTS.items()
        },
        "identity_relation_capabilities": IDENTITY_RELATION_CAPABILITIES,
        "token_binding_relation_capabilities": TOKEN_BINDING_RELATION_CAPABILITIES,
        "legacy_scenario_names_are_invariant_ids": False,
    }
