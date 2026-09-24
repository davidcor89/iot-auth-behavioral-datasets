"""C1.5 synchronized anomaly capability and injection provenance.

Names and invariant families here are implementation-facing contract mappings,
not newly assigned canonical identifiers or a final anomaly taxonomy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from simulator.invariant_registry import EXACT_VARIANT_TO_INVARIANTS, INVARIANT_REGISTRY

SYNCHRONIZED_GENERATION_PROFILE = "sync-v1.1.0"
HISTORICAL_GENERATION_PROFILE = "historical-27135bf"

SYNCHRONIZED_ACCESS_VARIANTS = (
    "access_success_without_auth_context",
    "access_success_without_valid_credential",
    "access_success_without_active_session",
    "access_success_without_applicable_grant",
    "access_success_despite_explicit_deny",
    "access_success_without_security_prerequisites",
)

SYNCHRONIZED_IDENTITY_VARIANTS = (
    "authenticated_identity_mismatch_persistent_device",
)

SYNCHRONIZED_TOKEN_BINDING_VARIANTS = (
    "token_bound_identity_mismatch_authenticated_context",
)


class VariantCapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    QUARANTINED_HISTORICAL = "quarantined_historical_variant"
    COMPOSITE_SCENARIO_PROVENANCE = "composite_scenario_provenance_only"
    UNSUPPORTED_HISTORY_DEPENDENT = "unsupported_history_dependent_behavior"


@dataclass(frozen=True)
class VariantCapability:
    historical_name: str
    status: VariantCapabilityStatus
    semantic_concept: str
    invariant_families: Tuple[str, ...] = ()
    invariant_ids: Tuple[str, ...] = ()
    history_scope: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["status"] = self.status.value
        record["canonical_identifier"] = None
        record["required_evidence_by_invariant"] = {
            invariant_id: INVARIANT_REGISTRY[invariant_id].minimum_evidence
            for invariant_id in self.invariant_ids
        }
        record["validator_checks_by_invariant"] = {
            invariant_id: list(INVARIANT_REGISTRY[invariant_id].current_checks)
            for invariant_id in self.invariant_ids
        }
        return record


VARIANT_CAPABILITIES = {
    "nonce_reuse": VariantCapability(
        "nonce_reuse",
        VariantCapabilityStatus.SUPPORTED,
        "reuse of challenge nonce material across authentication attempts",
        ("temporal/freshness consistency", "replay/single-use"),
        EXACT_VARIANT_TO_INVARIANTS["nonce_reuse"],
        "within_trace_across_authentication_attempts",
        "The v1 contract explicitly requires within-trace attempt history.",
    ),
    "timestamp_inconsistency": VariantCapability(
        "timestamp_inconsistency",
        VariantCapabilityStatus.SUPPORTED,
        "observed timestamp violates causal predecessor ordering",
        ("temporal/freshness consistency",),
        EXACT_VARIANT_TO_INVARIANTS["timestamp_inconsistency"],
        "within_trace_causal_predecessor",
        "Semantic time remains monotonic; only declared/observed evidence changes.",
    ),
    "replay_token": VariantCapability(
        "replay_token",
        VariantCapabilityStatus.QUARANTINED_HISTORICAL,
        "token replay/reuse",
        ("replay/single-use", "credential/token binding"),
        (),
        None,
        "No approved token reuse-forbidden domain exists; token age is insufficient.",
    ),
    "token_replay_after_replacement": VariantCapability(
        "replay_token",
        VariantCapabilityStatus.SUPPORTED,
        "actual reuse of a token superseded by successful renewal replacement",
        ("replay/single-use",),
        EXACT_VARIANT_TO_INVARIANTS["token_replay_after_replacement"],
        "within_continuing_protected_session_after_successful_replacement",
        "Reference-scoped C2.1 variant; it does not imply general or cross-session replay semantics.",
    ),
    "duplicate_sequence": VariantCapability(
        "duplicate_sequence",
        VariantCapabilityStatus.QUARANTINED_HISTORICAL,
        "sequence duplication",
        ("lifecycle/causal ordering", "replay/single-use"),
        (),
        None,
        "Historical injection does not duplicate a meaningful subsequence.",
    ),
    "impersonation": VariantCapability(
        "impersonation",
        VariantCapabilityStatus.COMPOSITE_SCENARIO_PROVENANCE,
        "identity/credential/authentication scenario bundle",
        ("identity consistency", "credential/token binding"),
        (),
        None,
        "Final decomposition requires authoritative identity taxonomy.",
    ),
    "identity_token_mismatch": VariantCapability(
        "identity_token_mismatch",
        VariantCapabilityStatus.COMPOSITE_SCENARIO_PROVENANCE,
        "identity and token-binding scenario bundle",
        ("identity consistency", "credential/token binding"),
        (),
        None,
        "The historical atomic label is not retained as synchronized truth.",
    ),
    "token_bound_identity_mismatch_authenticated_context": VariantCapability(
        "not_applicable_synchronized_token_binding_variant",
        VariantCapabilityStatus.SUPPORTED,
        "token-bound identity differs from the authenticated identity of its presentation context",
        ("credential/token binding",),
        EXACT_VARIANT_TO_INVARIANTS[
            "token_bound_identity_mismatch_authenticated_context"
        ],
        "token_presentation_in_authenticated_context",
        (
            "C2.4 atomic identity-binding variant; historical "
            "identity_token_mismatch remains composite provenance."
        ),
    ),
    "authenticated_identity_mismatch_persistent_device": VariantCapability(
        "not_applicable_synchronized_identity_variant",
        VariantCapabilityStatus.SUPPORTED,
        "authenticated identity differs from persistent device identity in a reliably associated authentication context",
        ("identity consistency",),
        EXACT_VARIANT_TO_INVARIANTS[
            "authenticated_identity_mismatch_persistent_device"
        ],
        "single_device_authentication_context",
        (
            "C2.3 atomic persistent-to-authenticated identity relation; historical "
            "impersonation and identity_token_mismatch remain composite provenance."
        ),
    ),
    "access_without_auth": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.COMPOSITE_SCENARIO_PROVENANCE,
        "protected-access prerequisite scenario bundle",
        ("authorization/access preconditions",),
        (),
        None,
        "Exact violated prerequisite is not safely encoded by the historical path.",
    ),
    "access_success_without_auth_context": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation without applicable authenticated context",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_without_auth_context"],
        "single_access_request_reference_profile",
        "C2.2 atomic access variant; historical access_without_auth remains composite.",
    ),
    "access_success_without_valid_credential": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation without required validated credential context",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_without_valid_credential"],
        "single_access_request_reference_profile",
    ),
    "access_success_without_active_session": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation without required active protected session",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_without_active_session"],
        "single_access_request_reference_profile",
        "The active-session prerequisite is reference-profile scoped.",
    ),
    "access_success_without_applicable_grant": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation after complete authorization evaluation found no applicable grant",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_without_applicable_grant"],
        "single_access_request_reference_profile",
    ),
    "access_success_despite_explicit_deny": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation despite authoritative explicit denial",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_despite_explicit_deny"],
        "single_access_request_reference_profile",
    ),
    "access_success_without_security_prerequisites": VariantCapability(
        "access_without_auth",
        VariantCapabilityStatus.SUPPORTED,
        "successful protected operation without authentication, credential, or active-session prerequisites",
        ("authorization/access preconditions",),
        EXACT_VARIANT_TO_INVARIANTS["access_success_without_security_prerequisites"],
        "single_access_request_reference_profile",
        "Explicit C2.2 multi-invariant validation case.",
    ),
    "abnormal_failure_rate": VariantCapability(
        "abnormal_failure_rate",
        VariantCapabilityStatus.UNSUPPORTED_HISTORY_DEPENDENT,
        "failure frequency/history behavior",
        ("frequency/history behavior",),
        (),
        None,
        "Requires an approved history window and policy/baseline.",
    ),
    "abnormal_renewal": VariantCapability(
        "abnormal_renewal",
        VariantCapabilityStatus.UNSUPPORTED_HISTORY_DEPENDENT,
        "behaviorally unusual renewal frequency/timing",
        ("frequency/history behavior",),
        (),
        None,
        "Deterministic renewal invalidity must later be separated from behavior.",
    ),
}


def supported_synchronized_variants() -> Tuple[str, ...]:
    return tuple(
        name
        for name, capability in VARIANT_CAPABILITIES.items()
        if capability.status is VariantCapabilityStatus.SUPPORTED
    )


def synchronized_capability_report() -> dict[str, Any]:
    historical_compatibility = {
        name: {
            "role": VARIANT_CAPABILITIES[name].status.value,
            "stable_invariant_identifier": False,
            "determines_invariant_truth": False,
            "rationale": VARIANT_CAPABILITIES[name].rationale,
        }
        for name in (
            "access_without_auth",
            "impersonation",
            "identity_token_mismatch",
            "replay_token",
            "duplicate_sequence",
            "abnormal_failure_rate",
            "abnormal_renewal",
        )
    }
    return {
        "generation_profile": SYNCHRONIZED_GENERATION_PROFILE,
        "canonical_identifiers_assigned": False,
        "variants": {
            name: capability.to_dict()
            for name, capability in VARIANT_CAPABILITIES.items()
        },
        "supported_variants": list(supported_synchronized_variants()),
        "historical_compatibility_vocabulary": historical_compatibility,
        "ground_truth_derivation": "independent_atomic_validator_evidence_only",
        "injection_provenance_contract": [
            "injection_id",
            "anomaly_variant_name",
            "canonical_invariant_ids",
            "declared_history_scope",
            "source_evidence_event_ids",
            "transformation_applied",
            "validation_status",
        ],
    }


@dataclass
class InjectionRecord:
    """Controlled transformation provenance; not final validated ground truth."""

    injection_id: str
    historical_scenario_name: str
    anomaly_variant_name: str
    capability_status: str
    mapped_semantic_concept: str
    invariant_families: Tuple[str, ...]
    canonical_invariant_ids: Tuple[str, ...] = ()
    targeted_event_type: Optional[str] = None
    targeted_action_or_context: Optional[str] = None
    source_evidence_event_ids: Tuple[str, ...] = ()
    source_material_identifier: Optional[str] = None
    source_trace_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_auth_attempt_id: Optional[str] = None
    original_value: Any = None
    injected_value: Any = None
    injection_parameters: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    auth_attempt_id: Optional[str] = None
    declared_history_scope: Optional[str] = None
    intended_observable_violation: Optional[str] = None
    transformation_applied: bool = False
    evidence_check_passed: bool = False
    observable_violation_candidate: bool = False
    validation_status: str = "candidate_pending_full_executable_validation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
