"""Opt-in, single-family HN construction over the existing scenario/event engine.

Construction records intent and event references only. The synchronized
validator runs before HN.2 qualification; this module never assigns qualified
status or changes anomaly ground truth.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine, ScenarioSpec
from simulator.event_model import AuthEvent, EventType
from simulator.hard_negative import (
    CandidateStatus, ConstructionStatus, HN_FAMILY_BY_ID, HN_VARIANT_BY_ID,
    HardNegativeQualificationRecord, validate_hard_negative_structure,
)
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.provenance import GenerationProvenance, build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationReport, validate_synchronized_dataset


@dataclass(frozen=True)
class HardNegativeRequest:
    family_id: str
    variant_id: str
    candidate_id: str
    construction_parameters: Mapping[str, Any]
    boundary_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class HardNegativeGenerationResult:
    events: tuple[AuthEvent, ...]
    context: Any
    candidate: HardNegativeQualificationRecord
    validation: ValidationReport
    gated_validation: ValidationReport
    provenance: GenerationProvenance
    generator_defect: Optional[str]


_RETRY = {"fresh_nonce_cross_attempt"}
_RENEWAL = {
    "last_valid_use_before_replacement", "correct_replacement_token_use",
}
_DENIAL = {
    "no_grant_enforced", "explicit_deny_enforced",
    "valid_context_legitimate_denial",
}
_REJECTED_ACCESS = {
    "missing_auth_context_enforced": "authentication_context",
    "invalid_credential_enforced": "credential_context",
    "inactive_session_enforced": "active_session",
}


def _template(variant_id: str, seed: int) -> ScenarioSpec:
    factory = ScenarioEngine(seed)
    if variant_id in _REJECTED_ACCESS:
        spec = factory.normal()
        if variant_id == "missing_auth_context_enforced":
            spec.normal_steps = list(NORMAL_FLOW[:NORMAL_FLOW.index(EventType.AUTHENTICATION_REQUEST)])
        elif variant_id == "invalid_credential_enforced":
            spec.normal_steps = [
                *NORMAL_FLOW[:NORMAL_FLOW.index(EventType.TOKEN_VALIDATED)],
                EventType.TOKEN_REJECTED,
            ]
        else:
            spec.normal_steps = list(NORMAL_FLOW)
        spec.rejected_access_prerequisite = _REJECTED_ACCESS[variant_id]
    elif variant_id in _RETRY:
        spec = factory.normal_with_retry()
    elif variant_id in _RENEWAL:
        spec = factory.normal_with_renewal()
    else:
        spec = factory.normal()
        spec.normal_steps = list(NORMAL_FLOW)
    if variant_id == "repeated_current_token_use":
        access_index = spec.normal_steps.index(EventType.ACCESS_GRANTED)
        spec.normal_steps[access_index + 1:access_index + 1] = [
            EventType.ACCESS_REQUEST, EventType.ACCESS_GRANTED,
        ]
    if variant_id in _DENIAL:
        grant_index = spec.normal_steps.index(EventType.ACCESS_GRANTED)
        spec.normal_steps[grant_index] = EventType.ACCESS_DENIED
    spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
    return spec


def _event(events: list[AuthEvent], event_type: EventType, *, last: bool = False) -> Optional[AuthEvent]:
    matches = [item for item in events if item.event_type is event_type]
    return (matches[-1] if last else matches[0]) if matches else None


def _references(
    variant_id: str, events: list[AuthEvent], parameters: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any], Optional[str]]:
    """Select evidence from the emitted trace; never infer normality here."""
    target: Optional[AuthEvent] = None
    evidence: list[AuthEvent] = []
    boundary: dict[str, Any] = dict(parameters)
    if variant_id in {"fresh_nonce_cross_attempt", "legitimate_same_attempt_nonce_recurrence"}:
        nonces = [item for item in events if item.event_type is EventType.NONCE_RECEIVED]
        if variant_id == "fresh_nonce_cross_attempt":
            if len(nonces) < 2:
                return (), (), boundary, "two authentication attempts were not reached"
            source, target = nonces[0], nonces[-1]
            link = next((item for item in events if (
                item.event_type in {EventType.RETRY, EventType.RENEWAL_REQUEST}
                and item.auth_attempt_id == target.auth_attempt_id
                and source.timestamp < item.timestamp < target.timestamp
            )), None)
            if link is None:
                return (), (), boundary, "related retry/re-auth attempt was not reached"
            evidence = [source, link, target]
            boundary.setdefault(
                "attempt_relationship",
                "retry" if link.event_type is EventType.RETRY else "renewal",
            )
        else:
            first = _event(events, EventType.CHALLENGE_SENT)
            second = _event(events, EventType.NONCE_RECEIVED)
            if first is None or second is None:
                return (), (), boundary, "represented same-attempt nonce relation was not reached"
            evidence, target = [first, second], second
    elif variant_id == "repeated_current_token_use":
        uses = [item for item in events if item.event_type is EventType.ACCESS_REQUEST]
        if len(uses) < 2:
            return (), (), boundary, "a second protected token use was not reached"
        evidence, target = uses[:2], uses[1]
    elif variant_id in _RENEWAL:
        replacement = next((item for item in events if item.event_type is EventType.TOKEN_ISSUED and item.supersedes_token_id), None)
        if replacement is None:
            return (), (), boundary, "successful replacement lifecycle was not reached"
        if variant_id == "last_valid_use_before_replacement":
            use = next((item for item in events if item.event_type is EventType.ACCESS_REQUEST and item.timestamp < replacement.timestamp), None)
            if use is None:
                return (), (), boundary, "pre-replacement protected token use was not reached"
            evidence, target = [use, replacement], use
        else:
            use = next((item for item in events if item.event_type is EventType.TOKEN_PRESENTED and item.timestamp > replacement.timestamp), None)
            if use is None:
                return (), (), boundary, "replacement token presentation was not reached"
            evidence, target = [replacement, use], use
    elif variant_id == "small_positive_causal_margin":
        predecessor = _event(events, EventType.NONCE_RECEIVED)
        target = _event(events, EventType.RESPONSE_SENT)
        if predecessor is None or target is None:
            return (), (), boundary, "declared causal predecessor pair was not reached"
        epsilon_hn = boundary.get("epsilon_hn")
        if (not isinstance(epsilon_hn, (int, float)) or isinstance(epsilon_hn, bool)
                or not math.isfinite(epsilon_hn) or epsilon_hn <= 0):
            return (), (), boundary, "invalid positive epsilon_hn configuration"
        # Preserve semantic chronology and change only declared observed time.
        # Vary the generator-side margin within the explicit non-semantic
        # boundary parameter; a fixed midpoint is an HN-only timing marker.
        epsilon = float(epsilon_hn) * random.uniform(0.2, 0.8)
        boundary["sampled_margin_s"] = epsilon
        target.observed_timestamp = predecessor.observed_timestamp + epsilon
        target.observed_timestamp_source = "injected_device_timestamp"
        target.targeted_temporal_relationship = "declared_event_time_after_predecessor"
        evidence = [predecessor, target]
    elif variant_id in _DENIAL or variant_id in _REJECTED_ACCESS:
        request = _event(events, EventType.ACCESS_REQUEST, last=variant_id in _REJECTED_ACCESS)
        target = _event(events, EventType.ACCESS_DENIED, last=variant_id in _REJECTED_ACCESS)
        if request is None or target is None or request.access_request_id != target.access_request_id:
            return (), (), boundary, "protected request and enforced outcome were not reached"
        evidence = [request, target]
    elif variant_id == "persistent_authenticated_identity_coherent":
        association = _event(events, EventType.AUTHENTICATION_REQUEST)
        target = _event(events, EventType.AUTHENTICATION_SUCCESS)
        if association is None or target is None:
            return (), (), boundary, "authenticated identity context was not reached"
        evidence = [association, target]
        boundary.setdefault("boundary_targeted", True)
    elif variant_id in {"token_identity_binding_coherent", "token_protected_context_binding_coherent"}:
        issuance = _event(events, EventType.TOKEN_ISSUED)
        if variant_id == "token_protected_context_binding_coherent":
            target = _event(events, EventType.SESSION_OPENED)
        else:
            target = _event(events, EventType.TOKEN_PRESENTED)
        if issuance is None or target is None:
            return (), (), boundary, "token issuance and actual use were not reached"
        evidence = [issuance, target]
        boundary.setdefault("boundary_targeted", True)
    else:
        return (), (), boundary, "variant has no approved HN construction"
    return (target.event_id,), tuple(item.event_id for item in evidence), boundary, None


def generate_hard_negative(
    request: HardNegativeRequest,
    *,
    device_id: str,
    seed: int,
    start_time: float,
    scenario: Optional[ScenarioSpec] = None,
    persistent_context: Optional[PersistentDeviceContext] = None,
    provenance: Optional[GenerationProvenance] = None,
) -> HardNegativeGenerationResult:
    """Make exactly one explicit construction attempt; no automatic retry."""
    if not request.candidate_id:
        raise ValueError("HN candidate_id is required")
    family = HN_FAMILY_BY_ID.get(request.family_id)
    variant = HN_VARIANT_BY_ID.get(request.variant_id)
    if family is None or variant is None or variant.family_id != family.family_id:
        raise ValueError("unknown or inconsistent HN family/variant mapping")
    random.seed(seed)
    spec = scenario if scenario is not None else _template(variant.variant_id, seed)
    if spec.is_anomaly or spec.generation_profile != SYNCHRONIZED_GENERATION_PROFILE:
        raise ValueError("HN construction requires a synchronized non-anomaly scenario")
    persistent = persistent_context or PersistentDeviceContext(device_id=device_id)
    engine = EventEngine(
        device_id=device_id, gateway_id="gateway", auth_server_id="auth_server",
        broker_id="broker", start_time=start_time, persistent_context=persistent,
    )
    events, context = engine.execute(spec)
    targets, evidence, boundary, blocker = _references(variant.variant_id, events, request.boundary_parameters)
    if engine.last_rejected_request_error:
        blocker = engine.last_rejected_request_error
    construction_status = (
        ConstructionStatus.CONSTRUCTION_ERROR
        if blocker == "invalid positive epsilon_hn configuration" or engine.last_rejected_request_error
        else ConstructionStatus.REQUESTED_NOT_APPLIED if blocker
        else ConstructionStatus.APPLIED
    )
    candidate = HardNegativeQualificationRecord(
        hard_negative_candidate_id=request.candidate_id,
        trace_id=context.trace_id, scenario_id=context.scenario_id,
        device_id=context.device_id, candidate_status=CandidateStatus.REQUESTED,
        primary_intended_family_id=family.family_id,
        intended_variant_id=variant.variant_id,
        intended_boundary_invariant_ids=variant.invariant_ids,
        construction_status=construction_status,
        construction_parameters={
            **request.construction_parameters,
            "hn_generation_seed": seed,
            "hn_start_time": start_time,
            "hn_template": "caller_supplied" if scenario is not None else (
                "normal_retry" if variant.variant_id in _RETRY else
                "normal_renewal" if variant.variant_id in _RENEWAL else
                "normal_rejected_request" if variant.variant_id in _REJECTED_ACCESS else
                "normal_access_denial" if variant.variant_id in _DENIAL else
                "normal_two_accesses" if variant.variant_id == "repeated_current_token_use" else
                "normal"
            ),
            **({"applicability_blocker": blocker} if blocker else {}),
        },
        boundary_parameters=boundary,
        target_event_ids=targets, evidence_event_ids=evidence,
        qualification_reason=blocker or "construction applied; HN.2 not yet invoked",
    )
    provenance = provenance or build_generation_provenance(
        cfg, run_id=f"hn3-{request.candidate_id}",
    )
    manifest = field_manifest()
    dataset_manifest = dataset_capability_manifest(
        provenance, generation_profiles=(context.generation_profile,)
    )
    semantic_validation = validate_synchronized_dataset(
        [(events, context)], provenance=provenance,
        field_manifest=manifest, dataset_manifest=dataset_manifest,
    )
    qualified = qualify_hard_negative(events, context, candidate, semantic_validation)
    gated_validation = validate_synchronized_dataset(
        [(events, context)], provenance=provenance,
        field_manifest=manifest, dataset_manifest=dataset_manifest,
        hard_negative_qualification_records=(qualified,),
    )
    defect = engine.last_rejected_request_error
    if construction_status is ConstructionStatus.APPLIED and (
        qualified.disqualifying_invariant_ids
        or semantic_validation.trace_validated_invariant_ids.get(context.trace_id)
    ):
        defect = "applied normal-side construction produced a validated invariant violation"
    if not validate_hard_negative_structure((qualified,)).passed and construction_status is not ConstructionStatus.CONSTRUCTION_ERROR:
        defect = defect or "HN qualification record failed the structural gate"
    return HardNegativeGenerationResult(
        tuple(events), context, qualified, semantic_validation,
        gated_validation, provenance, defect,
    )
