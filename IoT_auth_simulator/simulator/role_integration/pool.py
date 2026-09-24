"""Deterministic role-independent valid-base opportunity-pool construction."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Mapping

from simulator.config.activity_planning import ActivityPlanningConfiguration
from simulator.config.opportunity_pool import (
    OPPORTUNITY_POOL_CONTRACT_VERSION,
    OPPORTUNITY_RNG_DERIVATION_VERSION,
    ROLE_RNG_DERIVATION_VERSION,
    TRANSFORMATION_RNG_DERIVATION_VERSION,
    OpportunityPoolConfiguration,
)
from simulator.config.population_timing import PopulationTimingConfiguration
from simulator.data.synchronized_views import (
    CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
    build_synchronized_bundle,
    canonicalize_sequence_pairs,
)
from simulator.device_profiles.scientific_population import (
    ScientificDeviceProfilePopulation,
)
from simulator.engines.event_engine import EventEngine
from simulator.planning.models import ConstructionRole, PlanningMode, TraceConstructionRequest
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    adapt_trace_plan,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    reconstruct_profile_conditioned_renewal,
)
from simulator.planning.randomness import derive_planning_randomness
from simulator.planning.repertoire import compose_ordinary_lifecycle
from simulator.role_integration.applicability import (
    APPLICABILITY_EVALUATOR_VERSION,
    APPLICABILITY_VOCABULARY_VERSION,
    ApplicabilityPredicate,
    ApplicabilityResult,
    evaluate_applicability,
    validate_evidence_references,
)
from simulator.role_integration.models import (
    BASE_TRACE_REFERENCE_VERSION,
    TRACE_SECURITY_SNAPSHOT_VERSION,
    VALID_BASE_ENVELOPE_VERSION,
    BaseValidationResult,
    StructuralRealizationSummary,
    TokenLineageSnapshot,
    TraceSecuritySnapshot,
    ValidBaseTrace,
    canonical_json,
    normalized_event_records,
    stable_reference,
)
from simulator.runner import RunPopulation
from simulator.timing.realizer import TimingRealizer
from simulator.validation import VALIDATOR_VERSION, VALIDATION_SCHEMA_VERSION


OPPORTUNITY_POOL_FINGERPRINT_VERSION = "4d3.7-canonical-admission-sha256-v1"
OPPORTUNITY_POOL_RECORD_VERSION = "4d3.7-canonical-admission-record-v1"
OPPORTUNITY_POOL_BUILDER_VERSION = "4d3.7-canonical-admission-builder-v1"
REJECTED_OPPORTUNITY_PROVENANCE_VERSION = "4d3.7-rejected-opportunity-v1"
OPPORTUNITY_RNG_NAMESPACE = "role-integration-base-opportunity"
ROLE_RNG_NAMESPACE = "future-role-assignment"
TRANSFORMATION_RNG_NAMESPACE = "future-family-transformation"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def derive_opportunity_reference(
    *, pool_seed: int, population_device_key: str, base_construction_key: str,
) -> str:
    return stable_reference({
        "derivation_version": OPPORTUNITY_RNG_DERIVATION_VERSION,
        "namespace": OPPORTUNITY_RNG_NAMESPACE,
        "pool_seed": pool_seed,
        "population_device_key": population_device_key,
        "base_construction_key": base_construction_key,
    })


def derive_future_role_reference(
    *, role_seed: int, pool_fingerprint: str, assignment_key: str,
) -> str:
    """Reserved future namespace; this function is not consumed by pool building."""
    return stable_reference({
        "derivation_version": ROLE_RNG_DERIVATION_VERSION,
        "namespace": ROLE_RNG_NAMESPACE,
        "role_seed": role_seed,
        "pool_fingerprint": pool_fingerprint,
        "assignment_key": assignment_key,
    })


def derive_future_transformation_reference(
    *, transformation_seed: int, base_trace_reference: str, transformation_key: str,
) -> str:
    """Reserved family-adapter namespace; carries no GT meaning."""
    return stable_reference({
        "derivation_version": TRANSFORMATION_RNG_DERIVATION_VERSION,
        "namespace": TRANSFORMATION_RNG_NAMESPACE,
        "transformation_seed": transformation_seed,
        "base_trace_reference": base_trace_reference,
        "transformation_key": transformation_key,
    })


@dataclass(frozen=True)
class RejectedBaseOpportunity:
    base_construction_key: str
    population_device_key: str
    validation_status: str
    reason: str
    rejection_reference: str = ""
    provenance_version: str = REJECTED_OPPORTUNITY_PROVENANCE_VERSION
    behavioral_profile_id: str = ""
    opportunity_randomness_reference: str = ""
    rejection_stage: str = "canonical_serialized_admission"
    source_timestamp_evidence_json: str = "[]"
    canonical_event_records_json: tuple[str, ...] = ()
    validation_evidence_json: str = "{}"


def _normalized_validation_failures(validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in validation.get("records", ()):
        if record.get("result") not in {"FAIL", "ERROR"}:
            continue
        evidence = record.get("observed_evidence", {})
        failures.append({
            "validation_id": record.get("validation_id"),
            "invariant_identifier": record.get("invariant_identifier"),
            "result": record.get("result"),
            "reason": record.get("reason"),
            "issues": [
                {key: value for key, value in issue.items() if key != "event_id"}
                for issue in evidence.get("violations", ())
            ],
        })
    return failures


def _rejected_opportunity(
    *,
    base_construction_key: str,
    assignment: Any,
    opportunity_reference: str,
    source_events: list[Any],
    canonical_events: list[Any],
    validation: Mapping[str, Any],
    validation_status: str,
) -> RejectedBaseOpportunity:
    source_timestamps = [
        {
            "event_ordinal": ordinal,
            "event_type": event.event_type.value,
            "semantic_timestamp": event.timestamp,
            "observed_timestamp": (
                event.observed_timestamp
                if event.observed_timestamp is not None else event.timestamp
            ),
        }
        for ordinal, event in enumerate(source_events, 1)
    ]
    canonical_records = tuple(
        canonical_json(event.to_dict()) for event in canonical_events
    )
    normalized_failures = _normalized_validation_failures(validation)
    reference = stable_reference({
        "provenance_version": REJECTED_OPPORTUNITY_PROVENANCE_VERSION,
        "canonical_serialization_version": CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
        "base_construction_key": base_construction_key,
        "population_device_key": assignment.population_device_key,
        "behavioral_profile_id": assignment.profile.behavioral_profile_id,
        "opportunity_randomness_reference": opportunity_reference,
        "source_timestamps": source_timestamps,
        "canonical_timestamps": [
            {
                "event_ordinal": ordinal,
                "event_type": event.event_type.value,
                "semantic_timestamp": event.timestamp,
                "observed_timestamp": (
                    event.observed_timestamp
                    if event.observed_timestamp is not None else event.timestamp
                ),
            }
            for ordinal, event in enumerate(canonical_events, 1)
        ],
        "validation_status": validation_status,
        "validation_failures": normalized_failures,
    })
    return RejectedBaseOpportunity(
        base_construction_key=base_construction_key,
        population_device_key=assignment.population_device_key,
        validation_status=validation_status,
        reason="canonical serialized evidence failed independent admission validation",
        rejection_reference=reference,
        behavioral_profile_id=assignment.profile.behavioral_profile_id,
        opportunity_randomness_reference=opportunity_reference,
        source_timestamp_evidence_json=canonical_json(source_timestamps),
        canonical_event_records_json=canonical_records,
        validation_evidence_json=canonical_json({
            "canonical_serialization_version": CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
            "build_gate": validation.get("build_gate"),
            "trace_ground_truth_statuses": validation.get("trace_ground_truth_statuses"),
            "trace_validated_invariant_ids": validation.get("trace_validated_invariant_ids"),
            "failures": normalized_failures,
        }),
    )


@dataclass(frozen=True)
class BaseOpportunityPool:
    contract_version: str
    record_version: str
    builder_version: str
    population_reference: str
    configuration_json: str
    bases: tuple[ValidBaseTrace, ...]
    applicability_matrix: tuple[tuple[ApplicabilityResult, ...], ...]
    rejected: tuple[RejectedBaseOpportunity, ...]
    diagnostics_json: str
    pool_fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != OPPORTUNITY_POOL_CONTRACT_VERSION:
            raise ValueError("unsupported opportunity-pool contract")
        if len(self.bases) != len(self.applicability_matrix):
            raise ValueError("each valid base requires one complete applicability vector")
        expected = len(tuple(ApplicabilityPredicate))
        if any(len(vector) != expected for vector in self.applicability_matrix):
            raise ValueError("applicability matrix is incomplete")
        if any(not base.validation.accepted for base in self.bases):
            raise ValueError("opportunity pools contain validated normal bases only")

    @property
    def diagnostics(self) -> dict[str, Any]:
        return json.loads(self.diagnostics_json)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "fingerprint_version": OPPORTUNITY_POOL_FINGERPRINT_VERSION,
            "contract_version": self.contract_version,
            "record_version": self.record_version,
            "builder_version": self.builder_version,
            "population_reference": self.population_reference,
            "configuration": json.loads(self.configuration_json),
            "base_references": [base.base_trace_reference for base in self.bases],
            "structural_realizations": [asdict(base.structural_summary) for base in self.bases],
            "applicability_matrix": [
                [result.to_record() for result in vector]
                for vector in self.applicability_matrix
            ],
            "rejected_attempt_references": [
                item.rejection_reference for item in self.rejected
            ],
            "rng_versions": {
                "opportunity": OPPORTUNITY_RNG_DERIVATION_VERSION,
                "future_role": ROLE_RNG_DERIVATION_VERSION,
                "future_transformation": TRANSFORMATION_RNG_DERIVATION_VERSION,
            },
            "base_contract_versions": {
                "valid_base_envelope": VALID_BASE_ENVELOPE_VERSION,
                "base_trace_reference": BASE_TRACE_REFERENCE_VERSION,
                "applicability_vocabulary": APPLICABILITY_VOCABULARY_VERSION,
                "applicability_evaluator": APPLICABILITY_EVALUATOR_VERSION,
                "validator": VALIDATOR_VERSION,
                "validation_schema": VALIDATION_SCHEMA_VERSION,
                "canonical_evidence_serialization": (
                    CANONICAL_EVIDENCE_SERIALIZATION_VERSION
                ),
                "rejected_opportunity_provenance": (
                    REJECTED_OPPORTUNITY_PROVENANCE_VERSION
                ),
            },
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.fingerprint_payload(),
            "pool_fingerprint": self.pool_fingerprint,
            "bases": [base.to_record() for base in self.bases],
            "rejected": [asdict(item) for item in self.rejected],
            "diagnostics": self.diagnostics,
            "publication_status": "internal_privileged_not_a_synchronized_dataset_surface",
        }


def validate_pool_record(record: Mapping[str, Any]) -> None:
    """Validate canonical reload references without publishing a companion file."""
    bases = record.get("bases")
    matrix = record.get("applicability_matrix")
    if not isinstance(bases, list) or not isinstance(matrix, list) or len(bases) != len(matrix):
        raise ValueError("invalid opportunity-pool base/applicability structure")
    for base, vector in zip(bases, matrix):
        event_records = tuple(base.get("event_records", ()))
        event_count = len(event_records)
        normalized = list(normalized_event_records(event_records))
        if normalized != base.get("normalized_event_evidence"):
            raise ValueError("serialized normalized event evidence mismatch")
        base_identity = {
            "reference_version": BASE_TRACE_REFERENCE_VERSION,
            "envelope_version": base["envelope_version"],
            "base_construction_key": base["base_construction_key"],
            "opportunity_randomness_reference": base["opportunity_randomness_reference"],
            "population_reference": base["population_reference"],
            "population_device_key": base["population_device_key"],
            "behavioral_profile_id": base["behavioral_profile_id"],
            "trace_plan": base["trace_plan"],
            "planning_provenance": base["planning_provenance"],
            "timing_provenance": base["timing_provenance"],
            "structural_summary": base["structural_summary"],
            "normalized_events": normalized,
            "validation": base["validation"],
        }
        if stable_reference(base_identity) != base.get("base_trace_reference"):
            raise ValueError("serialized base-trace reference mismatch")
        for result in vector:
            for reference in result.get("evidence_references", ()):
                prefix, _, ordinal = str(reference).partition(":")
                if prefix != "event" or not ordinal.isdigit() or not 1 <= int(ordinal) <= event_count:
                    raise ValueError("serialized applicability evidence does not resolve")
    payload_keys = (
        "fingerprint_version", "contract_version", "record_version",
        "builder_version", "population_reference", "configuration",
        "base_references", "structural_realizations", "applicability_matrix",
        "rejected_attempt_references", "rng_versions", "base_contract_versions",
    )
    rejected = record.get("rejected", ())
    if [item.get("rejection_reference") for item in rejected] != record.get(
        "rejected_attempt_references"
    ):
        raise ValueError("serialized rejected-opportunity references mismatch")
    actual = stable_reference({key: record[key] for key in payload_keys})
    if actual != record.get("pool_fingerprint"):
        raise ValueError("opportunity-pool fingerprint mismatch")


def _security_snapshot(
    events: tuple[dict[str, Any], ...], context: Any,
) -> TraceSecuritySnapshot:
    def unique(field: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            str(event[field]) for event in events if event.get(field) is not None
        ))
    token_ids = unique("token_id")
    lineage: list[TokenLineageSnapshot] = []
    for token_id in token_ids:
        represented = [event for event in events if event.get("token_id") == token_id]
        latest = represented[-1]
        lineage.append(TokenLineageSnapshot(
            token_reference=token_id,
            generation=latest.get("token_generation"),
            lineage_status=latest.get("token_lineage_status"),
            supersedes_token_reference=latest.get("supersedes_token_id"),
            superseded_by_token_reference=latest.get("superseded_by_token_id"),
            bound_identity_matches_authenticated_identity=(
                latest.get("token_bound_identity") == latest.get("authenticated_identity")
                if latest.get("token_bound_identity") is not None
                and latest.get("authenticated_identity") is not None else None
            ),
            bound_context_matches_protected_session=(
                latest.get("token_bound_protected_session_id")
                == latest.get("protected_session_id")
                if latest.get("token_bound_protected_session_id") is not None
                and latest.get("protected_session_id") is not None else None
            ),
        ))
    return TraceSecuritySnapshot(
        version=TRACE_SECURITY_SNAPSHOT_VERSION,
        authentication_attempt_references=unique("auth_attempt_id"),
        protected_session_references=unique("protected_session_id"),
        token_references=token_ids,
        nonce_references=unique("nonce"),
        access_request_references=unique("access_request_id"),
        token_lineage=tuple(lineage),
        authentication_result=context.authentication_result_semantic,
        authenticated_identity_matches_device=(
            context.authenticated_identity == context.device_id
            if context.authenticated_identity is not None else None
        ),
        token_validation_result=context.token_validation_result,
        authorization_decisions=tuple(context.authorization_decisions),
        resource_operation_outcomes=tuple(context.resource_operation_outcomes),
        authenticated_context_active=context.authenticated_context_active,
        token_context_active=context.token_context_active,
        protected_session_active=context.protected_session_active,
    )


def _diagnostics(
    bases: tuple[ValidBaseTrace, ...],
    matrix: tuple[tuple[ApplicabilityResult, ...], ...],
    rejected: tuple[RejectedBaseOpportunity, ...],
    population: ScientificDeviceProfilePopulation,
) -> dict[str, Any]:
    device_counts = Counter(base.population_device_key for base in bases)
    structural = {
        name: dict(sorted(Counter(getattr(base.structural_summary, name) for base in bases).items()))
        for name in ("lifecycle", "recovery", "renewal", "activity_count")
    }
    predicate_diagnostics: dict[str, Any] = {}
    for predicate_index, predicate_results in enumerate(zip(*matrix)) if matrix else ():
        first = predicate_results[0]
        count = sum(result.applicable for result in predicate_results)
        supporting_devices = {
            bases[index].population_device_key
            for index, result in enumerate(predicate_results) if result.applicable
        }
        predicate_id = first.predicate_id
        if predicate_id in {
            "protected_access_exists", "multiple_current_token_uses_exist",
            "authoritative_grant_exists", "authoritative_denial_exists",
            "successful_token_replacement_exists",
            "current_t1_use_before_replacement_exists",
            "t2_presentation_after_replacement_exists",
        }:
            denominator = sum(base.structural_summary.lifecycle == "ACCESS" for base in bases)
            denominator_name = "ACCESS bases"
        elif predicate_id in {
            "active_protected_session_exists",
            "token_authenticated_identity_binding_evidence_exists",
            "token_protected_context_binding_evidence_exists",
        }:
            denominator = sum(
                base.structural_summary.lifecycle in {"SESSION", "ACCESS"}
                for base in bases
            )
            denominator_name = "SESSION-or-ACCESS bases"
        else:
            denominator = len(bases)
            denominator_name = "all valid bases"
        predicate_diagnostics[predicate_id] = {
            "applicable_count": count,
            "non_applicable_count": len(bases) - count,
            "applicable_fraction": (count / len(bases)) if bases else 0.0,
            "devices_with_applicable_opportunity": len(supporting_devices),
            "structural_denominator_name": denominator_name,
            "structural_denominator": denominator,
        }

    assignments = {item.population_device_key: item for item in population.assignments}
    concentration: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for base, vector in zip(bases, matrix):
        assignment = assignments[base.population_device_key]
        applicable_ids = [item.predicate_id for item in vector if item.applicable]
        for predicate_id in applicable_ids:
            concentration[f"{predicate_id}|lifecycle_cell"][assignment.lifecycle_cell_id] += 1
            for tendency, stratum in assignment.scalar_strata.items():
                concentration[f"{predicate_id}|{tendency}_stratum"][str(stratum)] += 1
            summary = base.structural_summary
            for key, value in (
                ("realized_L", summary.lifecycle), ("realized_R", summary.recovery),
                ("realized_N", summary.renewal), ("realized_A", summary.activity_count),
            ):
                concentration[f"{predicate_id}|{key}"][str(value)] += 1
    timing_family_counts: Counter[str] = Counter()
    intervention_counts: Counter[str] = Counter()
    realized_delays: list[float] = []
    for base in bases:
        timing = base.timing_provenance
        for relation in timing.get("relations", ()):
            timing_family_counts[str(relation["timing_family"])] += 1
            realized_delays.append(float(relation["realized_delay_s"]))
            reason = relation.get("envelope_intervention_reason")
            if reason:
                intervention_counts[str(reason)] += 1
    return {
        "claim_boundary": "descriptive controlled-synthetic opportunity characterization only",
        "canonical_admission_version": OPPORTUNITY_POOL_BUILDER_VERSION,
        "canonical_serialization_version": CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
        "rejection_provenance_version": REJECTED_OPPORTUNITY_PROVENANCE_VERSION,
        "attempted_bases": len(bases) + len(rejected),
        "valid_bases": len(bases),
        "invalid_or_unsupported_bases": len(rejected),
        "devices_represented": len(device_counts),
        "bases_per_device": dict(sorted(device_counts.items())),
        "repeated_opportunities_per_device": {
            key: max(0, count - 1) for key, count in sorted(device_counts.items())
        },
        "structural_realizations": structural,
        "timing_realization_present": sum(
            base.structural_summary.timing_realization_present for base in bases
        ),
        "timing_realization": {
            "relation_count": len(realized_delays),
            "family_counts": dict(sorted(timing_family_counts.items())),
            "envelope_intervention_reasons": dict(sorted(intervention_counts.items())),
            "minimum_realized_delay_s": min(realized_delays) if realized_delays else None,
            "maximum_realized_delay_s": max(realized_delays) if realized_delays else None,
            "mean_realized_delay_s": (
                sum(realized_delays) / len(realized_delays) if realized_delays else None
            ),
            "claim_boundary": "controlled-synthetic realized delays; not empirical timing",
        },
        "applicability": predicate_diagnostics,
        "profile_applicability_concentrations": {
            key: dict(sorted(value.items())) for key, value in sorted(concentration.items())
        },
        "construction_order": [
            "scientific_population", "persistent_profile", "base_trace",
            "independent_validation", "applicability",
        ],
        "role_assignment_present": False,
        "transformation_present": False,
        "diagnostics_influence_generation": False,
    }


def build_opportunity_pool(
    population: RunPopulation,
    configuration: OpportunityPoolConfiguration = OpportunityPoolConfiguration(),
    *,
    activity_configuration: ActivityPlanningConfiguration = ActivityPlanningConfiguration(),
    timing_configuration: PopulationTimingConfiguration = PopulationTimingConfiguration(),
) -> BaseOpportunityPool:
    """Build a frozen pool without requested roles, transformations, or backfill."""
    if not isinstance(population.behavioral_profiles, ScientificDeviceProfilePopulation):
        raise TypeError("4D.3.1 opportunity pools require a fixed scientific population")
    scientific_population = population.behavioral_profiles
    # A pool is a reproducible construction campaign, not a mutation of the
    # caller's run state. Local contexts retain persistence across bases in this
    # pool while leaving the fixed input population reusable for reconstruction.
    semantic_contexts = deepcopy(population.semantic_contexts)
    resolver = ApplicabilityResolver()
    planner = DeviceAwareTracePlanner()
    valid: list[ValidBaseTrace] = []
    vectors: list[tuple[ApplicabilityResult, ...]] = []
    rejected: list[RejectedBaseOpportunity] = []
    original_global_rng_state = random.getstate()
    ordinal = 0
    try:
        for repetition in range(configuration.opportunities_per_device):
            for assignment in scientific_population.assignments:
                ordinal += 1
                key = f"base/{assignment.population_device_key}/opportunity-{repetition + 1}"
                request = TraceConstructionRequest.profile_conditioned_lifecycle(
                    trace_construction_key=key,
                    scenario_id=f"ordinary-{key}",
                )
                semantic_context = semantic_contexts[assignment.device_id]
                applicable = resolver.resolve(semantic_context, ConstructionRole.ORDINARY)
                planning_randomness = derive_planning_randomness(
                    planning_root_seed=configuration.planning_root_seed,
                    population_device_key=assignment.population_device_key,
                    trace_construction_key=key,
                )
                lifecycle = reconstruct_profile_conditioned_lifecycle(
                    assignment.profile.lifecycle, applicable, planning_randomness,
                )
                recovery_applicable = resolver.resolve_recovery(
                    semantic_context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                )
                recovery = reconstruct_profile_conditioned_recovery(
                    assignment.profile.recovery, recovery_applicable, planning_randomness,
                )
                base_steps = compose_ordinary_lifecycle(
                    lifecycle.selected_lifecycle_depth, recovery.selected_recovery_choice,
                )
                renewal_applicable = resolver.resolve_renewal(
                    semantic_context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth, recovery.selected_recovery_choice,
                    base_steps,
                )
                renewal = reconstruct_profile_conditioned_renewal(
                    assignment.profile.renewal, renewal_applicable, planning_randomness,
                )
                activity_applicable = resolver.resolve_activity(
                    semantic_context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth, recovery.selected_recovery_choice,
                    renewal.selected_renewal_choice, base_steps, activity_configuration,
                )
                plan = planner.plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
                    request, lifecycle, recovery, renewal, activity_applicable,
                    assignment.profile.activity, planning_randomness, activity_configuration,
                )
                timing = TimingRealizer(
                    configuration=timing_configuration,
                    timing_profile=assignment.profile.timing,
                    planning_rng_reference=planning_randomness.reference,
                    event_steps=plan.event_steps,
                )
                opportunity_reference = derive_opportunity_reference(
                    pool_seed=configuration.pool_seed,
                    population_device_key=assignment.population_device_key,
                    base_construction_key=key,
                )
                random.seed(int(opportunity_reference.removeprefix("sha256:"), 16))
                device = population.devices_by_id[assignment.device_id]
                source_pair = EventEngine(
                    device_id=device.device_id,
                    gateway_id=f"opportunity-gateway-{ordinal % 5}",
                    auth_server_id="opportunity-auth",
                    broker_id="opportunity-broker",
                    source_ip=device.ip_address,
                    battery_level=device.battery_level,
                    firmware_version=device.firmware_version,
                    start_time=(configuration.semantic_time_origin
                                + ordinal * configuration.trace_start_spacing_s),
                    persistent_context=semantic_context,
                    timing_realizer=timing,
                ).execute(adapt_trace_plan(plan))
                source_events, _source_context = source_pair
                # Admission is authoritative only over the exact six-decimal
                # event representation received by downstream consumers.  No
                # timestamp is repaired and a failed attempt is never retried.
                events, context = canonicalize_sequence_pairs([source_pair])[0]
                bundle = build_synchronized_bundle([(events, context)])
                trace_id = context.trace_id
                status = bundle["validation"]["trace_ground_truth_statuses"][trace_id]
                invariant_ids = tuple(
                    bundle["validation"]["trace_validated_invariant_ids"][trace_id]
                )
                validation = BaseValidationResult(
                    validation_reference=stable_reference({
                        "validator_version": VALIDATOR_VERSION,
                        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                        "trace_status": status,
                        "trace_validated_invariant_ids": invariant_ids,
                        "build_gate_passed": bundle["validation"]["build_gate"]["passed"],
                        "base_construction_key": key,
                        "canonical_serialization_version": (
                            CANONICAL_EVIDENCE_SERIALIZATION_VERSION
                        ),
                    }),
                    validator_version=VALIDATOR_VERSION,
                    validation_schema_version=VALIDATION_SCHEMA_VERSION,
                    validation_status=status,
                    build_gate_passed=bundle["validation"]["build_gate"]["passed"],
                    validated_invariant_ids=invariant_ids,
                    anomaly_ground_truth_present=bool(
                        context.observable_anomaly
                        or context.attack_scenario_intent
                        or context.injection_records
                        or any(event.anomaly_label for event in events)
                    ),
                    hard_negative_role_present=False,
                    transformation_provenance_present=False,
                )
                if not validation.accepted:
                    rejected.append(_rejected_opportunity(
                        base_construction_key=key,
                        assignment=assignment,
                        opportunity_reference=opportunity_reference,
                        source_events=source_events,
                        canonical_events=events,
                        validation=bundle["validation"],
                        validation_status=status,
                    ))
                    continue
                event_records = tuple(event.to_dict() for event in events)
                planning_record = plan.provenance_record()
                timing_record = timing.provenance_record()
                planning_record["timing_realization"] = timing_record
                base = ValidBaseTrace(
                    envelope_version=VALID_BASE_ENVELOPE_VERSION,
                    base_trace_reference="pending",
                    base_construction_key=key,
                    opportunity_randomness_reference=opportunity_reference,
                    population_reference=scientific_population.population_fingerprint,
                    device_id=assignment.device_id,
                    population_device_key=assignment.population_device_key,
                    behavioral_profile_id=assignment.profile.behavioral_profile_id,
                    persistent_context_reference=stable_reference({
                        "population_reference": scientific_population.population_fingerprint,
                        "population_device_key": assignment.population_device_key,
                        "context_kind": "PersistentDeviceContext",
                    }),
                    trace_plan_json=canonical_json(_json_value(plan)),
                    planning_provenance_json=canonical_json(planning_record),
                    event_records_json=tuple(canonical_json(record) for record in event_records),
                    session_context_json=canonical_json(context.to_dict()),
                    security_snapshot=_security_snapshot(event_records, context),
                    timing_provenance_json=canonical_json(timing_record),
                    structural_summary=StructuralRealizationSummary(
                        lifecycle=plan.lifecycle_depth.value,
                        recovery=plan.recovery_choice.value,
                        renewal=plan.renewal_choice.value,
                        activity_count=plan.selected_protected_operation_count,
                        timing_realization_present=True,
                        population_device_key=assignment.population_device_key,
                        behavioral_profile_id=assignment.profile.behavioral_profile_id,
                    ),
                    validation=validation,
                )
                base = replace(base, base_trace_reference=stable_reference(
                    base.normalized_identity_record()
                ))
                vector = evaluate_applicability(base)
                validate_evidence_references(base, vector)
                valid.append(base)
                vectors.append(vector)
    finally:
        random.setstate(original_global_rng_state)

    bases_tuple, vectors_tuple, rejected_tuple = tuple(valid), tuple(vectors), tuple(rejected)
    diagnostics = _diagnostics(
        bases_tuple, vectors_tuple, rejected_tuple, scientific_population,
    )
    pool = BaseOpportunityPool(
        contract_version=OPPORTUNITY_POOL_CONTRACT_VERSION,
        record_version=OPPORTUNITY_POOL_RECORD_VERSION,
        builder_version=OPPORTUNITY_POOL_BUILDER_VERSION,
        population_reference=scientific_population.population_fingerprint,
        configuration_json=canonical_json(configuration.base_identity_record()),
        bases=bases_tuple,
        applicability_matrix=vectors_tuple,
        rejected=rejected_tuple,
        diagnostics_json=canonical_json(diagnostics),
        pool_fingerprint="pending",
    )
    return replace(pool, pool_fingerprint=stable_reference(pool.fingerprint_payload()))
