"""Independent executable validation for the C1 synchronized v1 subset.

The validator inspects emitted events and contexts.  Scenario names and
injection records select an intended candidate check, but only reconstructed
event evidence can confirm a violation.  This is intentionally a focused rule
layer, not a second copy of the historical FSM or a final invariant catalogue.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from simulator.anomaly_contract import (
    SYNCHRONIZED_ACCESS_VARIANTS,
    SYNCHRONIZED_GENERATION_PROFILE,
    SYNCHRONIZED_IDENTITY_VARIANTS,
    SYNCHRONIZED_TOKEN_BINDING_VARIANTS,
    supported_synchronized_variants,
)
from simulator.event_model import AuthEvent, AuthState, EventResult, EventType
from simulator.invariant_registry import (
    EXACT_VALIDATOR_CHECK_TO_INVARIANT,
    EXACT_VARIANT_TO_INVARIANTS,
    INVARIANT_REGISTRY_VERSION,
)
from simulator.provenance import BEHAVIOR_MODEL_VERSION, GenerationProvenance
from simulator.hard_negative import (
    HardNegativeQualificationRecord,
    validate_hard_negative_structure,
)


VALIDATOR_VERSION = "sync-v1.1.0-validator"
VALIDATION_SCHEMA_VERSION = "sync-v1.1.0-validation-report-schema"
EVENT_GROUND_TRUTH_SCHEMA_VERSION = "sync-v1.1.0-event-ground-truth-schema"
TRACE_GROUND_TRUTH_SCHEMA_VERSION = "sync-v1.1.0-trace-ground-truth-schema"
# Compatibility name for callers that consume the combined GT bundle.
GROUND_TRUTH_SCHEMA_VERSION = "sync-v1.1.0-ground-truth-bundle"

GROUND_TRUTH_DERIVATION_RULES = {
    "positive_and_unsupported": "validated_positive",
    "no_positive_and_unsupported": "unsupported",
    "injection_applied_but_not_confirmed": "validation_failed",
    "attack_intent_without_applied_injection": "validated_normal",
    "multiple_confirmed_invariants": "validated_positive_with_all_ids",
    "unlabeled_unexpected_validator_violation": "validation_failed",
    "no_positive_no_unsupported": "validated_normal",
    "atomic_rule_not_applicable": (
        "ValidationResult.NOT_APPLICABLE; this does not prevent the trace from "
        "being validated_normal under other applicable checks"
    ),
    "principle": (
        "stable invariant ground truth is derived only from confirmed atomic "
        "validator evidence; scenario intent and injection application are not truth"
    ),
}


class ValidationResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class GroundTruthValidationStatus(str, Enum):
    VALIDATED_POSITIVE = "validated_positive"
    VALIDATED_NORMAL = "validated_normal"
    VALIDATION_FAILED = "validation_failed"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class BuildGateError(RuntimeError):
    """Raised when a caller elects to enforce a failed deterministic gate."""


class ValidationCategory(str, Enum):
    STATE_RESULT = "state_result_consistency"
    LIFECYCLE_CAUSAL = "lifecycle_causal_ordering"
    TEMPORAL = "temporal_consistency"
    IDENTITY = "identity_consistency"
    CREDENTIAL_TOKEN = "credential_token_binding"
    SESSION_CONTINUITY = "session_continuity_context"
    AUTHORIZATION_ACCESS = "authorization_access_preconditions"
    REPLAY_SINGLE_USE = "replay_single_use"
    TIMESTAMP_EVIDENCE = "timestamp_inconsistency_evidence"
    GROUND_TRUTH = "ground_truth_injection_consistency"
    SCHEMA_PROVENANCE = "schema_provenance_completeness"
    HARD_NEGATIVE = "hard_negative_candidate_validity"
    UNSUPPORTED_CAPABILITY = "unsupported_validation"


@dataclass(frozen=True)
class ValidationRecord:
    validation_id: str
    validation_category: ValidationCategory
    invariant_family: Optional[str]
    invariant_identifier: Optional[str]
    trace_id: Optional[str]
    device_id: Optional[str]
    protected_session_id: Optional[str]
    event_references: tuple[str, ...]
    expected_condition: str
    observed_evidence: dict[str, Any]
    result: ValidationResult
    reason: str
    validator_version: str = VALIDATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["validation_category"] = self.validation_category.value
        result["result"] = self.result.value
        return result


@dataclass
class ValidationReport:
    records: list[ValidationRecord]
    behavior_model_version: str
    generator_version: str
    generator_git_commit: Optional[str]
    event_schema_version: str
    feature_schema_version: str
    traces_checked: int
    events_checked: int
    trace_ground_truth_statuses: dict[str, str] = field(default_factory=dict)
    event_ground_truth_statuses: dict[str, str] = field(default_factory=dict)
    validated_positive_event_ids: set[str] = field(default_factory=set)
    event_validated_invariant_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trace_validated_invariant_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    event_validated_invariant_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unexpected_normal_trace_failures: list[str] = field(default_factory=list)
    intended_candidates_confirmed: list[dict[str, str]] = field(default_factory=list)
    intended_candidates_not_confirmed: list[dict[str, str]] = field(default_factory=list)
    gt_consistency_failures: list[str] = field(default_factory=list)
    provenance_schema_failures: list[str] = field(default_factory=list)
    build_gate_passed: bool = False
    mandatory_failure_reasons: list[str] = field(default_factory=list)
    hard_negative_structural_gate: dict[str, Any] = field(default_factory=dict)

    def require_build_gate(self) -> None:
        """Provide an opt-in release gate without writing or deleting artifacts."""
        if not self.build_gate_passed:
            raise BuildGateError("; ".join(self.mandatory_failure_reasons))

    def to_dict(self) -> dict[str, Any]:
        result_counts = Counter(record.result.value for record in self.records)
        category_counts: dict[str, dict[str, int]] = {}
        for category in ValidationCategory:
            counts = Counter(
                record.result.value for record in self.records
                if record.validation_category is category
            )
            category_counts[category.value] = {
                status.value: counts.get(status.value, 0)
                for status in ValidationResult
            }
        return {
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "behavior_model_version": self.behavior_model_version,
            "generator_version": self.generator_version,
            "generator_git_commit": self.generator_git_commit,
            "event_schema_version": self.event_schema_version,
            "feature_schema_version": self.feature_schema_version,
            "traces_checked": self.traces_checked,
            "events_checked": self.events_checked,
            "checks_executed_by_category": category_counts,
            "result_counts": {
                status.value: result_counts.get(status.value, 0)
                for status in ValidationResult
            },
            "unexpected_normal_trace_failures": self.unexpected_normal_trace_failures,
            "intended_anomaly_candidates_confirmed": self.intended_candidates_confirmed,
            "intended_anomaly_candidates_not_confirmed": self.intended_candidates_not_confirmed,
            "ground_truth_consistency_failures": self.gt_consistency_failures,
            "provenance_schema_failures": self.provenance_schema_failures,
            "ground_truth_status_vocabulary": [
                status.value for status in GroundTruthValidationStatus
            ],
            "ground_truth_derivation_rules": GROUND_TRUTH_DERIVATION_RULES,
            "trace_ground_truth_statuses": self.trace_ground_truth_statuses,
            "event_ground_truth_statuses": self.event_ground_truth_statuses,
            "validated_positive_event_ids": sorted(self.validated_positive_event_ids),
            "event_validated_invariant_ids": {
                key: list(value)
                for key, value in sorted(self.event_validated_invariant_ids.items())
            },
            "trace_validated_invariant_ids": {
                key: list(value)
                for key, value in sorted(self.trace_validated_invariant_ids.items())
            },
            "event_validated_invariant_evidence": {
                key: value
                for key, value in sorted(
                    self.event_validated_invariant_evidence.items()
                )
            },
            "build_gate": {
                "passed": self.build_gate_passed,
                "mandatory_failure_reasons": self.mandatory_failure_reasons,
            },
            "records": [record.to_dict() for record in self.records],
        }


@dataclass
class _TraceOutcome:
    records: list[ValidationRecord]
    confirmed_event_ids: set[str]
    confirmed_variants: set[str]
    intended_variants: set[str]
    deterministic_failed: bool
    gt_failed: bool
    event_invariant_ids: dict[str, set[str]] = field(default_factory=dict)
    event_invariant_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unsupported_event_ids: set[str] = field(default_factory=set)
    has_unsupported_atomic_evidence: bool = False


class DatasetValidator:
    """Validate the deterministic subset represented by synchronized traces."""

    _DETERMINISTIC_CATEGORIES = {
        ValidationCategory.STATE_RESULT,
        ValidationCategory.LIFECYCLE_CAUSAL,
        ValidationCategory.TEMPORAL,
        ValidationCategory.IDENTITY,
        ValidationCategory.CREDENTIAL_TOKEN,
        ValidationCategory.SESSION_CONTINUITY,
        ValidationCategory.AUTHORIZATION_ACCESS,
    }

    def validate(
        self,
        sequences: list[tuple[list[AuthEvent], Any]],
        *,
        provenance: GenerationProvenance,
        field_manifest: Optional[dict[str, Any]],
        dataset_manifest: Optional[dict[str, Any]],
        hard_negative_trace_ids: Optional[Iterable[str]] = None,
        hard_negative_qualification_records: Optional[
            Iterable[HardNegativeQualificationRecord]
        ] = None,
    ) -> ValidationReport:
        records: list[ValidationRecord] = []
        outcomes: dict[str, _TraceOutcome] = {}
        internal_error_traces: list[str] = []

        for events, context in sequences:
            try:
                outcome = self._validate_trace(events, context)
            except Exception as exc:  # expose validator failure; never convert to pass
                record = self._record(
                    "c1.7.validator.internal_error",
                    ValidationCategory.STATE_RESULT,
                    context,
                    "mandatory trace validators execute without internal error",
                    {"exception_type": type(exc).__name__, "message": str(exc)},
                    ValidationResult.ERROR,
                    "validator execution failed",
                )
                outcome = _TraceOutcome([record], set(), set(), set(), True, True)
                internal_error_traces.append(context.trace_id)
            outcomes[context.trace_id] = outcome
            records.extend(outcome.records)

        persistence_record = self._validate_cross_trace_persistence(sequences)
        records.append(persistence_record)

        schema_record = self._validate_schema_provenance(
            sequences, provenance, field_manifest, dataset_manifest
        )
        records.append(schema_record)

        hard_negative_records = self._validate_hard_negative_candidates(
            sequences, outcomes, hard_negative_trace_ids
        )
        records.extend(hard_negative_records)
        hn_structural_gate = validate_hard_negative_structure(
            hard_negative_qualification_records or (),
            field_manifest=field_manifest,
            known_event_ids_by_trace={
                context.trace_id: {event.event_id for event in events}
                for events, context in sequences
            },
            trace_anomaly_gt={
                context.trace_id: bool(
                    context.observable_anomaly or any(event.anomaly_label for event in events)
                )
                for events, context in sequences
            },
        )
        records.extend(self._unsupported_records())

        unexpected_normal = sorted(
            context.trace_id
            for _events, context in sequences
            if not context.attack_scenario_intent
            and outcomes[context.trace_id].deterministic_failed
        )
        confirmed: list[dict[str, str]] = []
        not_confirmed: list[dict[str, str]] = []
        positive_event_ids: set[str] = set()
        gt_failures: list[str] = []
        trace_statuses: dict[str, str] = {}
        event_statuses: dict[str, str] = {}
        event_invariant_ids: dict[str, tuple[str, ...]] = {}
        trace_invariant_ids: dict[str, tuple[str, ...]] = {}
        event_invariant_evidence: dict[str, list[dict[str, Any]]] = {}

        for events, context in sequences:
            outcome = outcomes[context.trace_id]
            positive_event_ids.update(outcome.confirmed_event_ids)
            for variant in sorted(outcome.intended_variants):
                item = {"trace_id": context.trace_id, "variant": variant}
                (confirmed if variant in outcome.confirmed_variants else not_confirmed).append(item)
            if outcome.gt_failed:
                gt_failures.append(context.trace_id)

            validation_failed = (
                outcome.deterministic_failed
                or outcome.gt_failed
                or outcome.intended_variants != outcome.confirmed_variants
            )
            if validation_failed:
                status = GroundTruthValidationStatus.VALIDATION_FAILED.value
            elif outcome.confirmed_variants:
                status = GroundTruthValidationStatus.VALIDATED_POSITIVE.value
            elif outcome.has_unsupported_atomic_evidence:
                status = GroundTruthValidationStatus.UNSUPPORTED.value
            else:
                status = GroundTruthValidationStatus.VALIDATED_NORMAL.value
            trace_statuses[context.trace_id] = status
            trace_invariant_ids[context.trace_id] = tuple(sorted({
                invariant_id
                for ids in outcome.event_invariant_ids.values()
                for invariant_id in ids
            }))
            for event in events:
                if status == "validation_failed":
                    event_statuses[event.event_id] = GroundTruthValidationStatus.VALIDATION_FAILED.value
                elif event.event_id in outcome.confirmed_event_ids:
                    event_statuses[event.event_id] = GroundTruthValidationStatus.VALIDATED_POSITIVE.value
                elif event.event_id in outcome.unsupported_event_ids:
                    event_statuses[event.event_id] = GroundTruthValidationStatus.UNSUPPORTED.value
                else:
                    event_statuses[event.event_id] = GroundTruthValidationStatus.VALIDATED_NORMAL.value
                event_invariant_ids[event.event_id] = tuple(sorted(
                    outcome.event_invariant_ids.get(event.event_id, set())
                ))
                event_invariant_evidence[event.event_id] = list(
                    outcome.event_invariant_evidence.get(event.event_id, ())
                )

        provenance_failures = (
            list(schema_record.observed_evidence.get("violations", ()))
            if schema_record.result in {ValidationResult.FAIL, ValidationResult.ERROR}
            else []
        )
        deterministic_failures = sorted(
            trace_id for trace_id, outcome in outcomes.items()
            if outcome.deterministic_failed
        )
        mandatory_reasons: list[str] = []
        if deterministic_failures:
            mandatory_reasons.append(
                "deterministic invariant failures: " + ", ".join(deterministic_failures)
            )
        if not_confirmed:
            mandatory_reasons.append("supported intended anomaly candidate not confirmed")
        if gt_failures:
            mandatory_reasons.append("ground-truth/injection consistency failure")
        if provenance_failures:
            mandatory_reasons.append("required schema/provenance incomplete")
        if persistence_record.result in {ValidationResult.FAIL, ValidationResult.ERROR}:
            mandatory_reasons.append("persistent enrollment continuity failure")
        if any(record.result in {ValidationResult.FAIL, ValidationResult.ERROR} for record in hard_negative_records):
            mandatory_reasons.append("declared hard-negative candidate violates supported checks")
        if internal_error_traces:
            mandatory_reasons.append("mandatory validator internal error")
        if not hn_structural_gate.passed:
            mandatory_reasons.append("hard-negative structural gate failure")

        if provenance_failures or persistence_record.result in {
            ValidationResult.FAIL,
            ValidationResult.ERROR,
        }:
            for events, context in sequences:
                trace_statuses[context.trace_id] = GroundTruthValidationStatus.VALIDATION_FAILED.value
                for event in events:
                    event_statuses[event.event_id] = GroundTruthValidationStatus.VALIDATION_FAILED.value

        return ValidationReport(
            records=records,
            behavior_model_version=provenance.behavior_model_version,
            generator_version=provenance.generator_software_version,
            generator_git_commit=provenance.generator_git_commit,
            event_schema_version=provenance.event_schema_version,
            feature_schema_version=provenance.feature_schema_version,
            traces_checked=len(sequences),
            events_checked=sum(len(events) for events, _ in sequences),
            trace_ground_truth_statuses=trace_statuses,
            event_ground_truth_statuses=event_statuses,
            validated_positive_event_ids=positive_event_ids,
            event_validated_invariant_ids=event_invariant_ids,
            trace_validated_invariant_ids=trace_invariant_ids,
            event_validated_invariant_evidence=event_invariant_evidence,
            unexpected_normal_trace_failures=unexpected_normal,
            intended_candidates_confirmed=confirmed,
            intended_candidates_not_confirmed=not_confirmed,
            gt_consistency_failures=gt_failures,
            provenance_schema_failures=provenance_failures,
            build_gate_passed=not mandatory_reasons,
            mandatory_failure_reasons=mandatory_reasons,
            hard_negative_structural_gate=hn_structural_gate.to_dict(),
        )

    def _validate_trace(self, events: list[AuthEvent], context: Any) -> _TraceOutcome:
        records = [
            self._check_state_result(events, context),
            self._check_lifecycle(events, context),
            self._check_temporal(events, context),
            self._check_identity(events, context),
            self._check_token_binding(events, context),
            self._check_session_continuity(events, context),
            self._check_authorization(events, context),
        ]
        identity_relation_record = self._check_persistent_authenticated_identity(
            events, context
        )
        records.append(identity_relation_record)
        binding_records = self._check_atomic_token_binding(events, context)
        records.extend(binding_records)
        identity_binding_record = next(
            record for record in binding_records
            if record.validation_id == "c2.4.binding.identity"
        )
        access_records = self._check_atomic_access_invariants(events, context)
        records.extend(access_records)
        detected_access = {
            record.invariant_identifier: record
            for record in access_records
            if record.invariant_identifier and record.result is ValidationResult.PASS
        }
        intended = {
            record.get("anomaly_variant_name")
            for record in context.injection_records
            if record.get("anomaly_variant_name") in supported_synchronized_variants()
        }
        intended.discard(None)
        confirmed_events: set[str] = set()
        confirmed_variants: set[str] = set()
        covered_access_invariants: set[str] = set()
        identity_invariant_covered = False
        covered_binding_checks: set[str] = set()
        event_invariant_ids: dict[str, set[str]] = defaultdict(set)
        event_invariant_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def associate_confirmed(record: ValidationRecord) -> None:
            """Associate an atomic violation with its operation/target event.

            The final reference is the deviating operation event for current
            checks; the complete reference tuple remains explicit evidence.
            """
            if not record.invariant_identifier or not record.event_references:
                return
            observed = record.observed_evidence
            target_event_ids = [
                item.get("event_id")
                for item in observed.get("violations", ())
                if item.get("event_id")
            ]
            if not target_event_ids:
                target_event_ids = [
                    item.get("authentication_event_id") or item.get("use_event_id")
                    for item in observed.get("comparisons", ())
                    if item.get("identity_values_equal") is False
                    or item.get("context_values_equal") is False
                ]
            if not target_event_ids:
                target_event_ids = [record.event_references[-1]]
            for target_event_id in dict.fromkeys(target_event_ids):
                event_invariant_ids[target_event_id].add(record.invariant_identifier)
                linked_request_ids = [
                    item.get("linked_request_event_id")
                    for item in observed.get("violations", ())
                    if item.get("event_id") == target_event_id
                    and item.get("linked_request_event_id")
                ]
                evidence = {
                    "invariant_id": record.invariant_identifier,
                    "validation_id": record.validation_id,
                    "target_event_ids": [target_event_id],
                    "evidence_event_ids": list(dict.fromkeys(
                        linked_request_ids + list(record.event_references)
                    )),
                }
                if evidence not in event_invariant_evidence[target_event_id]:
                    event_invariant_evidence[target_event_id].append(evidence)

        for variant in sorted(intended):
            if variant in SYNCHRONIZED_ACCESS_VARIANTS:
                expected_ids = set(EXACT_VARIANT_TO_INVARIANTS[variant])
                if expected_ids and expected_ids.issubset(detected_access):
                    confirmed_variants.add(variant)
                    covered_access_invariants.update(expected_ids)
                    for invariant_id in expected_ids:
                        atomic = detected_access[invariant_id]
                        associate_confirmed(atomic)
                continue
            if variant in SYNCHRONIZED_IDENTITY_VARIANTS:
                if (
                    identity_relation_record.result is ValidationResult.PASS
                    and identity_relation_record.invariant_identifier
                    in EXACT_VARIANT_TO_INVARIANTS[variant]
                ):
                    confirmed_variants.add(variant)
                    identity_invariant_covered = True
                    associate_confirmed(identity_relation_record)
                continue
            if variant in SYNCHRONIZED_TOKEN_BINDING_VARIANTS:
                if (
                    identity_binding_record.result is ValidationResult.PASS
                    and identity_binding_record.invariant_identifier
                    in EXACT_VARIANT_TO_INVARIANTS[variant]
                ):
                    confirmed_variants.add(variant)
                    covered_binding_checks.add(identity_binding_record.validation_id)
                    associate_confirmed(identity_binding_record)
                continue
            if variant == "nonce_reuse":
                record = self._check_nonce_reuse(events, context)
            elif variant == "token_replay_after_replacement":
                record = self._check_superseded_token_replay(events, context)
            else:
                record = self._check_timestamp_inconsistency(events, context)
            records.append(record)
            if (
                record.result is ValidationResult.PASS
                and record.invariant_identifier in EXACT_VARIANT_TO_INVARIANTS[variant]
            ):
                confirmed_variants.add(variant)
                associate_confirmed(record)

        confirmed_events.update(event_invariant_ids)

        if not intended:
            records.append(self._record(
                "c1.7.supported_candidate.not_applicable",
                ValidationCategory.REPLAY_SINGLE_USE,
                context,
                "supported anomaly evidence is checked when declared",
                {"declared_supported_variants": []},
                ValidationResult.NOT_APPLICABLE,
                "trace declares no supported injected candidate",
            ))

        gt_record = self._check_ground_truth(
            events, context, confirmed_events, confirmed_variants, intended
        )
        records.append(gt_record)
        unexpected_access_violation = bool(
            set(detected_access) - covered_access_invariants
        )
        unexpected_identity_violation = bool(
            identity_relation_record.result is ValidationResult.PASS
            and not identity_invariant_covered
        )
        unexpected_binding_violation = any(
            record.result is ValidationResult.PASS
            and record.validation_id not in covered_binding_checks
            for record in binding_records
        )
        deterministic_failed = any(
            record.validation_category in self._DETERMINISTIC_CATEGORIES
            and record.result in {ValidationResult.FAIL, ValidationResult.ERROR}
            and not (
                record.validation_category is ValidationCategory.AUTHORIZATION_ACCESS
                and record.event_references
                and set(record.event_references).issubset(confirmed_events)
            )
            for record in records
        ) or (
            unexpected_access_violation
            or unexpected_identity_violation
            or unexpected_binding_violation
        )
        unsupported_atomic_records = [
            record for record in records
            if record.invariant_identifier
            and record.result is ValidationResult.UNSUPPORTED
        ]
        unsupported_event_ids: set[str] = set()
        for record in unsupported_atomic_records:
            targets = {
                item.get("event_id")
                or item.get("authentication_event_id")
                or item.get("use_event_id")
                for item in record.observed_evidence.get("underdetermined", ())
            }
            unsupported_event_ids.update(target for target in targets if target)
            if not targets:
                unsupported_event_ids.update(record.event_references[-1:])
        return _TraceOutcome(
            records,
            confirmed_events,
            confirmed_variants,
            set(intended),
            deterministic_failed,
            gt_record.result is ValidationResult.FAIL,
            dict(event_invariant_ids),
            dict(event_invariant_evidence),
            unsupported_event_ids,
            bool(unsupported_atomic_records),
        )

    def _check_state_result(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        successful_auth_seen = False
        validated_token_ids: set[str] = set()
        session_open_ids: set[str] = set()
        for event in events:
            if (
                event.event_type == EventType.AUTHENTICATION_SUCCESS
                and event.result is EventResult.SUCCESS
            ):
                if not event.authenticated_context_active:
                    violations.append({"event_id": event.event_id, "issue": "authentication_success_without_active_context"})
                else:
                    successful_auth_seen = True
            if event.event_type == EventType.AUTHENTICATION_FAILURE:
                if event.result is not EventResult.FAILURE or event.new_state != AuthState.AUTH_FAILED:
                    violations.append({"event_id": event.event_id, "issue": "authentication_failure_result_state_mismatch"})
                if event.authenticated_context_active and not successful_auth_seen:
                    violations.append({"event_id": event.event_id, "issue": "failure_established_authenticated_context"})
            if event.authenticated_context_active and not successful_auth_seen:
                violations.append({"event_id": event.event_id, "issue": "active_auth_context_without_success_evidence"})
            if event.event_type == EventType.TOKEN_VALIDATED and event.result is EventResult.SUCCESS:
                if event.token_id:
                    validated_token_ids.add(event.token_id)
            if event.token_context_active and not validated_token_ids:
                violations.append({"event_id": event.event_id, "issue": "active_token_context_without_validation_success"})
            rejected_superseded_noncurrent = (
                event.event_type == EventType.TOKEN_REJECTED
                and event.token_lineage_status == "superseded"
                and event.token_id != event.current_token_id
            )
            if event.event_type == EventType.TOKEN_REJECTED and (
                event.token_validation_result != "rejected"
                or (event.token_context_active and not rejected_superseded_noncurrent)
            ):
                violations.append({"event_id": event.event_id, "issue": "token_rejection_established_valid_context"})
            if event.event_type == EventType.SESSION_OPENED and event.result is EventResult.SUCCESS:
                if event.protected_session_id:
                    session_open_ids.add(event.protected_session_id)
            if event.protected_session_active and event.protected_session_id not in session_open_ids:
                violations.append({"event_id": event.event_id, "issue": "active_session_without_open_success"})
            if (
                event.event_type in {EventType.SESSION_CLOSED, EventType.DISCONNECT, EventType.TIMEOUT}
                and event.result is EventResult.SUCCESS
                and (
                event.authenticated_context_active or event.token_context_active or event.protected_session_active
                )
            ):
                violations.append({"event_id": event.event_id, "issue": "termination_left_session_local_context_active"})
        if any(
            event.event_type in {EventType.SESSION_CLOSED, EventType.DISCONNECT, EventType.TIMEOUT}
            and event.result is EventResult.SUCCESS
            for event in events
        ) and not context.persistent_enrolled:
            violations.append({"issue": "ordinary_termination_erased_or_lacks_persistent_enrollment"})
        return self._violations_record(
            "c1.7.state_result.supported_subset",
            ValidationCategory.STATE_RESULT,
            "state/transition validity",
            context,
            "failures cannot establish success context; dependent context needs success evidence; termination clears only session-local context",
            violations,
        )

    def _check_lifecycle(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        auth_success_attempts: set[str] = set()
        issued: set[str] = set()
        presented: set[str] = set()
        validated: set[str] = set()
        opened: set[str] = set()
        access_requests: set[str] = set()
        for event in events:
            succeeded = event.result is EventResult.SUCCESS
            if event.event_type == EventType.AUTHENTICATION_SUCCESS and succeeded and event.auth_attempt_id:
                auth_success_attempts.add(event.auth_attempt_id)
            elif event.event_type == EventType.TOKEN_ISSUED and succeeded:
                if event.auth_attempt_id not in auth_success_attempts:
                    violations.append({"event_id": event.event_id, "issue": "token_before_authentication_success"})
                if event.token_id:
                    issued.add(event.token_id)
            elif event.event_type == EventType.TOKEN_PRESENTED and succeeded:
                if event.token_id not in issued:
                    violations.append({"event_id": event.event_id, "issue": "presentation_without_matching_issuance"})
                if event.token_id:
                    presented.add(event.token_id)
            elif event.event_type == EventType.TOKEN_VALIDATED and succeeded:
                if event.token_id not in presented:
                    violations.append({"event_id": event.event_id, "issue": "validation_without_matching_presentation"})
                if event.token_id:
                    validated.add(event.token_id)
            elif event.event_type == EventType.SESSION_OPENED and succeeded:
                if event.token_id not in validated:
                    violations.append({"event_id": event.event_id, "issue": "session_open_without_validated_token"})
                if event.protected_session_id:
                    opened.add(event.protected_session_id)
            elif event.event_type == EventType.ACCESS_REQUEST:
                if succeeded and event.protected_session_id not in opened:
                    violations.append({"event_id": event.event_id, "issue": "access_request_before_session_open"})
                if event.access_request_id:
                    access_requests.add(event.access_request_id)
            elif (
                event.event_type == EventType.RETRY
                and event.previous_state == AuthState.ACCESS_DENIED
                and event.access_request_id
            ):
                # C1.3 represents the new access-decision/request context on
                # the retry event rather than emitting a second ACCESS_REQUEST.
                access_requests.add(event.access_request_id)
            elif (
                event.event_type == EventType.ACCESS_GRANTED
                and event.result is EventResult.SUCCESS
            ) or (
                event.event_type == EventType.ACCESS_DENIED
                and event.authorization_decision == "denied"
            ):
                if event.access_request_id not in access_requests:
                    violations.append({"event_id": event.event_id, "issue": "access_outcome_without_request"})
        return self._violations_record(
            "c1.7.lifecycle_causal.supported_subset",
            ValidationCategory.LIFECYCLE_CAUSAL,
            "lifecycle/causal ordering",
            context,
            "successful security operations have their required causal predecessors",
            violations,
        )

    def _check_temporal(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        for previous, current in zip(events, events[1:]):
            if current.timestamp <= previous.timestamp:
                violations.append({"event_id": current.event_id, "issue": "semantic_time_not_strictly_monotonic", "previous": previous.timestamp, "current": current.timestamp})
        unresolved_failures: list[AuthEvent] = []
        token_issued: dict[str, AuthEvent] = {}
        token_presented: dict[str, AuthEvent] = {}
        sessions_opened: dict[str, AuthEvent] = {}
        renewals: list[AuthEvent] = []
        for event in events:
            if event.result is EventResult.FAILURE:
                unresolved_failures.append(event)
            elif event.event_type == EventType.RETRY:
                if not unresolved_failures or event.timestamp <= unresolved_failures[-1].timestamp:
                    violations.append({"event_id": event.event_id, "issue": "retry_without_earlier_failure"})
                elif event.previous_state == AuthState.AUTH_FAILED:
                    failure = unresolved_failures[-1]
                    if (
                        failure.event_type != EventType.AUTHENTICATION_FAILURE
                        or not failure.auth_attempt_id
                        or not event.auth_attempt_id
                        or failure.auth_attempt_id == event.auth_attempt_id
                    ):
                        violations.append({"event_id": event.event_id, "issue": "authentication_retry_attempt_context_incoherent"})
                unresolved_failures.clear()
            if event.event_type == EventType.TOKEN_ISSUED and event.result is EventResult.SUCCESS and event.token_id:
                token_issued[event.token_id] = event
            elif event.event_type == EventType.TOKEN_PRESENTED and event.result is EventResult.SUCCESS and event.token_id:
                issued = token_issued.get(event.token_id)
                if not issued or event.timestamp <= issued.timestamp:
                    violations.append({"event_id": event.event_id, "issue": "token_presentation_time_precedes_issuance"})
                token_presented[event.token_id] = event
            elif event.event_type == EventType.TOKEN_VALIDATED and event.result is EventResult.SUCCESS and event.token_id:
                presented = token_presented.get(event.token_id)
                if not presented or event.timestamp <= presented.timestamp:
                    violations.append({"event_id": event.event_id, "issue": "token_validation_time_precedes_presentation"})
                if event.token_expiry is None or event.timestamp > event.token_expiry:
                    violations.append({"event_id": event.event_id, "issue": "token_validated_after_or_without_expiry_evidence"})
            elif event.event_type == EventType.SESSION_OPENED and event.result is EventResult.SUCCESS and event.protected_session_id:
                sessions_opened[event.protected_session_id] = event
            elif event.event_type in {EventType.SESSION_CLOSED, EventType.DISCONNECT, EventType.TIMEOUT} and event.protected_session_id:
                opened = sessions_opened.get(event.protected_session_id)
                if not opened or event.timestamp <= opened.timestamp:
                    violations.append({"event_id": event.event_id, "issue": "session_termination_not_after_open"})
            elif event.event_type == EventType.RENEWAL_REQUEST and event.result is EventResult.SUCCESS:
                renewals.append(event)
        if context.renewed_token_issued_at is not None:
            if not renewals or context.renewal_requested_at is None or context.renewed_token_issued_at <= context.renewal_requested_at:
                violations.append({"issue": "renewed_token_not_after_renewal_request"})
        return self._violations_record(
            "c1.7.temporal.supported_subset",
            ValidationCategory.TEMPORAL,
            "temporal/freshness consistency",
            context,
            "authoritative semantic chronology and enforced validity relationships are coherent",
            violations,
        )

    def _check_identity(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        established_authenticated_identity: Optional[str] = None
        for event in events:
            if event.device_id != context.device_id or event.trace_id != context.trace_id or event.scenario_id != context.scenario_id or event.session_id != context.legacy_session_id:
                violations.append({"event_id": event.event_id, "issue": "trace_or_device_identity_context_mismatch"})
            if (
                event.event_type == EventType.AUTHENTICATION_SUCCESS
                and event.result is EventResult.SUCCESS
                and event.authenticated_context_active
            ):
                established_authenticated_identity = event.authenticated_identity
            elif (
                established_authenticated_identity is not None
                and event.authenticated_identity is not None
                and event.authenticated_identity != established_authenticated_identity
            ):
                violations.append({
                    "event_id": event.event_id,
                    "issue": "authenticated_identity_context_drift",
                })
        return self._violations_record(
            "c1.7.identity.supported_subset",
            ValidationCategory.IDENTITY,
            "identity consistency",
            context,
            "represented device, trace, and session identifiers remain structurally coherent; semantic identity comparison is atomic",
            violations,
        )

    def _check_persistent_authenticated_identity(
        self,
        events: list[AuthEvent],
        context: Any,
    ) -> ValidationRecord:
        """Check the scoped persistent-to-authenticated identity relation.

        PASS means a violation was independently reconstructed.  Coherent,
        complete comparisons are NOT_APPLICABLE; missing identity or association
        evidence is UNSUPPORTED.  Injection/scenario metadata is not consulted.
        """
        persistent_identity = getattr(context, "device_id", None)
        declared_attempts = set(getattr(context, "auth_attempt_ids", ()) or ())
        successful = [
            event for event in events
            if event.event_type == EventType.AUTHENTICATION_SUCCESS
            and event.result is EventResult.SUCCESS
            and event.authenticated_context_active
        ]
        interaction_starts = {
            EventType.AUTHENTICATION_REQUEST,
            EventType.RENEWAL_REQUEST,
            EventType.RETRY,
            EventType.CHALLENGE_SENT,
        }
        comparisons: list[dict[str, Any]] = []
        violating: list[AuthEvent] = []
        underdetermined: list[dict[str, Any]] = []
        underdetermined_events: list[AuthEvent] = []

        if not successful:
            return self._record(
                "c2.3.identity.persistent_authenticated",
                ValidationCategory.IDENTITY,
                context,
                "persistent device identity equals authenticated identity in a reliably associated successful authentication context",
                {
                    "profile_scope": "single-device reference profile",
                    "comparisons": [],
                    "underdetermined": [],
                    "request_actor_comparison": "unsupported_by_current_evidence",
                },
                ValidationResult.NOT_APPLICABLE,
                "no successful authenticated context is present",
                invariant_family="identity consistency",
                invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                    "c2.3.identity.persistent_authenticated"
                ],
            )

        for target in successful:
            association = next(
                (
                    event for event in events
                    if event.event_id != target.event_id
                    and event.event_type in interaction_starts
                    and event.auth_attempt_id == target.auth_attempt_id
                    and event.device_id == target.device_id
                    and event.timestamp < target.timestamp
                ),
                None,
            )
            persistent_available = bool(persistent_identity)
            authenticated_available = bool(target.authenticated_identity)
            context_available = bool(target.auth_attempt_id)
            declared_link = bool(
                target.auth_attempt_id and target.auth_attempt_id in declared_attempts
            )
            device_link = bool(
                persistent_available
                and target.device_id == persistent_identity
                and association is not None
            )
            association_reliable = context_available and declared_link and device_link
            item = {
                "persistent_identity": persistent_identity,
                "persistent_identity_source": "SessionContext.device_id backed by PersistentDeviceContext.device_id",
                "authenticated_identity": target.authenticated_identity,
                "authenticated_identity_source": "AuthEvent.authenticated_identity established by successful authentication",
                "authentication_context_id": target.auth_attempt_id,
                "authentication_event_id": target.event_id,
                "device_context_id": target.device_id,
                "association_event_id": association.event_id if association else None,
                "authentication_context_declared": declared_link,
                "context_device_association_reliable": association_reliable,
                "identity_values_equal": (
                    target.authenticated_identity == persistent_identity
                    if persistent_available and authenticated_available else None
                ),
            }
            comparisons.append(item)
            if not (
                persistent_available
                and authenticated_available
                and association_reliable
            ):
                underdetermined.append(item)
                underdetermined_events.append(target)
            elif target.authenticated_identity != persistent_identity:
                violating.append(target)

        if violating:
            result = ValidationResult.PASS
            reason = "persistent-to-authenticated identity contradiction confirmed"
        elif underdetermined:
            result = ValidationResult.UNSUPPORTED
            reason = "identity or authentication-context/device association evidence is insufficient"
        else:
            result = ValidationResult.NOT_APPLICABLE
            reason = "all supported persistent-to-authenticated comparisons are coherent"

        association_refs = tuple(
            item["association_event_id"]
            for item in comparisons
            if item["association_event_id"]
            and any(
                target.auth_attempt_id == item["authentication_context_id"]
                for target in violating
            )
        )
        return self._record(
            "c2.3.identity.persistent_authenticated",
            ValidationCategory.IDENTITY,
            context,
            "persistent device identity equals authenticated identity in a reliably associated successful authentication context",
            {
                "profile_scope": "single-device reference profile",
                "comparisons": comparisons,
                "underdetermined": underdetermined,
                "request_actor_comparison": "unsupported_by_current_evidence",
                "request_actor_reason": (
                    "AccessRequestContext.authenticated_identity is copied from the "
                    "authentication context and is not independent acting identity evidence."
                ),
            },
            result,
            reason,
            invariant_family="identity consistency",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c2.3.identity.persistent_authenticated"
            ],
            event_references=association_refs + tuple(
                target.event_id for target in underdetermined_events
            ) + tuple(
                target.event_id for target in violating
            ),
        )

    def _check_token_binding(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        issued: set[str] = set()
        presented: set[str] = set()
        for event in events:
            if event.event_type == EventType.TOKEN_ISSUED and event.result is EventResult.SUCCESS:
                if not event.authenticated_context_active or not event.token_id:
                    violations.append({"event_id": event.event_id, "issue": "token_issued_without_authenticated_context_or_id"})
                elif event.token_id:
                    issued.add(event.token_id)
            elif event.event_type == EventType.TOKEN_PRESENTED and event.result is EventResult.SUCCESS:
                if event.token_id not in issued:
                    violations.append({"event_id": event.event_id, "issue": "presented_token_not_issued_in_trace"})
                if event.token_id:
                    presented.add(event.token_id)
            elif event.event_type == EventType.TOKEN_VALIDATED and event.result is EventResult.SUCCESS:
                if event.token_id not in presented or event.token_validation_result != "validated":
                    violations.append({"event_id": event.event_id, "issue": "validated_token_not_presented_or_not_marked_validated"})
            if event.event_type == EventType.ACCESS_GRANTED and event.result is EventResult.SUCCESS:
                if not event.token_scope or not event.resource_id or not event.requested_action:
                    violations.append({"event_id": event.event_id, "issue": "grant_lacks_scope_resource_action_evidence"})
                else:
                    try:
                        scope_resource, permission = event.token_scope.split(":", 1)
                        requested_resource = event.resource_id.split("/", 3)[-1].split("/", 1)[0]
                    except (ValueError, AttributeError):
                        violations.append({"event_id": event.event_id, "issue": "malformed_scope_or_resource"})
                    else:
                        if requested_resource != scope_resource or (
                            permission != "readwrite" and permission != event.requested_action
                        ):
                            violations.append({"event_id": event.event_id, "issue": "grant_outside_reference_scope"})
        return self._violations_record(
            "c1.7.credential_token.supported_subset",
            ValidationCategory.CREDENTIAL_TOKEN,
            "credential/token binding",
            context,
            "issued, presented, validated, and used token evidence retains implemented identity/context/scope relationships",
            violations,
        )

    def _check_atomic_token_binding(
        self,
        events: list[AuthEvent],
        context: Any,
    ) -> list[ValidationRecord]:
        """Return independent identity and represented protected-context checks.

        PASS means the applicable binding relation was violated. Missing bound
        identity/context evidence is UNSUPPORTED, never inferred as coherent.
        Scenario and injection metadata are deliberately not consulted.
        """
        issued = {
            event.token_id: event for event in events
            if event.event_type == EventType.TOKEN_ISSUED
            and event.result is EventResult.SUCCESS
            and event.token_id
        }
        identity_use_types = {
            EventType.TOKEN_PRESENTED,
            EventType.TOKEN_VALIDATED,
            EventType.SESSION_OPENED,
            EventType.ACCESS_REQUEST,
            EventType.ACCESS_GRANTED,
            EventType.ACCESS_DENIED,
        }
        identity_uses = [
            event for event in events
            if event.event_type in identity_use_types
            and event.token_id
            and (
                event.result is EventResult.SUCCESS
                or event.event_type == EventType.ACCESS_DENIED
            )
        ]

        identity_comparisons: list[dict[str, Any]] = []
        identity_violations: list[AuthEvent] = []
        identity_unknown: list[dict[str, Any]] = []
        identity_unknown_events: list[AuthEvent] = []
        for use in identity_uses:
            origin = issued.get(use.token_id)
            reliable_link = bool(
                origin
                and origin.timestamp < use.timestamp
                and origin.trace_id == use.trace_id == context.trace_id
                and origin.device_id == use.device_id == context.device_id
            )
            binding_evidence_coherent = bool(
                origin
                and origin.token_bound_identity
                and use.token_bound_identity
                and origin.token_bound_identity == use.token_bound_identity
            )
            authenticated_context_available = bool(
                use.authenticated_context_active and use.authenticated_identity
            )
            item = {
                "token_id": use.token_id,
                "issuance_event_id": origin.event_id if origin else None,
                "use_event_id": use.event_id,
                "use_event_type": use.event_type.value,
                "token_bound_identity": use.token_bound_identity,
                "token_bound_identity_source": "TokenLineageContext.bound_identity",
                "authenticated_identity": use.authenticated_identity,
                "authenticated_identity_source": "authenticated context carried by AuthEvent",
                "token_use_link_reliable": reliable_link,
                "binding_evidence_coherent": binding_evidence_coherent,
                "authenticated_context_available": authenticated_context_available,
                "identity_values_equal": (
                    use.token_bound_identity == use.authenticated_identity
                    if use.token_bound_identity and use.authenticated_identity else None
                ),
            }
            identity_comparisons.append(item)
            if not (
                reliable_link
                and binding_evidence_coherent
                and authenticated_context_available
            ):
                identity_unknown.append(item)
                identity_unknown_events.append(use)
            elif use.token_bound_identity != use.authenticated_identity:
                identity_violations.append(use)

        if identity_violations:
            identity_result = ValidationResult.PASS
            identity_reason = "token-bound and authenticated identity mismatch confirmed"
        elif identity_unknown:
            identity_result = ValidationResult.UNSUPPORTED
            identity_reason = "token binding or authenticated-context evidence is insufficient"
        else:
            identity_result = ValidationResult.NOT_APPLICABLE
            identity_reason = "all applicable token identity-binding comparisons are coherent"

        identity_record = self._record(
            "c2.4.binding.identity",
            ValidationCategory.CREDENTIAL_TOKEN,
            context,
            "token-bound identity equals the authenticated identity of each reliably linked token use",
            {
                "profile_scope": "token presentation/use in authenticated reference-profile context",
                "comparisons": identity_comparisons,
                "underdetermined": identity_unknown,
            },
            identity_result,
            identity_reason,
            invariant_family="credential/token binding",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c2.4.binding.identity"
            ],
            event_references=tuple(
                event.event_id
                for event in identity_unknown_events + identity_violations
            ),
        )

        context_use_types = {
            EventType.SESSION_OPENED,
            EventType.TOKEN_PRESENTED,
            EventType.TOKEN_VALIDATED,
            EventType.ACCESS_REQUEST,
            EventType.ACCESS_GRANTED,
            EventType.ACCESS_DENIED,
        }
        context_uses = [
            event for event in events
            if event.event_type in context_use_types
            and event.token_id
            and event.protected_session_active
            and event.protected_session_id
        ]
        context_comparisons: list[dict[str, Any]] = []
        context_violations: list[AuthEvent] = []
        context_unknown: list[dict[str, Any]] = []
        context_unknown_events: list[AuthEvent] = []
        for use in context_uses:
            origin = issued.get(use.token_id)
            reliable_link = bool(
                origin
                and origin.timestamp < use.timestamp
                and origin.trace_id == use.trace_id == context.trace_id
                and origin.device_id == use.device_id == context.device_id
            )
            bound_context_available = bool(use.token_bound_protected_session_id)
            item = {
                "token_id": use.token_id,
                "issuance_event_id": origin.event_id if origin else None,
                "use_event_id": use.event_id,
                "use_event_type": use.event_type.value,
                "token_bound_protected_session_id": use.token_bound_protected_session_id,
                "active_protected_session_id": use.protected_session_id,
                "binding_source": "TokenLineageContext.protected_session_id",
                "token_use_link_reliable": reliable_link,
                "bound_context_available": bound_context_available,
                "context_values_equal": (
                    use.token_bound_protected_session_id == use.protected_session_id
                    if bound_context_available else None
                ),
            }
            context_comparisons.append(item)
            if not (reliable_link and bound_context_available):
                context_unknown.append(item)
                context_unknown_events.append(use)
            elif use.token_bound_protected_session_id != use.protected_session_id:
                context_violations.append(use)

        if context_violations:
            context_result = ValidationResult.PASS
            context_reason = "represented token/protected-context binding mismatch confirmed"
        elif context_unknown:
            context_result = ValidationResult.UNSUPPORTED
            context_reason = "represented token/protected-context association is insufficient"
        else:
            context_result = ValidationResult.NOT_APPLICABLE
            context_reason = (
                "all applicable represented protected-context bindings are coherent"
            )

        context_record = self._record(
            "c2.4.binding.protected_context",
            ValidationCategory.CREDENTIAL_TOKEN,
            context,
            "where the reference token lineage defines a protected-context binding, it equals the active protected session",
            {
                "profile_scope": (
                    "existing TokenLineageContext association only; no universal "
                    "token/session-binding requirement"
                ),
                "comparisons": context_comparisons,
                "underdetermined": context_unknown,
            },
            context_result,
            context_reason,
            invariant_family="credential/token binding",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c2.4.binding.protected_context"
            ],
            event_references=tuple(
                event.event_id
                for event in context_unknown_events + context_violations
            ),
        )
        return [identity_record, context_record]

    def _check_session_continuity(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        declared_attempts = set(context.auth_attempt_ids)
        event_attempts = {event.auth_attempt_id for event in events if event.auth_attempt_id}
        if not event_attempts.issubset(declared_attempts):
            violations.append({"issue": "event_auth_attempt_not_declared_by_trace", "unexpected": sorted(event_attempts - declared_attempts)})
        session_ids = {event.protected_session_id for event in events if event.protected_session_id}
        if session_ids and session_ids != {context.protected_session_id}:
            violations.append({"issue": "mixed_protected_session_context", "observed": sorted(session_ids)})
        renewal_events = [event for event in events if event.event_type == EventType.RENEWAL_REQUEST and event.result is EventResult.SUCCESS]
        for event in renewal_events:
            if event.auth_attempt_id not in set(context.renewal_auth_attempt_ids):
                violations.append({"event_id": event.event_id, "issue": "renewal_attempt_not_declared"})
            if event.protected_session_id != context.refreshes_protected_session_id:
                violations.append({"event_id": event.event_id, "issue": "renewal_not_linked_to_refreshed_session"})
        return self._violations_record(
            "c1.7.session_continuity.supported_subset",
            ValidationCategory.SESSION_CONTINUITY,
            "session continuity/context",
            context,
            "attempt and protected-session references remain within the intended trace/device context",
            violations,
        )

    def _check_authorization(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        for event in events:
            if event.event_type == EventType.ACCESS_GRANTED and event.result is EventResult.SUCCESS:
                if not event.access_request_id or event.resource_operation_outcome != "success":
                    violations.append({"event_id": event.event_id, "issue": "successful_access_event_missing_request_or_successful_operation_outcome"})
            elif event.event_type == EventType.ACCESS_DENIED:
                if event.authorization_decision != "denied" or event.resource_operation_outcome not in {"denied", "not_executed"}:
                    violations.append({"event_id": event.event_id, "issue": "denial_decision_outcome_mismatch"})
        return self._violations_record(
            "c1.7.authorization_access.supported_subset",
            ValidationCategory.AUTHORIZATION_ACCESS,
            "authorization/access preconditions",
            context,
            "access event/result/outcome structure is coherent; atomic prerequisite semantics are reported separately",
            violations,
        )

    def _check_atomic_access_invariants(
        self,
        events: list[AuthEvent],
        context: Any,
    ) -> list[ValidationRecord]:
        """Return one request-scoped evidence result for each stable access rule.

        In this anomaly-evidence layer PASS means a violation was positively
        reconstructed.  NOT_APPLICABLE means applicable evidence was complete
        and no violation was found (or no protected operation succeeded).
        UNSUPPORTED preserves genuinely incomplete applicability/decision
        evidence instead of treating absence as a missing grant.
        """
        successful = [
            event for event in events
            if event.result is EventResult.SUCCESS
            and event.resource_operation_outcome == "success"
            and event.access_request_id
            and event.event_type == EventType.ACCESS_GRANTED
        ]
        request_events: dict[str, list[AuthEvent]] = defaultdict(list)
        for event in events:
            if event.event_type == EventType.ACCESS_REQUEST and event.access_request_id:
                request_events[event.access_request_id].append(event)

        def linked_request(item: AuthEvent) -> Optional[AuthEvent]:
            return next(
                (
                    request for request in request_events.get(item.access_request_id, ())
                    if request.timestamp < item.timestamp
                    and request.device_id == item.device_id
                    and request.protected_session_id == item.protected_session_id
                    and request.resource_id == item.resource_id
                    and request.requested_action == item.requested_action
                ),
                None,
            )

        def evidence(item: AuthEvent) -> dict[str, Any]:
            return {
                "event_id": item.event_id,
                "access_request_id": item.access_request_id,
                "resource_id": item.resource_id,
                "requested_action": item.requested_action,
                "authenticated_context_active": item.authenticated_context_active,
                "authenticated_identity": item.authenticated_identity,
                "token_id": item.token_id,
                "token_validation_result": item.token_validation_result,
                "token_context_active": item.token_context_active,
                "protected_session_id": item.protected_session_id,
                "protected_session_active": item.protected_session_active,
                "authorization_decision": item.authorization_decision,
                "authorization_applicable": item.authorization_applicable,
                "authorization_evidence_complete": item.authorization_evidence_complete,
                "authorization_authoritative": item.authorization_authoritative,
                "resource_operation_outcome": item.resource_operation_outcome,
                "linked_request_event_id": (
                    linked_request(item).event_id if linked_request(item) else None
                ),
            }

        def atomic_record(
            check_id: str,
            expected: str,
            violating: list[AuthEvent],
            *,
            underdetermined: Optional[list[AuthEvent]] = None,
            scope_note: str = "current reference profile",
        ) -> ValidationRecord:
            unknown = underdetermined or []
            if violating:
                result = ValidationResult.PASS
                reason = "request-scoped invariant violation confirmed"
            elif unknown:
                result = ValidationResult.UNSUPPORTED
                reason = "request-scoped evidence is insufficient to decide the invariant"
            else:
                result = ValidationResult.NOT_APPLICABLE
                reason = "no successful protected operation violates this invariant"
            return self._record(
                check_id,
                ValidationCategory.AUTHORIZATION_ACCESS,
                context,
                expected,
                {
                    "profile_scope": scope_note,
                    "successful_operations_checked": len(successful),
                    "violations": [evidence(item) for item in violating],
                    "underdetermined": [evidence(item) for item in unknown],
                },
                result,
                reason,
                invariant_family="authorization/access preconditions",
                invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[check_id],
                event_references=tuple(
                    item.event_id for item in unknown + violating
                ),
            )

        auth_unknown = [
            item for item in successful
            if item.authentication_required is None or linked_request(item) is None
        ]
        auth_violations = [
            item for item in successful
            if item not in auth_unknown
            and item.authentication_required is True
            and not (
                item.authenticated_context_active
                and item.authenticated_identity == item.device_id
            )
        ]
        credential_unknown = [
            item for item in successful
            if item.credential_required is None or linked_request(item) is None
        ]
        credential_violations = [
            item for item in successful
            if item not in credential_unknown
            and item.credential_required is True
            and not (
                item.token_id
                and item.token_validation_result == "validated"
                and item.token_context_active
            )
        ]
        session_unknown = [
            item for item in successful
            if item.active_session_required is None or linked_request(item) is None
        ]
        session_violations = [
            item for item in successful
            if item not in session_unknown
            and item.active_session_required is True
            and not (item.protected_session_id and item.protected_session_active)
        ]
        grant_applicable = [
            item for item in successful if item.authorization_applicable is True
        ]
        grant_unknown = [
            item for item in successful
            if item.authorization_applicable is None
            or (
                item.authorization_applicable is True
                and not (
                    item.authorization_evidence_complete
                    and item.authorization_authoritative
                    and item.resource_id
                    and item.requested_action
                    and linked_request(item) is not None
                )
            )
        ]
        grant_violations = [
            item for item in grant_applicable
            if item not in grant_unknown
            and item.authorization_decision != "granted"
        ]
        deny_violations = [
            item for item in successful
            if item.authorization_applicable is True
            and item.authorization_evidence_complete
            and item.authorization_authoritative
            and item.authorization_decision == "denied"
            and linked_request(item) is not None
        ]

        return [
            atomic_record(
                "c2.2.access.auth_context",
                "successful protected operation has an applicable valid authenticated context",
                auth_violations,
                underdetermined=auth_unknown,
            ),
            atomic_record(
                "c2.2.access.credential",
                "successful protected operation has its required validated credential context",
                credential_violations,
                underdetermined=credential_unknown,
            ),
            atomic_record(
                "c2.2.access.active_session",
                "successful protected operation has an applicable active protected session",
                session_violations,
                underdetermined=session_unknown,
                scope_note="Behavioral-model v1.1 reference profile only",
            ),
            atomic_record(
                "c2.2.access.authz_grant",
                "complete applicable authorization evidence contains a positive grant for the request resource/action",
                grant_violations,
                underdetermined=grant_unknown,
            ),
            atomic_record(
                "c2.2.access.deny_enforced",
                "authoritative explicit denial prevents successful execution of the associated request",
                deny_violations,
            ),
        ]

    def _check_nonce_reuse(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        groups: dict[str, list[AuthEvent]] = defaultdict(list)
        for event in events:
            if event.nonce and event.auth_attempt_id:
                groups[event.nonce].append(event)
        candidates: list[tuple[AuthEvent, AuthEvent]] = []
        for occurrences in groups.values():
            for index, first in enumerate(occurrences):
                for second in occurrences[index + 1:]:
                    if first.auth_attempt_id != second.auth_attempt_id:
                        candidates.append((first, second))
        record = next(
            item for item in context.injection_records
            if item.get("anomaly_variant_name") == "nonce_reuse"
        )
        linked_pair = next((pair for pair in candidates if pair[0].event_id in set(record.get("source_evidence_event_ids", ())) and pair[1].event_id == next((event.event_id for event in events if event.injection_id == record.get("injection_id")), None)), None)
        valid_scope = record.get("declared_history_scope") == "within_trace_across_authentication_attempts"
        passed = linked_pair is not None and valid_scope
        references = tuple(event.event_id for event in linked_pair) if linked_pair else ()
        return self._record(
            "c1.7.nonce_reuse.within_trace_attempts",
            ValidationCategory.REPLAY_SINGLE_USE,
            context,
            "same nonce is evidenced in distinct authentication attempts and source/reuse linkage is reconstructable in the declared supported scope",
            {
                "repeated_nonce_pairs_across_attempts": [
                    {"source_event_id": first.event_id, "reuse_event_id": second.event_id,
                     "source_auth_attempt_id": first.auth_attempt_id, "reuse_auth_attempt_id": second.auth_attempt_id,
                     "same_material": first.nonce == second.nonce}
                    for first, second in candidates
                ],
                "declared_scope": record.get("declared_history_scope"),
                "linked_pair_found": linked_pair is not None,
            },
            ValidationResult.PASS if passed else ValidationResult.FAIL,
            "observable nonce-reuse evidence confirmed" if passed else "declared nonce reuse is not supported by emitted linked evidence",
            invariant_family="replay/single-use and temporal/freshness consistency",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c1.7.nonce_reuse.within_trace_attempts"
            ],
            event_references=references,
        )

    def _check_superseded_token_replay(
        self,
        events: list[AuthEvent],
        context: Any,
    ) -> ValidationRecord:
        """Reconstruct scoped T1 -> T2 -> later use(T1) without intent metadata."""
        successful_issuance = {
            event.token_id: event for event in events
            if event.event_type == EventType.TOKEN_ISSUED
            and event.result is EventResult.SUCCESS
            and event.token_id
        }
        renewal_attempts = set(context.renewal_auth_attempt_ids)
        successful_renewal_auth = {
            event.auth_attempt_id: event for event in events
            if event.event_type == EventType.AUTHENTICATION_SUCCESS
            and event.result is EventResult.SUCCESS
            and event.auth_attempt_id in renewal_attempts
        }
        opened_by_token: dict[str, list[AuthEvent]] = defaultdict(list)
        for event in events:
            if (
                event.event_type == EventType.SESSION_OPENED
                and event.result is EventResult.SUCCESS
                and event.token_id
                and event.protected_session_id
            ):
                opened_by_token[event.token_id].append(event)

        reconstructed: list[dict[str, Any]] = []
        confirmed: Optional[tuple[AuthEvent, AuthEvent, AuthEvent, AuthEvent, Optional[AuthEvent]]] = None
        for replacement in successful_issuance.values():
            source_id = replacement.supersedes_token_id
            source = successful_issuance.get(source_id) if source_id else None
            renewal_auth = successful_renewal_auth.get(replacement.auth_attempt_id)
            if source is None:
                continue
            for session_open in opened_by_token.get(source.token_id, ()):
                replay_candidates = [
                    event for event in events
                    if event.event_type == EventType.TOKEN_PRESENTED
                    and event.result is EventResult.SUCCESS
                    and event.token_id == source.token_id
                    and event.timestamp > replacement.timestamp
                ]
                for replay in replay_candidates:
                    rejection = next(
                        (
                            event for event in events
                            if event.event_type == EventType.TOKEN_REJECTED
                            and event.token_id == source.token_id
                            and event.timestamp > replay.timestamp
                            and event.protected_session_id == replay.protected_session_id
                        ),
                        None,
                    )
                    same_session = bool(
                        session_open.protected_session_id
                        == replacement.protected_session_id
                        == replay.protected_session_id
                        == context.protected_session_id
                    )
                    distinct = source.token_id != replacement.token_id
                    source_was_current = bool(
                        source.current_token_id == source.token_id
                        and source.token_lineage_status == "current"
                        and session_open.current_token_id == source.token_id
                    )
                    generations_coherent = bool(
                        source.token_generation is not None
                        and replacement.token_generation == source.token_generation + 1
                        and replay.token_generation == source.token_generation
                    )
                    lineage = bool(
                        source_was_current
                        and generations_coherent
                        and replacement.current_token_id == replacement.token_id
                        and replacement.token_lineage_status == "current"
                        and replay.current_token_id == replacement.token_id
                        and replay.token_lineage_status == "superseded"
                        and replay.superseded_by_token_id == replacement.token_id
                        and replay.token_replacement_timestamp == replacement.timestamp
                    )
                    ordered = (
                        source.timestamp < replacement.timestamp < replay.timestamp
                    )
                    successful_replacement = bool(
                        renewal_auth
                        and renewal_auth.timestamp < replacement.timestamp
                        and replacement.auth_attempt_id == renewal_auth.auth_attempt_id
                    )
                    candidate = {
                        "source_token_id": source.token_id,
                        "replacement_token_id": replacement.token_id,
                        "replayed_token_id": replay.token_id,
                        "source_event_id": source.event_id,
                        "session_open_event_id": session_open.event_id,
                        "renewal_success_event_id": renewal_auth.event_id if renewal_auth else None,
                        "replacement_event_id": replacement.event_id,
                        "replay_event_id": replay.event_id,
                        "enforcement_event_id": rejection.event_id if rejection else None,
                        "enforcement_outcome": (
                            rejection.token_validation_result if rejection else "not_observed"
                        ),
                        "same_actual_token_reused": replay.token_id == source.token_id,
                        "tokens_distinct": distinct,
                        "source_was_current": source_was_current,
                        "generations_coherent": generations_coherent,
                        "successful_renewal_replacement": successful_replacement,
                        "same_continuing_protected_session": same_session,
                        "lineage_coherent": lineage,
                        "semantic_order_coherent": ordered,
                    }
                    reconstructed.append(candidate)
                    if distinct and successful_replacement and same_session and lineage and ordered:
                        confirmed = (
                            source, session_open, replacement, replay, rejection
                        )
                        break
                if confirmed:
                    break
            if confirmed:
                break

        references = (
            tuple(event.event_id for event in confirmed[:4])
            if confirmed else ()
        )
        return self._record(
            "c2.1.replay_token.superseded_within_continuing_session",
            ValidationCategory.REPLAY_SINGLE_USE,
            context,
            "actual T1 is presented after successful distinct T2 replacement in the same continuing protected session",
            {
                "reconstructed_candidates": reconstructed,
                "confirmed_candidate_count": 1 if confirmed else 0,
            },
            ValidationResult.PASS if confirmed else ValidationResult.FAIL,
            (
                "superseded-token replay evidence confirmed independently"
                if confirmed else
                "declared superseded-token replay lacks complete equality, replacement, ordering, or session evidence"
            ),
            invariant_family="replay/single-use",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c2.1.replay_token.superseded_within_continuing_session"
            ],
            event_references=references,
        )

    def _check_timestamp_inconsistency(self, events: list[AuthEvent], context: Any) -> ValidationRecord:
        record = next(
            item for item in context.injection_records
            if item.get("anomaly_variant_name") == "timestamp_inconsistency"
        )
        source_ids = set(record.get("source_evidence_event_ids", ()))
        source = next((event for event in events if event.event_id in source_ids), None)
        target = next((event for event in events if event.injection_id == record.get("injection_id")), None)
        source_observed = (
            source.observed_timestamp if source and source.observed_timestamp is not None
            else source.timestamp if source else None
        )
        target_observed = (
            target.observed_timestamp if target and target.observed_timestamp is not None
            else target.timestamp if target else None
        )
        semantic_coherent = bool(source and target and source.timestamp < target.timestamp)
        observed_violation = bool(
            source_observed is not None and target_observed is not None
            and target_observed < source_observed
        )
        targeted_relation = bool(
            target and target.targeted_temporal_relationship
            == "declared_event_time_after_predecessor"
        )
        passed = semantic_coherent and observed_violation and targeted_relation
        references = tuple(
            event.event_id for event in (source, target) if event is not None
        )
        return self._record(
            "c1.7.timestamp_inconsistency.causal_predecessor",
            ValidationCategory.TIMESTAMP_EVIDENCE,
            context,
            "semantic predecessor order remains coherent while observed target time violates the declared after-predecessor relation",
            {
                "source_semantic_timestamp": source.timestamp if source else None,
                "target_semantic_timestamp": target.timestamp if target else None,
                "source_observed_timestamp": source_observed,
                "target_observed_timestamp": target_observed,
                "semantic_order_coherent": semantic_coherent,
                "observed_order_violated": observed_violation,
                "targeted_relation_present": targeted_relation,
            },
            ValidationResult.PASS if passed else ValidationResult.FAIL,
            "observable timestamp violation confirmed" if passed else "declared timestamp candidate is not supported by emitted temporal evidence",
            invariant_family="temporal/freshness consistency",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c1.7.timestamp_inconsistency.causal_predecessor"
            ],
            event_references=references,
        )

    def _check_ground_truth(
        self,
        events: list[AuthEvent],
        context: Any,
        confirmed_event_ids: set[str],
        confirmed_variants: set[str],
        intended_variants: set[str],
    ) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        positive_events = {event.event_id for event in events if event.anomaly_label}
        if positive_events != confirmed_event_ids:
            violations.append({"issue": "event_gt_does_not_equal_validated_deviation_evidence", "positive_event_ids": sorted(positive_events), "validated_event_ids": sorted(confirmed_event_ids)})
        expected_trace_positive = bool(confirmed_variants)
        if bool(context.observable_anomaly) != expected_trace_positive:
            violations.append({"issue": "trace_gt_not_supported_by_validated_evidence", "trace_gt": bool(context.observable_anomaly), "validated_positive": expected_trace_positive})
        if bool(context.observable_violation_candidate) != expected_trace_positive:
            violations.append({"issue": "candidate_status_not_supported_by_validated_evidence"})
        if context.attack_scenario_intent and not confirmed_variants and context.observable_anomaly:
            violations.append({"issue": "scenario_intent_only_trace_labeled_anomalous"})
        if intended_variants != confirmed_variants:
            violations.append({"issue": "supported_intended_candidate_not_confirmed", "intended": sorted(intended_variants), "confirmed": sorted(confirmed_variants)})
        return self._violations_record(
            "c1.7.ground_truth.consistency",
            ValidationCategory.GROUND_TRUTH,
            "ground-truth consistency",
            context,
            "event/trace GT follows independently validated observable evidence, never intent or injection status alone",
            violations,
        )

    def _validate_cross_trace_persistence(self, sequences: list[tuple[list[AuthEvent], Any]]) -> ValidationRecord:
        violations: list[dict[str, Any]] = []
        enrolled_by_device: dict[str, bool] = defaultdict(bool)
        for _events, context in sequences:
            if enrolled_by_device[context.device_id] and not context.persistent_enrolled:
                violations.append({"trace_id": context.trace_id, "device_id": context.device_id, "issue": "persistent_enrollment_regressed_across_traces"})
            enrolled_by_device[context.device_id] = enrolled_by_device[context.device_id] or context.persistent_enrolled
        return ValidationRecord(
            validation_id="c1.7.persistence.cross_trace_enrollment",
            validation_category=ValidationCategory.SESSION_CONTINUITY,
            invariant_family="session continuity/context",
            invariant_identifier=EXACT_VALIDATOR_CHECK_TO_INVARIANT[
                "c1.7.persistence.cross_trace_enrollment"
            ],
            trace_id=None,
            device_id=None,
            protected_session_id=None,
            event_references=(),
            expected_condition="once observed enrolled, ordinary later traces for that device retain enrollment",
            observed_evidence={"violations": violations},
            result=ValidationResult.FAIL if violations else ValidationResult.PASS,
            reason="persistent enrollment regression found" if violations else "cross-trace enrollment persistence confirmed",
        )

    def _validate_schema_provenance(
        self,
        sequences: list[tuple[list[AuthEvent], Any]],
        provenance: GenerationProvenance,
        field_manifest: Optional[dict[str, Any]],
        dataset_manifest: Optional[dict[str, Any]],
    ) -> ValidationRecord:
        violations: list[str] = []
        required_generation = {
            "behavior_model_version": provenance.behavior_model_version,
            "generator_version": provenance.generator_software_version,
            "generator_git_commit": provenance.generator_git_commit,
            "generator_git_dirty": provenance.generator_git_dirty,
            "generator_git_state_scope": provenance.generator_git_state_scope,
            "event_schema_version": provenance.event_schema_version,
            "feature_schema_version": provenance.feature_schema_version,
            "event_ground_truth_schema_version": provenance.event_ground_truth_schema_version,
            "trace_ground_truth_schema_version": provenance.trace_ground_truth_schema_version,
            "validation_schema_version": provenance.validation_schema_version,
            "field_manifest_version": provenance.field_manifest_version,
            "dataset_interface_version": provenance.dataset_interface_version,
            "dataset_manifest_version": provenance.dataset_manifest_version,
            "dataset_version": provenance.dataset_version,
            "invariant_registry_version": provenance.invariant_registry_version,
            "run_id": provenance.run_id,
            "configuration_version": provenance.configuration_version,
            "configuration_snapshot_reference": provenance.configuration_snapshot_reference,
            "configuration_snapshot": provenance.configuration_snapshot,
            "randomness_contract": provenance.randomness_contract,
            "hard_negative_capability_contract": provenance.hard_negative_capability_contract,
        }
        for name, value in required_generation.items():
            if value is None or value == "" or value == {}:
                violations.append(f"missing generation provenance: {name}")
        if provenance.behavior_model_version != BEHAVIOR_MODEL_VERSION:
            violations.append("behavior-model version does not match validator contract")
        if not provenance.configuration_snapshot_reference.startswith("sha256:"):
            violations.append("configuration snapshot reference is not content-addressed")
        seed = provenance.configuration_snapshot.get("simulation", {}).get("random_seed")
        if seed is None:
            violations.append("missing randomness master seed")
        if not field_manifest or not field_manifest.get("fields"):
            violations.append("missing field manifest")
        elif any(
            not item.get("field_name")
            or not item.get("export_view")
            or not item.get("role")
            or not item.get("availability")
            for item in field_manifest["fields"]
        ):
            violations.append("field manifest contains incomplete field definitions")
        if not dataset_manifest:
            violations.append("missing dataset manifest")
        else:
            if (
                "generator_git_dirty" not in dataset_manifest
                or dataset_manifest.get("generator_git_dirty") is None
            ):
                violations.append("missing dataset manifest field: generator_git_dirty")
            for name in (
                "dataset_interface_version", "behavior_model_version",
                "event_schema_version", "feature_schema_version", "run_id",
                "invariant_registry_version", "ground_truth_schema_version",
                "event_ground_truth_schema_version",
                "trace_ground_truth_schema_version",
                "manifest_version", "field_manifest_version", "dataset_version",
                "validation_schema_version", "generator_software_version",
                "generator_git_commit", "generator_git_state_scope",
                "configuration_version", "configuration_snapshot_reference",
                "randomness_contract",
                "hard_negative_capability_contract",
            ):
                if not dataset_manifest.get(name):
                    violations.append(f"missing dataset manifest field: {name}")
            for name, expected in (
                ("behavior_model_version", provenance.behavior_model_version),
                ("event_schema_version", provenance.event_schema_version),
                ("feature_schema_version", provenance.feature_schema_version),
                ("run_id", provenance.run_id),
                ("invariant_registry_version", INVARIANT_REGISTRY_VERSION),
                ("ground_truth_schema_version", GROUND_TRUTH_SCHEMA_VERSION),
                ("event_ground_truth_schema_version", EVENT_GROUND_TRUTH_SCHEMA_VERSION),
                ("trace_ground_truth_schema_version", TRACE_GROUND_TRUTH_SCHEMA_VERSION),
                ("manifest_version", provenance.dataset_manifest_version),
                ("field_manifest_version", provenance.field_manifest_version),
                ("dataset_interface_version", provenance.dataset_interface_version),
                ("dataset_version", provenance.dataset_version),
                ("validation_schema_version", provenance.validation_schema_version),
                ("generator_software_version", provenance.generator_software_version),
                ("generator_git_commit", provenance.generator_git_commit),
                ("generator_git_dirty", provenance.generator_git_dirty),
                ("generator_git_state_scope", provenance.generator_git_state_scope),
                ("configuration_version", provenance.configuration_version),
                ("configuration_snapshot_reference", provenance.configuration_snapshot_reference),
                ("randomness_contract", provenance.randomness_contract),
                ("hard_negative_capability_contract", provenance.hard_negative_capability_contract),
            ):
                if dataset_manifest.get(name) != expected:
                    violations.append(f"dataset manifest mismatch: {name}")
        trace_ids = [context.trace_id for _, context in sequences]
        if len(trace_ids) != len(set(trace_ids)):
            violations.append("trace identifiers are not unique")
        event_ids = [event.event_id for events, _ in sequences for event in events]
        if len(event_ids) != len(set(event_ids)):
            violations.append("event identifiers are not unique")
        for events, context in sequences:
            if context.generation_profile != SYNCHRONIZED_GENERATION_PROFILE:
                violations.append(f"trace {context.trace_id}: non-synchronized generation profile")
            for name in ("trace_id", "scenario_id", "device_id", "legacy_session_id"):
                if not getattr(context, name, None):
                    violations.append(f"trace context missing {name}")
            for event in events:
                for name in ("event_id", "trace_id", "scenario_id", "device_id", "session_id"):
                    if not getattr(event, name, None):
                        violations.append(f"event {event.event_id}: missing {name}")
                if event.auth_attempt_id and event.auth_attempt_id not in set(context.auth_attempt_ids):
                    violations.append(f"event {event.event_id}: auth attempt absent from trace provenance")
                if event.event_type in {
                    EventType.AUTHENTICATION_REQUEST,
                    EventType.CHALLENGE_SENT,
                    EventType.NONCE_RECEIVED,
                    EventType.RESPONSE_SENT,
                    EventType.AUTHENTICATION_SUCCESS,
                    EventType.AUTHENTICATION_FAILURE,
                    EventType.TOKEN_ISSUED,
                    EventType.TOKEN_PRESENTED,
                    EventType.TOKEN_VALIDATED,
                    EventType.TOKEN_REJECTED,
                    EventType.RENEWAL_REQUEST,
                } and not event.auth_attempt_id:
                    violations.append(f"event {event.event_id}: missing applicable auth attempt id")
                if (
                    event.result is EventResult.SUCCESS
                    and event.event_type in {
                        EventType.SESSION_OPENED,
                        EventType.SESSION_CLOSED,
                        EventType.ACCESS_REQUEST,
                        EventType.ACCESS_GRANTED,
                    }
                    and not event.protected_session_id
                ):
                    violations.append(f"event {event.event_id}: missing applicable protected session id")
            injection_ids = {
                record.get("injection_id") for record in context.injection_records
                if record.get("injection_id")
            }
            for record in context.injection_records:
                for name in (
                    "injection_id", "anomaly_variant_name", "trace_id", "device_id",
                    "session_id", "declared_history_scope", "validation_status",
                ):
                    if not record.get(name):
                        violations.append(f"trace {context.trace_id}: injection record missing {name}")
                variant = record.get("anomaly_variant_name")
                expected_ids = EXACT_VARIANT_TO_INVARIANTS.get(variant)
                if expected_ids and tuple(record.get("canonical_invariant_ids", ())) != expected_ids:
                    violations.append(
                        f"trace {context.trace_id}: supported injection invariant IDs do not match registry"
                    )
            for event in events:
                if event.injection_id and event.injection_id not in injection_ids:
                    violations.append(f"event {event.event_id}: injection id lacks trace provenance")
            if context.observable_violation_candidate and not context.injection_records:
                violations.append(f"trace {context.trace_id}: candidate lacks injection provenance")
        return ValidationRecord(
            validation_id="c1.7.schema_provenance.mandatory",
            validation_category=ValidationCategory.SCHEMA_PROVENANCE,
            invariant_family=None,
            invariant_identifier=None,
            trace_id=None,
            device_id=None,
            protected_session_id=None,
            event_references=(),
            expected_condition="mandatory synchronized schema, run, configuration, randomness, identifier, and injection provenance is complete",
            observed_evidence={"violations": violations},
            result=ValidationResult.FAIL if violations else ValidationResult.PASS,
            reason="mandatory metadata incomplete" if violations else "mandatory metadata complete",
        )

    def _validate_hard_negative_candidates(
        self,
        sequences: list[tuple[list[AuthEvent], Any]],
        outcomes: dict[str, _TraceOutcome],
        hard_negative_trace_ids: Optional[Iterable[str]],
    ) -> list[ValidationRecord]:
        requested = set(hard_negative_trace_ids or ())
        if not requested:
            return [ValidationRecord(
                validation_id="c1.7.hard_negative.not_applicable",
                validation_category=ValidationCategory.HARD_NEGATIVE,
                invariant_family=None,
                invariant_identifier=None,
                trace_id=None,
                device_id=None,
                protected_session_id=None,
                event_references=(),
                expected_condition="declared hard-negative candidate has no supported deterministic violation",
                observed_evidence={"candidate_count": 0},
                result=ValidationResult.NOT_APPLICABLE,
                reason="no hard-negative candidates declared",
            )]
        contexts = {context.trace_id: context for _, context in sequences}
        result: list[ValidationRecord] = []
        for trace_id in sorted(requested):
            context = contexts.get(trace_id)
            outcome = outcomes.get(trace_id)
            valid = bool(outcome and not outcome.deterministic_failed and not outcome.confirmed_variants and not outcome.gt_failed)
            result.append(self._record(
                "c1.7.hard_negative.candidate_validity",
                ValidationCategory.HARD_NEGATIVE,
                context,
                "candidate has no supported deterministic or observable invariant violation",
                {"trace_found": context is not None, "validated_variants": sorted(outcome.confirmed_variants) if outcome else [], "deterministic_failed": outcome.deterministic_failed if outcome else None},
                ValidationResult.PASS if valid else ValidationResult.FAIL,
                "hard-negative candidate remains semantically valid" if valid else "hard-negative candidate is missing or violates a supported check",
            ))
        return result

    def _unsupported_records(self) -> list[ValidationRecord]:
        unsupported = {
            "historical_replay_token": "historical token-age/backdating scenario is not the C2.1 superseded-token invariant",
            "duplicate_sequence": "semantic subsequence duplication not implemented",
            "abnormal_failure_rate": "history window and threshold/baseline unresolved",
            "abnormal_renewal": "behavioral history and baseline unresolved",
            "impersonation": "composite scenario lacks decomposed semantic variants",
            "identity_token_mismatch": "composite scenario lacks decomposed semantic variants",
            "authenticated_request_actor_identity_comparison": (
                "no independent request-level ActingIdentity(R) is represented; "
                "the request authenticated_identity field is copied from its authentication context"
            ),
            "access_without_auth": "composite scenario lacks explicit violated prerequisite variant",
            "cross_session_replay_or_nonce": "cross-session material history not implemented",
            "statistical_behavioral_baselines": "statistical semantics not established",
            "global_concurrency_invariants": "global concurrency not implemented",
            "operational_trace_validation": "operational trace support not implemented",
        }
        return [ValidationRecord(
            validation_id=f"c1.7.unsupported.{name}",
            validation_category=ValidationCategory.UNSUPPORTED_CAPABILITY,
            invariant_family=None,
            invariant_identifier=None,
            trace_id=None,
            device_id=None,
            protected_session_id=None,
            event_references=(),
            expected_condition="validator must not claim support outside the synchronized subset",
            observed_evidence={"capability": name},
            result=ValidationResult.UNSUPPORTED,
            reason=reason,
        ) for name, reason in unsupported.items()]

    def _violations_record(
        self,
        validation_id: str,
        category: ValidationCategory,
        invariant_family: Optional[str],
        context: Any,
        expected: str,
        violations: list[dict[str, Any]],
    ) -> ValidationRecord:
        return self._record(
            validation_id,
            category,
            context,
            expected,
            {"violations": violations},
            ValidationResult.FAIL if violations else ValidationResult.PASS,
            "evidence violates expected condition" if violations else "applicable evidence satisfies expected condition",
            invariant_family=invariant_family,
            event_references=tuple(
                item["event_id"] for item in violations if item.get("event_id")
            ),
        )

    @staticmethod
    def _record(
        validation_id: str,
        category: ValidationCategory,
        context: Any,
        expected: str,
        observed: dict[str, Any],
        result: ValidationResult,
        reason: str,
        *,
        invariant_family: Optional[str] = None,
        invariant_identifier: Optional[str] = None,
        event_references: tuple[str, ...] = (),
    ) -> ValidationRecord:
        return ValidationRecord(
            validation_id=validation_id,
            validation_category=category,
            invariant_family=invariant_family,
            invariant_identifier=invariant_identifier,
            trace_id=getattr(context, "trace_id", None),
            device_id=getattr(context, "device_id", None),
            protected_session_id=getattr(context, "protected_session_id", None),
            event_references=event_references,
            expected_condition=expected,
            observed_evidence=observed,
            result=result,
            reason=reason,
        )


def validate_synchronized_dataset(
    sequences: list[tuple[list[AuthEvent], Any]],
    *,
    provenance: GenerationProvenance,
    field_manifest: Optional[dict[str, Any]],
    dataset_manifest: Optional[dict[str, Any]],
    hard_negative_trace_ids: Optional[Iterable[str]] = None,
    hard_negative_qualification_records: Optional[
        Iterable[HardNegativeQualificationRecord]
    ] = None,
) -> ValidationReport:
    sequence_pairs = tuple(sequences)
    qualification_records = tuple(hard_negative_qualification_records or ())
    report = DatasetValidator().validate(
        sequence_pairs,
        provenance=provenance,
        field_manifest=field_manifest,
        dataset_manifest=dataset_manifest,
        hard_negative_trace_ids=hard_negative_trace_ids,
        hard_negative_qualification_records=qualification_records,
    )
    if not qualification_records:
        return report

    # HN.1 structural checks alone cannot detect a stale/fabricated qualified
    # record whose event IDs still resolve but whose values have changed.
    # Re-run frozen HN.2 on the final emitted evidence before accepting the
    # dataset build. This changes no anomaly or HN semantics.
    from simulator.hard_negative_qualification import qualify_hard_negative

    pairs_by_trace = {context.trace_id: (events, context) for events, context in sequence_pairs}
    requalification_errors: list[str] = []
    for record in qualification_records:
        if not isinstance(record, HardNegativeQualificationRecord):
            continue  # already rejected by the structural gate
        pair = pairs_by_trace.get(record.trace_id)
        if pair is None:
            continue  # already rejected as an unknown trace
        actual = qualify_hard_negative(*pair, record, report)
        if actual != record:
            requalification_errors.append(
                f"{record.hard_negative_candidate_id}: supplied HN qualification "
                f"disagrees with final evidence ({record.qualification_status.value} "
                f"versus {actual.qualification_status.value})"
            )
    if requalification_errors:
        previous = report.hard_negative_structural_gate
        report.hard_negative_structural_gate = {
            **previous,
            "passed": False,
            "errors": tuple((*previous.get("errors", ()), *requalification_errors)),
        }
        report.build_gate_passed = False
        report.mandatory_failure_reasons.append("hard-negative semantic requalification failure")
    return report
