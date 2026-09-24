"""HN.2 positive-normal qualification tests using controlled semantic fixtures."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest

from simulator.event_model import AuthEvent, AuthState, EventResult, EventType
from simulator.hard_negative import (
    CandidateStatus,
    ConstructionStatus,
    HARD_NEGATIVE_PRIVILEGED_FIELD_NAMES,
    HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
    HN_CHECK_BY_ID,
    HN_VARIANT_BY_ID,
    HardNegativeQualificationRecord,
    HardNegativeRunSummary,
    QualificationStatus,
    hard_negative_capability_report,
    validate_hard_negative_structure,
)
from simulator.hard_negative_qualification import (
    qualification_engine_report,
    qualify_hard_negative,
)
from simulator.data.dataset_contract import field_names_for_view
from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


TRACE_ID = "hn2-trace"
SCENARIO_ID = "hn2-scenario"
DEVICE_ID = "hn2-device"
SESSION_ID = "hn2-protected-session"


def event(event_id, event_type, timestamp, **overrides):
    values = {
        "event_type": event_type,
        "device_id": DEVICE_ID,
        "gateway_id": "gateway",
        "auth_server_id": "auth",
        "previous_state": AuthState.ENROLLED,
        "new_state": AuthState.ENROLLED,
        "result": EventResult.SUCCESS,
        "event_id": event_id,
        "timestamp": timestamp,
        "observed_timestamp": timestamp,
        "trace_id": TRACE_ID,
        "scenario_id": SCENARIO_ID,
        "session_id": TRACE_ID,
    }
    values.update(overrides)
    return AuthEvent(**values)


def context(*, attempts=(), observable_anomaly=False, **overrides):
    values = {
        "device_id": DEVICE_ID,
        "trace_id": TRACE_ID,
        "scenario_id": SCENARIO_ID,
        "auth_attempt_ids": tuple(attempts),
        "observable_anomaly": observable_anomaly,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def validator_report(check_id, *, result=ValidationResult.NOT_APPLICABLE, comparison=None, trace_status="validated_normal", invariants=()):
    records = []
    for validator_id in HN_CHECK_BY_ID[check_id].validator_check_ids:
        evidence = {"comparisons": [comparison]} if comparison and validator_id == "c2.3.identity.persistent_authenticated" else {}
        records.append(SimpleNamespace(
            validation_id=validator_id,
            trace_id=TRACE_ID,
            result=result,
            invariant_identifier=next(
                (
                    invariant
                    for invariant in HN_VARIANT_BY_ID[HN_CHECK_BY_ID[check_id].variant_id].invariant_ids
                ),
                None,
            ),
            observed_evidence=evidence,
        ))
    return SimpleNamespace(
        records=records,
        trace_ground_truth_statuses={TRACE_ID: trace_status},
        trace_validated_invariant_ids={TRACE_ID: tuple(invariants)},
    )


def candidate(check_id, events, *, boundary=None, construction=ConstructionStatus.APPLIED, targets=None, evidence=None):
    check = HN_CHECK_BY_ID[check_id]
    variant = HN_VARIANT_BY_ID[check.variant_id]
    target_ids = tuple(targets if targets is not None else [events[-1].event_id])
    evidence_ids = tuple(evidence if evidence is not None else [item.event_id for item in events])
    return HardNegativeQualificationRecord(
        hard_negative_candidate_id=f"candidate-{check.variant_id}",
        trace_id=TRACE_ID,
        scenario_id=SCENARIO_ID,
        device_id=DEVICE_ID,
        candidate_status=CandidateStatus.REQUESTED,
        primary_intended_family_id=check.family_id,
        intended_variant_id=check.variant_id,
        intended_boundary_invariant_ids=variant.invariant_ids,
        construction_status=construction,
        construction_parameters={"fixture": "explicit-HN.2-construction"},
        boundary_parameters=boundary or {},
        target_event_ids=target_ids,
        evidence_event_ids=evidence_ids,
    )


def access_events(**operation_overrides):
    request = event(
        "access-request", EventType.ACCESS_REQUEST, 10.0,
        access_request_id="request-1", protected_session_id=SESSION_ID,
        resource_id="telemetry/device", requested_action="read",
    )
    values = {
        "result": EventResult.FAILURE,
        "access_request_id": "request-1",
        "protected_session_id": SESSION_ID,
        "resource_id": "telemetry/device",
        "requested_action": "read",
        "resource_operation_outcome": "denied",
        "authorization_decision": "denied",
    }
    values.update(operation_overrides)
    operation_type = EventType.ACCESS_GRANTED if values["resource_operation_outcome"] == "success" else EventType.ACCESS_DENIED
    return [request, event("access-outcome", operation_type, 11.0, **values)]


def positive_fixture(check_id):
    if check_id == "hn.v1.nonce.fresh_cross_attempt":
        events = [
            event("nonce-a", EventType.NONCE_RECEIVED, 10, nonce="nonce-1", auth_attempt_id="attempt-1"),
            event("retry", EventType.RETRY, 10.5, auth_attempt_id="attempt-2"),
            event("nonce-b", EventType.NONCE_RECEIVED, 11, nonce="nonce-2", auth_attempt_id="attempt-2"),
        ]
        return events, context(attempts=("attempt-1", "attempt-2")), candidate(check_id, events, boundary={"attempt_relationship": "retry"}), validator_report(check_id)
    if check_id == "hn.v1.nonce.same_attempt_recurrence":
        events = [
            event("challenge", EventType.CHALLENGE_SENT, 10, nonce="nonce-1", auth_attempt_id="attempt-1"),
            event("nonce", EventType.NONCE_RECEIVED, 11, nonce="nonce-1", auth_attempt_id="attempt-1"),
        ]
        return events, context(attempts=("attempt-1",)), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.token.current_repeated_use":
        events = [
            event("present-1", EventType.TOKEN_PRESENTED, 10, token_id="T1", current_token_id="T1", token_lineage_status="current", protected_session_id=SESSION_ID, token_expiry=50),
            event("present-2", EventType.TOKEN_PRESENTED, 11, token_id="T1", current_token_id="T1", token_lineage_status="current", protected_session_id=SESSION_ID, token_expiry=50),
        ]
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.token.last_use_before_replacement":
        events = [
            event("present-t1", EventType.TOKEN_PRESENTED, 10, token_id="T1", current_token_id="T1", token_lineage_status="current", protected_session_id=SESSION_ID, token_expiry=50),
            event("issue-t2", EventType.TOKEN_ISSUED, 11, token_id="T2", current_token_id="T2", token_lineage_status="current", supersedes_token_id="T1", protected_session_id=SESSION_ID),
        ]
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.token.replacement_token_use":
        events = [
            event("issue-t2", EventType.TOKEN_ISSUED, 10, token_id="T2", current_token_id="T2", token_lineage_status="current", supersedes_token_id="T1", protected_session_id=SESSION_ID),
            event("present-t2", EventType.TOKEN_PRESENTED, 11, token_id="T2", current_token_id="T2", token_lineage_status="current", protected_session_id=SESSION_ID, token_expiry=50),
        ]
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.time.positive_causal_margin":
        events = [
            event("predecessor", EventType.DISCOVERY, 10, observed_timestamp=10.0),
            event("target", EventType.GATEWAY_ADVERTISEMENT, 11, observed_timestamp=10.1),
        ]
        return events, context(), candidate(check_id, events, boundary={"epsilon_hn": 0.2}), validator_report(check_id)
    if check_id == "hn.v1.access.missing_auth_enforced":
        events = access_events(authentication_required=True, authenticated_context_active=False)
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.access.invalid_credential_enforced":
        events = access_events(credential_required=True, token_id="T1", token_context_active=False, token_validation_result="rejected")
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.access.inactive_session_enforced":
        events = access_events(active_session_required=True, protected_session_active=False)
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.access.no_grant_enforced":
        events = access_events(authorization_applicable=True, authorization_evidence_complete=True, authorization_authoritative=True, authorization_decision="denied")
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.access.explicit_deny_enforced":
        events = access_events(authorization_applicable=True, authorization_evidence_complete=True, authorization_authoritative=True, authorization_decision="denied")
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.authz.valid_context_legitimate_denial":
        events = access_events(
            authenticated_context_active=True, authenticated_identity=DEVICE_ID,
            token_id="T1", token_context_active=True, token_validation_result="validated",
            protected_session_active=True, authorization_applicable=True,
            authorization_evidence_complete=True, authorization_authoritative=True,
            authorization_decision="denied",
        )
        return events, context(), candidate(check_id, events), validator_report(check_id)
    if check_id == "hn.v1.identity.persistent_authenticated_coherent":
        events = [
            event("auth-request", EventType.AUTHENTICATION_REQUEST, 10, auth_attempt_id="attempt-1"),
            event("auth-success", EventType.AUTHENTICATION_SUCCESS, 11, auth_attempt_id="attempt-1", authenticated_context_active=True, authenticated_identity=DEVICE_ID),
        ]
        comparison = {
            "authentication_event_id": "auth-success",
            "context_device_association_reliable": True,
            "persistent_identity_source": "PersistentDeviceContext.device_id",
            "authenticated_identity_source": "successful authentication result",
        }
        return events, context(attempts=("attempt-1",)), candidate(check_id, events, boundary={"boundary_targeted": True}), validator_report(check_id, comparison=comparison)
    if check_id == "hn.v1.binding.identity_coherent":
        events = [
            event("issue", EventType.TOKEN_ISSUED, 10, token_id="T1", token_bound_identity=DEVICE_ID),
            event("use", EventType.TOKEN_PRESENTED, 11, token_id="T1", token_bound_identity=DEVICE_ID, authenticated_identity=DEVICE_ID),
        ]
        return events, context(), candidate(check_id, events, boundary={"boundary_targeted": True}), validator_report(check_id)
    if check_id == "hn.v1.binding.protected_context_coherent":
        events = [
            event("issue", EventType.TOKEN_ISSUED, 10, token_id="T1", token_bound_protected_session_id=SESSION_ID),
            event("use", EventType.TOKEN_PRESENTED, 11, token_id="T1", token_bound_protected_session_id=SESSION_ID, protected_session_id=SESSION_ID, protected_session_active=True),
        ]
        return events, context(), candidate(check_id, events, boundary={"boundary_targeted": True}), validator_report(check_id)
    raise AssertionError(check_id)


class HN2PositiveQualificationTests(unittest.TestCase):
    def test_all_fifteen_checks_qualify_only_from_positive_event_evidence(self):
        self.assertEqual(15, len(HN_CHECK_BY_ID))
        for check_id in HN_CHECK_BY_ID:
            with self.subTest(check_id=check_id):
                events, ctx, record, report = positive_fixture(check_id)
                result = qualify_hard_negative(events, ctx, record, report)
                self.assertEqual(QualificationStatus.QUALIFIED, result.qualification_status)
                self.assertTrue(result.normal_side_predicates)
                self.assertTrue(all(item["satisfied"] for item in result.normal_side_predicates.values()))
                self.assertTrue(result.boundary_proximity_evidence["boundary_reached"]["satisfied"])
                self.assertEqual((check_id,), result.qualification_check_ids)
                self.assertEqual("validated_normal", result.trace_ground_truth_status)
                self.assertFalse(result.trace_observable_anomaly)
                self.assertEqual((), result.structural_errors())

    def test_capability_reports_qualification_and_explicit_hn3_generation(self):
        capability = hard_negative_capability_report()
        engine = qualification_engine_report()
        self.assertTrue(capability["semantic_qualification_available"])
        self.assertTrue(capability["generation_available"])
        self.assertTrue(capability["explicit_single_family_generation_available"])
        self.assertFalse(capability["random_sampling_available"])
        self.assertEqual(HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION, engine["qualifier_implementation_version"])
        self.assertEqual(set(HN_CHECK_BY_ID), set(engine["implemented_check_ids"]))
        self.assertFalse(engine["generation_available"])
        self.assertFalse(engine["sampling_available"])

    def test_real_validator_report_can_qualify_and_gate_an_explicit_fixture(self):
        spec = ScenarioEngine(8201).normal()
        spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
        persistent = PersistentDeviceContext(device_id="integration-device")
        events, ctx = EventEngine(
            device_id=persistent.device_id,
            gateway_id="gateway",
            auth_server_id="auth",
            broker_id="broker",
            start_time=50_000.0,
            persistent_context=persistent,
        ).execute(spec)
        provenance = build_generation_provenance(
            cfg,
            run_id="hn2-integration",
            generator_git_commit="hn2-test",
            generator_git_dirty=False,
        )
        manifest = dataset_capability_manifest(
            provenance,
            generation_profiles=(ctx.generation_profile,),
        )
        initial = validate_synchronized_dataset(
            [(events, ctx)],
            provenance=provenance,
            field_manifest=field_manifest(),
            dataset_manifest=manifest,
        )
        target = next(item for item in events if item.event_type is EventType.AUTHENTICATION_SUCCESS)
        source = next(
            item for item in events
            if item.event_type is EventType.AUTHENTICATION_REQUEST
            and item.auth_attempt_id == target.auth_attempt_id
        )
        check_id = "hn.v1.identity.persistent_authenticated_coherent"
        check = HN_CHECK_BY_ID[check_id]
        variant = HN_VARIANT_BY_ID[check.variant_id]
        explicit = HardNegativeQualificationRecord(
            hard_negative_candidate_id="hn2-integration-candidate",
            trace_id=ctx.trace_id,
            scenario_id=ctx.scenario_id,
            device_id=ctx.device_id,
            candidate_status=CandidateStatus.REQUESTED,
            primary_intended_family_id=check.family_id,
            intended_variant_id=variant.variant_id,
            intended_boundary_invariant_ids=variant.invariant_ids,
            construction_status=ConstructionStatus.APPLIED,
            construction_parameters={"fixture": "explicit"},
            boundary_parameters={"boundary_targeted": True},
            target_event_ids=(target.event_id,),
            evidence_event_ids=(source.event_id, target.event_id),
        )
        qualified = qualify_hard_negative(events, ctx, explicit, initial)
        self.assertEqual(QualificationStatus.QUALIFIED, qualified.qualification_status)
        final = validate_synchronized_dataset(
            [(events, ctx)],
            provenance=provenance,
            field_manifest=field_manifest(),
            dataset_manifest=manifest,
            hard_negative_qualification_records=(qualified,),
        )
        self.assertTrue(final.build_gate_passed)
        self.assertTrue(final.hard_negative_structural_gate["passed"])


class HN2AdversarialQualificationTests(unittest.TestCase):
    def qualify(self, check_id, transform=None, *, report_transform=None, context_transform=None):
        events, ctx, record, report = positive_fixture(check_id)
        if transform:
            events, record = transform(events, record)
        if report_transform:
            report = report_transform(report)
        if context_transform:
            ctx = context_transform(ctx)
        return qualify_hard_negative(events, ctx, record, report)

    def test_intent_or_construction_metadata_without_evidence_never_qualifies(self):
        check_id = "hn.v1.nonce.fresh_cross_attempt"
        empty = candidate(check_id, [], boundary={"attempt_relationship": "retry"}, targets=(), evidence=())
        result = qualify_hard_negative([], context(), empty, validator_report(check_id))
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, result.qualification_status)

        not_applied = replace(empty, construction_status=ConstructionStatus.REQUESTED_NOT_APPLIED)
        result = qualify_hard_negative([], context(), not_applied, validator_report(check_id))
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, result.qualification_status)

    def test_nonce_violation_missing_and_unknown_attempt_are_distinct(self):
        def equal(events, record):
            events[-1].nonce = events[0].nonce
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.nonce.fresh_cross_attempt", equal).qualification_status)

        def missing(events, record):
            events[-1].nonce = None
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, self.qualify("hn.v1.nonce.fresh_cross_attempt", missing).qualification_status)

        def unknown(events, record):
            events[1].auth_attempt_id = None
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, self.qualify("hn.v1.nonce.same_attempt_recurrence", unknown).qualification_status)

    def test_token_supersession_and_unknown_currentness_do_not_qualify(self):
        def superseded(events, record):
            events[1].current_token_id = "T2"
            events[1].token_lineage_status = "superseded"
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.token.current_repeated_use", superseded).qualification_status)

        def unknown(events, record):
            events[1].current_token_id = None
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, self.qualify("hn.v1.token.current_repeated_use", unknown).qualification_status)

        def hidden_replacement(events, record):
            events.insert(1, event(
                "replacement-between-uses", EventType.TOKEN_ISSUED, 10.5,
                token_id="T2", current_token_id="T2", token_lineage_status="current",
                supersedes_token_id="T1", protected_session_id=SESSION_ID,
            ))
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.token.current_repeated_use", hidden_replacement).qualification_status)

    def test_causal_inversion_equality_and_outside_margin_are_separate(self):
        def inversion(events, record):
            events[1].observed_timestamp = 9.9
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.time.positive_causal_margin", inversion).qualification_status)

        def equality(events, record):
            events[1].observed_timestamp = 10.0
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, self.qualify("hn.v1.time.positive_causal_margin", equality).qualification_status)

        def outside(events, record):
            events[1].observed_timestamp = 10.5
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, self.qualify("hn.v1.time.positive_causal_margin", outside).qualification_status)

    def test_access_requires_positive_outcome_and_enforces_prerequisites(self):
        def no_outcome(events, record):
            events[-1].resource_operation_outcome = None
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, self.qualify("hn.v1.access.missing_auth_enforced", no_outcome).qualification_status)

        def success(events, record):
            events[-1].event_type = EventType.ACCESS_GRANTED
            events[-1].result = EventResult.SUCCESS
            events[-1].resource_operation_outcome = "success"
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.access.missing_auth_enforced", success).qualification_status)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.access.explicit_deny_enforced", success).qualification_status)

    def test_authz_separation_does_not_switch_family_for_invalid_credential(self):
        def invalid(events, record):
            events[-1].token_context_active = False
            events[-1].token_validation_result = "rejected"
            return events, record
        result = self.qualify("hn.v1.authz.valid_context_legitimate_denial", invalid)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED, result.qualification_status)
        self.assertEqual((), result.qualified_family_ids)

    def test_identity_alias_or_network_value_cannot_replace_independent_identity(self):
        def remove_comparison(report):
            report.records[0].observed_evidence = {"comparisons": []}
            return report
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED, self.qualify("hn.v1.identity.persistent_authenticated_coherent", report_transform=remove_comparison).qualification_status)

        def network_only(events, record):
            events[-1].authenticated_identity = None
            return events, record
        self.assertEqual(
            QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE,
            self.qualify(
                "hn.v1.identity.persistent_authenticated_coherent",
                network_only,
                context_transform=lambda _ctx: context(
                    attempts=("attempt-1",), source_ip="192.0.2.1"
                ),
            ).qualification_status,
        )

    def test_binding_missing_or_wrong_never_qualifies(self):
        def missing(events, record):
            events[-1].token_bound_identity = None
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE, self.qualify("hn.v1.binding.identity_coherent", missing).qualification_status)

        def wrong(events, record):
            events[-1].token_bound_identity = "other-device"
            return events, record
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, self.qualify("hn.v1.binding.identity_coherent", wrong).qualification_status)

    def test_validator_unsupported_error_fail_and_positive_gt_are_authoritative(self):
        check_id = "hn.v1.access.missing_auth_enforced"
        for validator_result, expected in (
            (ValidationResult.UNSUPPORTED, QualificationStatus.NOT_QUALIFIED_RELEVANT_UNSUPPORTED),
            (ValidationResult.ERROR, QualificationStatus.QUALIFICATION_ERROR),
            (ValidationResult.FAIL, QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION),
        ):
            with self.subTest(validator_result=validator_result):
                result = self.qualify(check_id, report_transform=lambda report, value=validator_result: validator_report(check_id, result=value))
                self.assertEqual(expected, result.qualification_status)
                if expected is QualificationStatus.QUALIFICATION_ERROR:
                    self.assertFalse(validate_hard_negative_structure((result,)).passed)

        result = self.qualify(check_id, context_transform=lambda ctx: context(observable_anomaly=True))
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION, result.qualification_status)

    def test_mapping_tampering_is_qualification_error_and_gate_failure(self):
        events, ctx, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        tampered = replace(record, intended_boundary_invariant_ids=("INV-TIME-OBSERVED-CAUSAL",))
        result = qualify_hard_negative(events, ctx, tampered, report)
        self.assertEqual(QualificationStatus.QUALIFICATION_ERROR, result.qualification_status)
        self.assertFalse(validate_hard_negative_structure((result,)).passed)

    def test_cross_trace_or_device_evidence_is_a_qualification_error(self):
        def cross_trace(events, record):
            events[-1].trace_id = "other-trace"
            return events, record
        self.assertEqual(
            QualificationStatus.QUALIFICATION_ERROR,
            self.qualify("hn.v1.nonce.fresh_cross_attempt", cross_trace).qualification_status,
        )

    def test_semantic_gate_rejects_tampered_qualified_claims_and_unknown_references(self):
        events, ctx, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        qualified = qualify_hard_negative(events, ctx, record, report)
        missing_predicate = replace(qualified, normal_side_predicates={})
        self.assertFalse(validate_hard_negative_structure((missing_predicate,)).passed)
        tampered_claims = (
            replace(qualified, boundary_proximity_evidence={}),
            replace(
                qualified,
                relevant_validation_outcomes={
                    qualified.relevant_validation_check_ids[0]: "FAIL"
                },
            ),
            replace(qualified, disqualifying_invariant_ids=("INV-TIME-OBSERVED-CAUSAL",)),
            replace(qualified, trace_observable_anomaly=True),
        )
        for item in tampered_claims:
            with self.subTest(item=item):
                self.assertFalse(validate_hard_negative_structure((item,)).passed)
        bad_reference = replace(qualified, evidence_event_ids=("not-an-event",))
        gate = validate_hard_negative_structure(
            (bad_reference,),
            known_event_ids_by_trace={TRACE_ID: {event.event_id for event in events}},
            trace_anomaly_gt={TRACE_ID: False},
        )
        self.assertFalse(gate.passed)

    def test_reporting_counts_status_by_family_variant_and_check(self):
        qualified_results = []
        for check_id in HN_CHECK_BY_ID:
            events, ctx, record, report = positive_fixture(check_id)
            qualified_results.append(qualify_hard_negative(events, ctx, record, report))
        check_id = "hn.v1.time.positive_causal_margin"
        events, ctx, record, report = positive_fixture(check_id)
        events[-1].observed_timestamp = 10.0
        underdetermined = qualify_hard_negative(events, ctx, record, report)
        summary = HardNegativeRunSummary.from_records(
            total_traces=16,
            ordinary_normal_traces=0,
            records=(*qualified_results, underdetermined),
        )
        self.assertEqual(15, summary.qualification_status_counts["qualified"])
        self.assertEqual(1, summary.qualification_status_counts["not_qualified_underdetermined"])
        self.assertEqual(1, summary.qualification_status_by_check[check_id]["qualified"])
        self.assertEqual(1, summary.qualification_status_by_check[check_id]["not_qualified_underdetermined"])

    def test_detector_boundary_remains_exact_and_privileged(self):
        self.assertEqual(
            (
                "event_type", "observed_timestamp", "observed_delay_since_previous_event",
                "result", "failure_reason", "retry_count", "token_id", "nonce",
                "topic", "resource_id", "requested_action",
            ),
            field_names_for_view("detector_observations"),
        )
        self.assertTrue({
            "normal_side_predicates", "boundary_proximity_evidence", "qualification_status",
            "qualification_check_ids", "evidence_event_ids", "relevant_validation_outcomes",
        }.issubset(HARD_NEGATIVE_PRIVILEGED_FIELD_NAMES))

    def test_explicit_status_matrix_exercises_each_outcome_once(self):
        results = []

        events, ctx, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        results.append(qualify_hard_negative(events, ctx, record, report))
        results.append(qualify_hard_negative(
            [], context(),
            replace(candidate("hn.v1.nonce.fresh_cross_attempt", [], boundary={"attempt_relationship": "retry"}, targets=(), evidence=()), construction_status=ConstructionStatus.REQUESTED_NOT_APPLIED),
            validator_report("hn.v1.nonce.fresh_cross_attempt"),
        ))
        results.append(qualify_hard_negative(
            [], context(),
            candidate("hn.v1.nonce.fresh_cross_attempt", [], boundary={"attempt_relationship": "retry"}, targets=(), evidence=()),
            validator_report("hn.v1.nonce.fresh_cross_attempt"),
        ))

        events, ctx, record, report = positive_fixture("hn.v1.time.positive_causal_margin")
        events[-1].observed_timestamp = events[0].observed_timestamp
        results.append(qualify_hard_negative(events, ctx, record, report))

        events, ctx, record, _report = positive_fixture("hn.v1.access.missing_auth_enforced")
        results.append(qualify_hard_negative(events, ctx, record, validator_report("hn.v1.access.missing_auth_enforced", result=ValidationResult.UNSUPPORTED)))

        events, ctx, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        events[-1].nonce = events[0].nonce
        results.append(qualify_hard_negative(events, ctx, record, report))

        events, ctx, record, report = positive_fixture("hn.v1.time.positive_causal_margin")
        events[-1].observed_timestamp = 10.5
        results.append(qualify_hard_negative(events, ctx, record, report))

        events, ctx, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        record = replace(record, intended_boundary_invariant_ids=("INV-TIME-OBSERVED-CAUSAL",))
        results.append(qualify_hard_negative(events, ctx, record, report))

        counts = Counter(item.qualification_status for item in results)
        self.assertEqual({status: 1 for status in QualificationStatus}, dict(counts))


if __name__ == "__main__":
    unittest.main()
