"""HN.3R rejected-request evidence and anomalous-success counterparts."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine
from simulator.event_model import AuthState, EventResult, EventType
from simulator.hard_negative import ConstructionStatus, HN_VARIANT_BY_ID, QualificationStatus
from simulator.hard_negative_generation import HardNegativeRequest, generate_hard_negative
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import validate_synchronized_dataset


CASES = {
    "missing_auth_context_enforced": (
        "access_success_without_auth_context", "INV-ACCESS-AUTH-CONTEXT",
        "authentication_context",
    ),
    "invalid_credential_enforced": (
        "access_success_without_valid_credential", "INV-ACCESS-CREDENTIAL",
        "credential_context",
    ),
    "inactive_session_enforced": (
        "access_success_without_active_session", "INV-ACCESS-ACTIVE-SESSION",
        "active_session",
    ),
}


def hn_request(variant_id, suffix):
    variant = HN_VARIANT_BY_ID[variant_id]
    return HardNegativeRequest(
        variant.family_id, variant_id, f"hn3r-{variant_id}-{suffix}", {}, {},
    )


class RejectedAccessRepresentationTests(unittest.TestCase):
    def test_all_three_repaired_variants_qualify_from_emitted_evidence(self):
        for variant_id in CASES:
            with self.subTest(variant=variant_id):
                result = generate_hard_negative(
                    hn_request(variant_id, "positive"),
                    device_id=f"hn3r-{variant_id}", seed=4501, start_time=1000.0,
                )
                request, denial = result.events[-2:]
                self.assertEqual((EventType.ACCESS_REQUEST, EventType.ACCESS_DENIED),
                                 (request.event_type, denial.event_type))
                self.assertEqual(request.access_request_id, denial.access_request_id)
                self.assertEqual((request.resource_id, request.requested_action),
                                 (denial.resource_id, denial.requested_action))
                self.assertEqual(EventResult.FAILURE, request.result)
                self.assertEqual(EventResult.FAILURE, denial.result)
                self.assertEqual("not_executed", denial.resource_operation_outcome)
                self.assertEqual("denied", denial.authorization_decision)
                self.assertEqual(request.previous_state, request.new_state)
                self.assertEqual(denial.previous_state, denial.new_state)
                self.assertEqual(ConstructionStatus.APPLIED, result.candidate.construction_status)
                self.assertEqual(QualificationStatus.QUALIFIED, result.candidate.qualification_status)
                self.assertEqual("validated_normal", result.validation.trace_ground_truth_statuses[result.context.trace_id])
                self.assertEqual((), result.validation.trace_validated_invariant_ids[result.context.trace_id])
                self.assertTrue(result.gated_validation.build_gate_passed)
                self.assertIsNone(result.generator_defect)
                if variant_id == "missing_auth_context_enforced":
                    self.assertFalse(request.authenticated_context_active)
                    self.assertIsNone(request.authenticated_identity)
                    self.assertFalse(any(
                        event.event_type is EventType.AUTHENTICATION_SUCCESS
                        and event.result is EventResult.SUCCESS
                        for event in result.events
                    ))
                    self.assertEqual(AuthState.ENROLLED, request.previous_state)
                elif variant_id == "invalid_credential_enforced":
                    self.assertTrue(request.authenticated_context_active)
                    self.assertIsNotNone(request.token_id)
                    self.assertEqual("rejected", request.token_validation_result)
                    self.assertFalse(request.token_context_active)
                    self.assertTrue(any(event.event_type is EventType.TOKEN_REJECTED for event in result.events))
                else:
                    self.assertTrue(any(event.event_type is EventType.SESSION_CLOSED for event in result.events))
                    self.assertIsNotNone(request.protected_session_id)
                    self.assertEqual(request.protected_session_id, denial.protected_session_id)
                    self.assertFalse(request.protected_session_active)
                    self.assertFalse(denial.protected_session_active)
                    self.assertFalse(result.context.protected_session_active)

    def test_successful_operation_counterparts_remain_anomalous(self):
        for index, (variant_id, (anomaly_name, invariant_id, _)) in enumerate(CASES.items()):
            with self.subTest(variant=variant_id):
                normal = generate_hard_negative(
                    hn_request(variant_id, "counterfactual-source"),
                    device_id=f"hn3r-source-{index}", seed=4600 + index, start_time=1000.0,
                )
                with patch.dict(STEALTH_FRACTION, {anomaly_name: 0.0}):
                    spec = ScenarioEngine(4700 + index).synchronized_anomaly(anomaly_name)
                    persistent = PersistentDeviceContext(device_id=f"hn3r-counterpart-{index}")
                    events, context = EventEngine(
                        device_id=persistent.device_id, gateway_id="gateway",
                        auth_server_id="auth_server", broker_id="broker",
                        start_time=2000.0, persistent_context=persistent,
                    ).execute(spec)
                provenance = build_generation_provenance(
                    cfg, run_id=f"hn3r-counterpart-{index}",
                )
                report = validate_synchronized_dataset(
                    [(events, context)], provenance=provenance,
                    field_manifest=field_manifest(),
                    dataset_manifest=dataset_capability_manifest(
                        provenance, generation_profiles=(context.generation_profile,),
                    ),
                )
                operation = next(event for event in events if event.event_type is EventType.ACCESS_GRANTED)
                request = next(event for event in events if event.event_type is EventType.ACCESS_REQUEST
                               and event.access_request_id == operation.access_request_id)
                candidate = replace(
                    normal.candidate, trace_id=context.trace_id, scenario_id=context.scenario_id,
                    device_id=context.device_id, target_event_ids=(operation.event_id,),
                    evidence_event_ids=(request.event_id, operation.event_id),
                )
                qualified = qualify_hard_negative(events, context, candidate, report)
                self.assertEqual("validated_positive", report.trace_ground_truth_statuses[context.trace_id])
                self.assertIn(invariant_id, report.trace_validated_invariant_ids[context.trace_id])
                self.assertEqual("success", operation.resource_operation_outcome)
                self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION,
                                 qualified.qualification_status)

    def test_contradictory_prerequisite_is_construction_defect_and_gate_failure(self):
        for variant_id, (_, _, prerequisite) in CASES.items():
            with self.subTest(variant=variant_id):
                spec = ScenarioEngine(4801).normal()
                spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
                spec.normal_steps = list(NORMAL_FLOW[:NORMAL_FLOW.index(EventType.ACCESS_REQUEST)])
                spec.rejected_access_prerequisite = prerequisite
                result = generate_hard_negative(
                    hn_request(variant_id, "contradiction"),
                    device_id=f"hn3r-defect-{variant_id}", seed=4801,
                    start_time=1000.0, scenario=spec,
                )
                self.assertEqual(ConstructionStatus.CONSTRUCTION_ERROR,
                                 result.candidate.construction_status)
                self.assertEqual(QualificationStatus.QUALIFICATION_ERROR,
                                 result.candidate.qualification_status)
                self.assertIsNotNone(result.generator_defect)
                self.assertFalse(result.gated_validation.build_gate_passed)


if __name__ == "__main__":
    unittest.main()
