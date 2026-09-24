"""C2.4 atomic credential/token-binding invariant tests."""

import copy
import random
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine, ScenarioSpec
from simulator.event_model import EventResult, EventType
from simulator.invariant_registry import (
    EXACT_VARIANT_TO_INVARIANTS,
    TOKEN_BINDING_RELATION_CAPABILITIES,
)
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


VARIANT = "token_bound_identity_mismatch_authenticated_context"
IDENTITY_CHECK = "c2.4.binding.identity"
CONTEXT_CHECK = "c2.4.binding.protected_context"
IDENTITY_CONSISTENCY_CHECK = "c2.3.identity.persistent_authenticated"


def execute(spec, *, seed=2401):
    random.seed(seed)
    persistent = PersistentDeviceContext(device_id="persistent-A")
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="gateway-c2-binding",
        auth_server_id="auth-c2-binding",
        broker_id="broker-c2-binding",
        start_time=60_000.0,
        persistent_context=persistent,
    ).execute(spec)


def normal_pair(seed=2401, *, steps=None):
    return execute(ScenarioSpec(
        scenario_id=f"normal-binding-{seed}",
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
        normal_steps=list(NORMAL_FLOW if steps is None else steps),
    ), seed=seed)


def variant_pair(name=VARIANT, seed=2402, *, stealth=0.0):
    with patch.dict(STEALTH_FRACTION, {name: stealth}):
        return execute(ScenarioEngine(seed).synchronized_anomaly(name), seed=seed)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2.4-binding-test", generator_git_commit="c2.4-test"
    )
    return validate_synchronized_dataset(
        [pair],
        provenance=provenance,
        field_manifest=field_manifest(),
        dataset_manifest=dataset_capability_manifest(
            provenance,
            generation_profiles=(pair[1].generation_profile,),
        ),
    )


def record(report, check):
    return next(item for item in report.records if item.validation_id == check)


def binding_records(report):
    return record(report, IDENTITY_CHECK), record(report, CONTEXT_CHECK)


def set_authenticated_identity(pair, identity):
    events, context = pair
    established = False
    for event in events:
        if (
            event.event_type == EventType.AUTHENTICATION_SUCCESS
            and event.result is EventResult.SUCCESS
        ):
            established = True
        if established and event.authenticated_identity is not None:
            event.authenticated_identity = identity
    context.authenticated_identity = identity


def set_token_bound_identity(pair, identity):
    events, _context = pair
    for event in events:
        if event.token_id:
            event.token_bound_identity = identity


class AtomicTokenBindingTests(unittest.TestCase):
    def test_token_bound_to_authenticated_identity_is_coherent(self):
        report = validate(normal_pair())
        identity, protected_context = binding_records(report)
        self.assertEqual(ValidationResult.NOT_APPLICABLE, identity.result)
        self.assertEqual(ValidationResult.NOT_APPLICABLE, protected_context.result)
        self.assertTrue(report.build_gate_passed)

    def test_atomic_variant_confirms_binding_without_identity_inconsistency(self):
        events, context = variant_pair()
        report = validate((events, context))
        target = next(event for event in events if event.anomaly_label == VARIANT)
        identity, protected_context = binding_records(report)

        self.assertEqual("persistent-A", context.device_id)
        self.assertEqual("persistent-A", target.authenticated_identity)
        self.assertNotEqual(target.token_bound_identity, target.authenticated_identity)
        self.assertEqual(
            "not_evaluated_at_presentation",
            context.injection_records[0]["injection_parameters"][
                "security_enforcement_outcome"
            ],
        )
        self.assertEqual(ValidationResult.PASS, identity.result)
        self.assertEqual(ValidationResult.NOT_APPLICABLE, protected_context.result)
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            record(report, IDENTITY_CONSISTENCY_CHECK).result,
        )
        self.assertTrue(report.build_gate_passed)

    def test_mismatch_metadata_without_actual_mismatch_is_not_confirmed(self):
        events, context = copy.deepcopy(variant_pair(seed=2403))
        for event in events:
            if event.token_id:
                event.token_bound_identity = event.authenticated_identity
        report = validate((events, context))
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result
        )
        self.assertTrue(report.intended_candidates_not_confirmed)
        self.assertFalse(report.build_gate_passed)

    def test_identity_can_be_inconsistent_while_token_binding_is_coherent(self):
        pair = normal_pair(2404, steps=NORMAL_FLOW[:13])
        set_authenticated_identity(pair, "authenticated-B")
        set_token_bound_identity(pair, "authenticated-B")
        report = validate(pair)

        self.assertEqual(ValidationResult.PASS, record(report, IDENTITY_CONSISTENCY_CHECK).result)
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertFalse(report.build_gate_passed)

    def test_superseded_replay_with_correct_binding_is_not_binding_mismatch(self):
        pair = variant_pair("token_replay_after_replacement", 2405)
        report = validate(pair)
        identity, protected_context = binding_records(report)
        replay = record(
            report, "c2.1.replay_token.superseded_within_continuing_session"
        )
        self.assertEqual(ValidationResult.PASS, replay.result)
        self.assertNotEqual(ValidationResult.PASS, identity.result)
        self.assertNotEqual(ValidationResult.PASS, protected_context.result)
        self.assertTrue(report.build_gate_passed)

    def test_expired_but_correctly_bound_token_is_not_binding_mismatch(self):
        events, context = normal_pair(2406, steps=NORMAL_FLOW[:14])
        validated = next(
            event for event in events if event.event_type == EventType.TOKEN_VALIDATED
        )
        validated.token_expiry = validated.timestamp - 1.0
        report = validate((events, context))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertEqual(
            ValidationResult.FAIL,
            record(report, "c1.7.temporal.supported_subset").result,
        )

    def test_valid_bound_token_without_grant_is_access_not_binding(self):
        pair = variant_pair("access_success_without_applicable_grant", 2407)
        report = validate(pair)
        self.assertNotEqual(ValidationResult.PASS, record(report, IDENTITY_CHECK).result)
        self.assertEqual(
            ValidationResult.PASS, record(report, "c2.2.access.authz_grant").result
        )
        self.assertTrue(report.build_gate_passed)

    def test_wrong_represented_protected_context_confirms_context_binding_only(self):
        events, context = normal_pair(2408)
        opened = next(event for event in events if event.event_type == EventType.SESSION_OPENED)
        opened.token_bound_protected_session_id = "other-protected-context"
        report = validate((events, context))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertEqual(ValidationResult.PASS, record(report, CONTEXT_CHECK).result)
        self.assertFalse(report.build_gate_passed)

    def test_missing_token_bound_identity_is_unsupported(self):
        events, context = normal_pair(2409, steps=NORMAL_FLOW[:13])
        for event in events:
            if event.token_id:
                event.token_bound_identity = None
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            record(validate((events, context)), IDENTITY_CHECK).result,
        )

    def test_historical_label_alone_establishes_no_binding_truth(self):
        events, context = normal_pair(2410)
        context.attack_type = "identity_token_mismatch"
        context.attack_scenario_intent = True
        report = validate((events, context))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertFalse(report.intended_candidates_confirmed)
        self.assertTrue(report.build_gate_passed)

    def test_ip_change_is_not_binding_evidence(self):
        events, context = normal_pair(2411)
        context.source_ip = "203.0.113.44"
        context.source_ip_change = 1
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            record(validate((events, context)), IDENTITY_CHECK).result,
        )

    def test_injection_metadata_cannot_fool_binding_validator(self):
        events, context = normal_pair(2412, steps=NORMAL_FLOW[:13])
        context.injection_records = ({
            "injection_id": "tampered",
            "anomaly_variant_name": VARIANT,
            "canonical_invariant_ids": ("INV-BIND-TOKEN-CONTEXT",),
        },)
        report = validate((events, context))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertTrue(report.intended_candidates_not_confirmed)
        self.assertFalse(report.build_gate_passed)

    def test_relation_registry_and_variant_mapping_are_conservative(self):
        self.assertEqual(("INV-BIND-TOKEN-CONTEXT",), EXACT_VARIANT_TO_INVARIANTS[VARIANT])
        self.assertEqual(
            "unsupported_by_current_evidence",
            TOKEN_BINDING_RELATION_CAPABILITIES["issuer_applicability"]["status"],
        )
        self.assertEqual(
            "partial",
            TOKEN_BINDING_RELATION_CAPABILITIES[
                "resource_action_scope_applicability"
            ]["status"],
        )

    def test_binding_fields_are_debug_only_not_detector_visible(self):
        detector_fields = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        debug_fields = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "debug_events"
        }
        self.assertNotIn("token_bound_identity", detector_fields)
        self.assertNotIn("token_bound_protected_session_id", detector_fields)
        self.assertIn("token_bound_identity", debug_fields)
        self.assertIn("token_bound_protected_session_id", debug_fields)

    def test_stealth_intent_is_coherent_and_unlabeled(self):
        events, context = variant_pair(seed=2413, stealth=1.0)
        report = validate((events, context))
        self.assertFalse(context.injection_records)
        self.assertTrue(all(event.anomaly_label is None for event in events))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record(report, IDENTITY_CHECK).result)
        self.assertTrue(report.build_gate_passed)


if __name__ == "__main__":
    unittest.main()
