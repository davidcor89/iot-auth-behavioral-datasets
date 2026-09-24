"""C2.3 persistent-device to authenticated-identity invariant tests."""

import copy
import random
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine, ScenarioSpec
from simulator.event_model import EventType
from simulator.invariant_registry import (
    EXACT_VARIANT_TO_INVARIANTS,
    IDENTITY_RELATION_CAPABILITIES,
)
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


VARIANT = "authenticated_identity_mismatch_persistent_device"
CHECK = "c2.3.identity.persistent_authenticated"


def execute(spec, *, seed=2301):
    random.seed(seed)
    persistent = PersistentDeviceContext(device_id="persistent-A")
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="gateway-c2-identity",
        auth_server_id="auth-c2-identity",
        broker_id="broker-c2-identity",
        start_time=50_000.0,
        persistent_context=persistent,
    ).execute(spec)


def normal_pair(seed=2301):
    return execute(ScenarioSpec(
        scenario_id=f"normal-identity-{seed}",
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
        normal_steps=list(NORMAL_FLOW),
    ), seed=seed)


def variant_pair(seed=2302, *, stealth=0.0):
    with patch.dict(STEALTH_FRACTION, {VARIANT: stealth}):
        return execute(ScenarioEngine(seed).synchronized_anomaly(VARIANT), seed=seed)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2.3-identity-test", generator_git_commit="c2.3-test"
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


def identity_record(report):
    return next(record for record in report.records if record.validation_id == CHECK)


class PersistentAuthenticatedIdentityTests(unittest.TestCase):
    def test_coherent_persistent_and_authenticated_identity_has_no_violation(self):
        report = validate(normal_pair())
        record = identity_record(report)
        self.assertEqual(ValidationResult.NOT_APPLICABLE, record.result)
        self.assertTrue(report.build_gate_passed)

    def test_independent_mismatch_with_valid_context_link_confirms_violation(self):
        events, context = variant_pair()
        target = next(event for event in events if event.anomaly_label == VARIANT)
        record = identity_record(validate((events, context)))

        self.assertEqual("persistent-A", context.device_id)
        self.assertNotEqual(context.device_id, target.authenticated_identity)
        self.assertEqual(
            "not_applicable_synchronized_identity_variant",
            context.injection_records[0]["historical_scenario_name"],
        )
        self.assertNotIn(
            context.injection_records[0]["historical_scenario_name"],
            {"impersonation", "identity_token_mismatch"},
        )
        self.assertEqual(ValidationResult.PASS, record.result)
        self.assertEqual(
            "INV-IDENT-AUTH-ACTOR-CONSISTENCY", record.invariant_identifier
        )
        self.assertTrue(record.observed_evidence["comparisons"][0][
            "context_device_association_reliable"
        ])
        self.assertTrue(validate((events, context)).build_gate_passed)

    def test_injection_metadata_cannot_substitute_for_identity_contradiction(self):
        events, context = copy.deepcopy(variant_pair(2303))
        target = next(event for event in events if event.anomaly_label == VARIANT)
        target.authenticated_identity = context.device_id
        context.authenticated_identity = context.device_id

        report = validate((events, context))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, identity_record(report).result)
        self.assertTrue(report.intended_candidates_not_confirmed)
        self.assertFalse(report.build_gate_passed)

    def test_auth_context_associated_with_another_device_is_unsupported(self):
        events, context = copy.deepcopy(variant_pair(2304))
        target = next(event for event in events if event.anomaly_label == VARIANT)
        target.device_id = "different-device-context"

        record = identity_record(validate((events, context)))
        self.assertEqual(ValidationResult.UNSUPPORTED, record.result)
        self.assertFalse(record.observed_evidence["comparisons"][0][
            "context_device_association_reliable"
        ])

    def test_missing_persistent_identity_is_unsupported(self):
        events, context = copy.deepcopy(variant_pair(2305))
        context.device_id = ""
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            identity_record(validate((events, context))).result,
        )

    def test_missing_authenticated_identity_is_unsupported(self):
        events, context = copy.deepcopy(variant_pair(2306))
        target = next(event for event in events if event.anomaly_label == VARIANT)
        target.authenticated_identity = None
        context.authenticated_identity = None
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            identity_record(validate((events, context))).result,
        )

    def test_missing_context_link_is_unsupported_and_repaired_link_is_decidable(self):
        events, context = copy.deepcopy(variant_pair(2307))
        target = next(event for event in events if event.anomaly_label == VARIANT)
        original_attempts = context.auth_attempt_ids
        context.auth_attempt_ids = ()
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            identity_record(validate((events, context))).result,
        )

        context.auth_attempt_ids = original_attempts
        repaired = identity_record(validate((events, context)))
        self.assertEqual(ValidationResult.PASS, repaired.result)
        self.assertEqual(target.event_id, repaired.event_references[-1])

    def test_token_bound_to_other_identity_is_not_this_identity_violation(self):
        events, context = normal_pair(2308)
        token_event = next(
            event for event in events if event.event_type == EventType.TOKEN_ISSUED
        )
        # Deliberately attach independent token-owner evidence for this
        # adversarial validator test. C2.3 must not reinterpret it as the
        # persistent/authenticated relation or change token-binding semantics.
        token_event.token_bound_identity = "token-owner-B"
        context.token_device_mismatch = 1
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            identity_record(validate((events, context))).result,
        )

    def test_ip_change_with_coherent_identities_is_not_identity_violation(self):
        events, context = normal_pair(2309)
        context.source_ip = "203.0.113.9"
        context.source_ip_change = 1
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            identity_record(validate((events, context))).result,
        )

    def test_historical_identity_labels_alone_establish_no_stable_truth(self):
        for historical in ("impersonation", "identity_token_mismatch"):
            with self.subTest(historical=historical):
                events, context = copy.deepcopy(normal_pair(2310))
                context.attack_type = historical
                context.attack_scenario_intent = True
                self.assertEqual(
                    ValidationResult.NOT_APPLICABLE,
                    identity_record(validate((events, context))).result,
                )

    def test_request_actor_relation_remains_explicitly_unsupported(self):
        events, context = normal_pair(2311)
        record = identity_record(validate((events, context)))
        self.assertEqual(
            "unsupported_by_current_evidence",
            record.observed_evidence["request_actor_comparison"],
        )
        self.assertEqual(
            "unsupported_by_current_evidence",
            IDENTITY_RELATION_CAPABILITIES[
                "authenticated_identity_to_request_acting_identity"
            ]["status"],
        )

    def test_variant_mapping_and_identity_fields_remain_outside_detector_view(self):
        self.assertEqual(
            ("INV-IDENT-AUTH-ACTOR-CONSISTENCY",),
            EXACT_VARIANT_TO_INVARIANTS[VARIANT],
        )
        detector_fields = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        self.assertNotIn("device_id", detector_fields)
        self.assertNotIn("authenticated_identity", detector_fields)

    def test_stealth_intent_keeps_identity_coherent_and_unlabeled(self):
        events, context = variant_pair(2312, stealth=1.0)
        report = validate((events, context))
        self.assertFalse(context.injection_records)
        self.assertFalse(context.observable_anomaly)
        self.assertTrue(all(event.anomaly_label is None for event in events))
        self.assertEqual(ValidationResult.NOT_APPLICABLE, identity_record(report).result)
        self.assertTrue(report.build_gate_passed)


if __name__ == "__main__":
    unittest.main()
