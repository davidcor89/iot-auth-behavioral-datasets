"""C2.2 request-scoped protected-access invariant tests."""

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
from simulator.invariant_registry import EXACT_VARIANT_TO_INVARIANTS
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


ACCESS_CHECKS = {
    "INV-ACCESS-AUTH-CONTEXT": "c2.2.access.auth_context",
    "INV-ACCESS-CREDENTIAL": "c2.2.access.credential",
    "INV-ACCESS-ACTIVE-SESSION": "c2.2.access.active_session",
    "INV-ACCESS-AUTHZ-GRANT": "c2.2.access.authz_grant",
    "INV-ACCESS-DENY-ENFORCED": "c2.2.access.deny_enforced",
}


def execute(spec, *, seed=2201):
    random.seed(seed)
    persistent = PersistentDeviceContext(device_id="device-c2-access")
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="gateway-c2-access",
        auth_server_id="auth-c2-access",
        broker_id="broker-c2-access",
        start_time=40_000.0,
        persistent_context=persistent,
    ).execute(spec)


def normal_pair(seed=2201):
    return execute(ScenarioSpec(
        scenario_id=f"normal-access-{seed}",
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
        normal_steps=list(NORMAL_FLOW),
    ), seed=seed)


def denied_pair(seed=2202):
    return execute(ScenarioSpec(
        scenario_id=f"denied-access-{seed}",
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
        normal_steps=[
            *NORMAL_FLOW[:16],
            EventType.ACCESS_DENIED,
            EventType.SESSION_CLOSED,
        ],
    ), seed=seed)


def variant_pair(name, seed=2203, *, stealth=0.0):
    with patch.dict(STEALTH_FRACTION, {name: stealth}):
        return execute(ScenarioEngine(seed).synchronized_anomaly(name), seed=seed)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2.2-access-test", generator_git_commit="c2.2-test"
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


def access_records(report):
    return {
        record.invariant_identifier: record
        for record in report.records
        if record.validation_id in ACCESS_CHECKS.values()
    }


class AtomicAccessInvariantTests(unittest.TestCase):
    def test_fully_valid_access_has_no_atomic_violation(self):
        report = validate(normal_pair())
        self.assertTrue(report.build_gate_passed)
        self.assertEqual(
            {ValidationResult.NOT_APPLICABLE},
            {record.result for record in access_records(report).values()},
        )

    def test_denied_operation_with_missing_prerequisites_is_not_anomaly(self):
        events, context = denied_pair()
        denied = next(e for e in events if e.event_type == EventType.ACCESS_DENIED)
        denied.authenticated_context_active = False
        denied.token_context_active = False
        denied.protected_session_active = False

        report = validate((events, context))
        self.assertEqual(
            {ValidationResult.NOT_APPLICABLE},
            {record.result for record in access_records(report).values()},
        )
        self.assertTrue(report.build_gate_passed)

    def test_each_atomic_variant_confirms_only_its_declared_access_rules(self):
        variants = {
            "access_success_without_auth_context": {"INV-ACCESS-AUTH-CONTEXT"},
            "access_success_without_valid_credential": {"INV-ACCESS-CREDENTIAL"},
            "access_success_without_active_session": {"INV-ACCESS-ACTIVE-SESSION"},
            "access_success_without_applicable_grant": {"INV-ACCESS-AUTHZ-GRANT"},
        }
        for index, (variant, expected) in enumerate(variants.items()):
            with self.subTest(variant=variant):
                pair = variant_pair(variant, 2210 + index)
                report = validate(pair)
                detected = {
                    invariant_id for invariant_id, record in access_records(report).items()
                    if record.result is ValidationResult.PASS
                }
                self.assertEqual(expected, detected)
                self.assertTrue(report.build_gate_passed)

    def test_explicit_deny_and_success_confirms_grant_and_enforcement_rules(self):
        pair = variant_pair("access_success_despite_explicit_deny", 2220)
        report = validate(pair)
        detected = {
            invariant_id for invariant_id, record in access_records(report).items()
            if record.result is ValidationResult.PASS
        }
        self.assertEqual(
            {"INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED"},
            detected,
        )
        self.assertTrue(report.build_gate_passed)

    def test_explicit_deny_with_denied_operation_is_enforced(self):
        report = validate(denied_pair(2221))
        records = access_records(report)
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            records["INV-ACCESS-DENY-ENFORCED"].result,
        )

    def test_incomplete_authorization_observation_is_underdetermined(self):
        events, context = copy.deepcopy(normal_pair(2222))
        grant = next(e for e in events if e.event_type == EventType.ACCESS_GRANTED)
        grant.authorization_decision = None
        grant.authorization_evidence_complete = False
        grant.authorization_authoritative = False

        report = validate((events, context))
        records = access_records(report)
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            records["INV-ACCESS-AUTHZ-GRANT"].result,
        )
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            records["INV-ACCESS-DENY-ENFORCED"].result,
        )

    def test_valid_token_does_not_supply_authorization(self):
        pair = variant_pair("access_success_without_applicable_grant", 2223)
        events, _context = pair
        target = next(e for e in events if e.anomaly_label)
        self.assertTrue(target.token_context_active)
        self.assertEqual("validated", target.token_validation_result)

        records = access_records(validate(pair))
        self.assertEqual(
            ValidationResult.NOT_APPLICABLE,
            records["INV-ACCESS-CREDENTIAL"].result,
        )
        self.assertEqual(
            ValidationResult.PASS,
            records["INV-ACCESS-AUTHZ-GRANT"].result,
        )

    def test_one_operation_can_confirm_multiple_atomic_invariants(self):
        pair = variant_pair("access_success_without_security_prerequisites", 2224)
        report = validate(pair)
        detected = {
            invariant_id for invariant_id, record in access_records(report).items()
            if record.result is ValidationResult.PASS
        }
        self.assertEqual(
            {
                "INV-ACCESS-AUTH-CONTEXT",
                "INV-ACCESS-CREDENTIAL",
                "INV-ACCESS-ACTIVE-SESSION",
            },
            detected,
        )
        event_sets = {
            record.event_references for record in access_records(report).values()
            if record.result is ValidationResult.PASS
        }
        self.assertEqual(1, len(event_sets))
        self.assertTrue(report.build_gate_passed)

    def test_missing_request_linkage_cannot_confirm_access_invariant(self):
        events, context = copy.deepcopy(
            variant_pair("access_success_without_auth_context", 2225)
        )
        target = next(e for e in events if e.anomaly_label)
        target.access_request_id = "unlinked-request"

        report = validate((events, context))
        self.assertEqual(
            ValidationResult.UNSUPPORTED,
            access_records(report)["INV-ACCESS-AUTH-CONTEXT"].result,
        )
        self.assertTrue(report.intended_candidates_not_confirmed)
        self.assertFalse(report.build_gate_passed)

    def test_legacy_scenario_name_alone_establishes_no_access_invariant(self):
        events, context = normal_pair(2226)
        context.attack_type = "access_without_auth"
        context.attack_scenario_intent = True

        report = validate((events, context))
        self.assertEqual(
            {ValidationResult.NOT_APPLICABLE},
            {record.result for record in access_records(report).values()},
        )
        self.assertFalse(report.intended_candidates_confirmed)
        self.assertTrue(report.build_gate_passed)

    def test_variant_provenance_maps_to_stable_ids(self):
        for index, variant in enumerate(
            name for name in EXACT_VARIANT_TO_INVARIANTS if name.startswith("access_")
        ):
            with self.subTest(variant=variant):
                _events, context = variant_pair(variant, 2230 + index)
                self.assertEqual(
                    EXACT_VARIANT_TO_INVARIANTS[variant],
                    tuple(context.injection_records[0]["canonical_invariant_ids"]),
                )


if __name__ == "__main__":
    unittest.main()
