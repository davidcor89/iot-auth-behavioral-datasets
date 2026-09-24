"""C2.5 invariant-aware synchronized ground-truth reconciliation tests."""

import copy
import random
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import (
    SYNCHRONIZED_GENERATION_PROFILE,
    synchronized_capability_report,
)
from simulator.config.settings import cfg
from simulator.data.dataset_contract import (
    dataset_capability_manifest,
    field_manifest,
)
from simulator.data.synchronized_views import (
    to_detector_observation_df,
    to_ground_truth_views,
)
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine, ScenarioSpec
from simulator.event_model import EventType
from simulator.invariant_registry import (
    EXACT_VALIDATOR_CHECK_TO_INVARIANT,
    EXACT_VARIANT_TO_INVARIANTS,
    INVARIANT_REGISTRY,
    invariant_registry_report,
)
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


def execute_variant(name, seed=2501, *, stealth=0.0):
    random.seed(seed)
    with patch.dict(STEALTH_FRACTION, {name: stealth}):
        spec = ScenarioEngine(seed).synchronized_anomaly(name)
        persistent = PersistentDeviceContext(device_id=f"device-{seed}")
        return EventEngine(
            device_id=persistent.device_id,
            gateway_id="gateway-c2.5",
            auth_server_id="auth-c2.5",
            broker_id="broker-c2.5",
            start_time=70_000.0,
            persistent_context=persistent,
        ).execute(spec)


def normal_pair(seed=2590):
    random.seed(seed)
    persistent = PersistentDeviceContext(device_id=f"normal-device-{seed}")
    spec = ScenarioSpec(
        scenario_id=f"normal-{seed}",
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
        normal_steps=list(NORMAL_FLOW),
    )
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="gateway-c2.5",
        auth_server_id="auth-c2.5",
        broker_id="broker-c2.5",
        start_time=70_000.0,
        persistent_context=persistent,
    ).execute(spec)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2.5-test", generator_git_commit="c2.5-test"
    )
    report = validate_synchronized_dataset(
        [pair],
        provenance=provenance,
        field_manifest=field_manifest(),
        dataset_manifest=dataset_capability_manifest(
            provenance, generation_profiles=(pair[1].generation_profile,)
        ),
    )
    return report


def views(pair, report):
    return to_ground_truth_views([pair], report)


class GroundTruthConsolidationTests(unittest.TestCase):
    def test_one_confirmed_invariant_is_exported_exactly(self):
        pair = execute_variant("nonce_reuse")
        report = validate(pair)
        gt = views(pair, report)
        positive = [row for row in gt["events"] if row["validated_invariant_ids"]]
        self.assertEqual(1, len(positive))
        self.assertEqual(
            ["INV-REPLAY-NONCE-CROSS-ATTEMPT"],
            positive[0]["validated_invariant_ids"],
        )
        self.assertEqual(
            positive[0]["validated_invariant_ids"],
            gt["traces"][0]["validated_invariant_ids"],
        )

    def test_two_invariants_on_one_operation_are_both_retained(self):
        pair = execute_variant("access_success_despite_explicit_deny", 2502)
        gt = views(pair, validate(pair))
        positive = [row for row in gt["events"] if row["validated_invariant_ids"]]
        self.assertEqual(1, len(positive))
        self.assertEqual(
            {"INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED"},
            set(positive[0]["validated_invariant_ids"]),
        )
        self.assertEqual(2, len(positive[0]["validated_invariant_evidence"]))

    def test_variant_mapping_supports_one_many_and_many_one(self):
        self.assertEqual(
            3,
            len(EXACT_VARIANT_TO_INVARIANTS[
                "access_success_without_security_prerequisites"
            ]),
        )
        report = invariant_registry_report()
        authz = next(
            item for item in report["invariants"]
            if item["invariant_id"] == "INV-ACCESS-AUTHZ-GRANT"
        )
        self.assertGreaterEqual(len(authz["synchronized_anomaly_variants"]), 2)

    def test_attack_intent_without_injection_has_no_positive_truth(self):
        pair = execute_variant("nonce_reuse", 2503, stealth=1.0)
        report = validate(pair)
        self.assertEqual(
            "validated_normal", report.trace_ground_truth_statuses[pair[1].trace_id]
        )
        self.assertEqual((), report.trace_validated_invariant_ids[pair[1].trace_id])

    def test_applied_metadata_without_evidence_fails_without_positive_ids(self):
        events, context = copy.deepcopy(execute_variant("nonce_reuse", 2504))
        target = next(event for event in events if event.anomaly_label == "nonce_reuse")
        source = next(
            event for event in events
            if event.event_type == EventType.NONCE_RECEIVED
            and event.event_id != target.event_id
        )
        target.nonce = f"not-{source.nonce}"
        report = validate((events, context))
        self.assertFalse(report.build_gate_passed)
        self.assertEqual(
            "validation_failed", report.trace_ground_truth_statuses[context.trace_id]
        )
        self.assertEqual((), report.trace_validated_invariant_ids[context.trace_id])

    def test_unsupported_atomic_comparison_is_not_normal_or_positive(self):
        events, context = copy.deepcopy(normal_pair())
        for event in events:
            if event.token_id:
                event.token_bound_identity = None
        report = validate((events, context))
        self.assertEqual("unsupported", report.trace_ground_truth_statuses[context.trace_id])
        self.assertEqual((), report.trace_validated_invariant_ids[context.trace_id])
        self.assertIn("unsupported", set(report.event_ground_truth_statuses.values()))

    def test_historical_label_alone_carries_no_invariant_id(self):
        events, context = normal_pair(2591)
        context.attack_type = "impersonation"
        context.attack_scenario_intent = True
        report = validate((events, context))
        self.assertEqual((), report.trace_validated_invariant_ids[context.trace_id])
        self.assertTrue(all(not ids for ids in report.event_validated_invariant_ids.values()))

    def test_provenance_mapping_disagreement_fails_gate(self):
        events, context = copy.deepcopy(execute_variant("timestamp_inconsistency", 2505))
        context.injection_records[0]["canonical_invariant_ids"] = (
            "INV-REPLAY-NONCE-CROSS-ATTEMPT",
        )
        report = validate((events, context))
        self.assertFalse(report.build_gate_passed)
        self.assertTrue(report.provenance_schema_failures)

    def test_validator_identifier_disagreement_cannot_confirm_candidate(self):
        pair = execute_variant("nonce_reuse", 2506)
        with patch.dict(
            EXACT_VALIDATOR_CHECK_TO_INVARIANT,
            {
                "c1.7.nonce_reuse.within_trace_attempts":
                    "INV-TIME-OBSERVED-CAUSAL"
            },
        ):
            report = validate(pair)
        self.assertFalse(report.build_gate_passed)
        self.assertTrue(report.intended_candidates_not_confirmed)
        self.assertEqual((), report.trace_validated_invariant_ids[pair[1].trace_id])

    def test_invariant_truth_and_privileged_evidence_are_not_detector_fields(self):
        detector = set(to_detector_observation_df([normal_pair(2592)]).columns)
        for field in (
            "validated_invariant_ids",
            "validated_invariant_evidence",
            "authenticated_identity",
            "token_bound_identity",
            "injection_records",
            "scenario_intent",
        ):
            self.assertNotIn(field, detector)
        gt_fields = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"].startswith("ground_truth")
        }
        self.assertIn("validated_invariant_ids", gt_fields)

    def test_timestamp_and_superseded_replay_map_to_specific_ids(self):
        expected = {
            "timestamp_inconsistency": "INV-TIME-OBSERVED-CAUSAL",
            "token_replay_after_replacement": "INV-REPLAY-TOKEN-SUPERSEDED",
        }
        for index, (variant, invariant_id) in enumerate(expected.items()):
            with self.subTest(variant=variant):
                pair = execute_variant(variant, 2510 + index)
                report = validate(pair)
                self.assertEqual(
                    (invariant_id,), report.trace_validated_invariant_ids[pair[1].trace_id]
                )

    def test_identity_and_binding_truth_remain_separate(self):
        cases = {
            "authenticated_identity_mismatch_persistent_device":
                "INV-IDENT-AUTH-ACTOR-CONSISTENCY",
            "token_bound_identity_mismatch_authenticated_context":
                "INV-BIND-TOKEN-CONTEXT",
        }
        for index, (variant, invariant_id) in enumerate(cases.items()):
            pair = execute_variant(variant, 2520 + index)
            self.assertEqual(
                (invariant_id,), validate(pair).trace_validated_invariant_ids[pair[1].trace_id]
            )

    def test_composite_access_operation_preserves_three_atomic_ids(self):
        pair = execute_variant("access_success_without_security_prerequisites", 2530)
        report = validate(pair)
        self.assertEqual(
            set(EXACT_VARIANT_TO_INVARIANTS[
                "access_success_without_security_prerequisites"
            ]),
            set(report.trace_validated_invariant_ids[pair[1].trace_id]),
        )

    def test_all_emitted_ids_exist_in_twenty_entry_registry(self):
        self.assertEqual(20, len(INVARIANT_REGISTRY))
        for index, variant in enumerate(EXACT_VARIANT_TO_INVARIANTS):
            pair = execute_variant(variant, 2540 + index)
            emitted = validate(pair).trace_validated_invariant_ids[pair[1].trace_id]
            self.assertTrue(set(emitted).issubset(INVARIANT_REGISTRY))

    def test_historical_vocabulary_is_explicitly_non_authoritative(self):
        compatibility = synchronized_capability_report()[
            "historical_compatibility_vocabulary"
        ]
        self.assertEqual(7, len(compatibility))
        self.assertTrue(all(not item["determines_invariant_truth"] for item in compatibility.values()))

    def test_registry_report_has_complete_c2_5_status_columns(self):
        for item in invariant_registry_report()["invariants"]:
            for key in (
                "implementation_status",
                "relation_capabilities",
                "current_checks",
                "synchronized_anomaly_variants",
                "minimum_evidence",
                "scope_caveat",
                "version_introduced",
                "implementation_version",
            ):
                self.assertIn(key, item)


if __name__ == "__main__":
    unittest.main()
