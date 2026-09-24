"""C2.1 scoped superseded-token replay generation and adversarial validation."""

import copy
import random
import unittest
from unittest.mock import patch

from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.event_model import EventResult, EventType
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset


VARIANT = "token_replay_after_replacement"
CHECK = "c2.1.replay_token.superseded_within_continuing_session"


def generated_pair(seed=2101, *, stealth=0.0):
    random.seed(seed)
    with patch.dict(STEALTH_FRACTION, {VARIANT: stealth}):
        spec = ScenarioEngine(seed).synchronized_anomaly(VARIANT)
        persistent = PersistentDeviceContext(device_id="device-c2-replay")
        return EventEngine(
            device_id=persistent.device_id,
            gateway_id="gateway-c2-replay",
            auth_server_id="auth-c2-replay",
            broker_id="broker-c2-replay",
            start_time=30_000.0,
            persistent_context=persistent,
        ).execute(spec)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2.1-replay-test", generator_git_commit="c2.1-test"
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


def replay_check(report):
    return next(record for record in report.records if record.validation_id == CHECK)


def replay_events(events):
    replay = next(
        event for event in events
        if event.anomaly_variant_name == VARIANT
        and event.event_type == EventType.TOKEN_PRESENTED
    )
    rejection = next(
        event for event in events
        if event.anomaly_variant_name == VARIANT
        and event.event_type == EventType.TOKEN_REJECTED
    )
    return replay, rejection


class SupersededTokenReplayTests(unittest.TestCase):
    def test_genuine_t1_t2_reuse_t1_is_confirmed(self):
        events, context = generated_pair()
        issued = [
            event for event in events
            if event.event_type == EventType.TOKEN_ISSUED
            and event.result is EventResult.SUCCESS
        ]
        replay, _rejection = replay_events(events)

        self.assertEqual(2, len(issued))
        t1, t2 = issued
        self.assertNotEqual(t1.token_id, t2.token_id)
        self.assertEqual(t1.token_id, t2.supersedes_token_id)
        self.assertEqual(t1.token_id, replay.token_id)
        self.assertEqual(t2.token_id, replay.current_token_id)
        self.assertEqual("superseded", replay.token_lineage_status)
        self.assertEqual(t2.token_id, replay.superseded_by_token_id)
        self.assertEqual(t2.timestamp, replay.token_replacement_timestamp)
        self.assertEqual(t2.protected_session_id, replay.protected_session_id)

        report = validate((events, context))
        self.assertEqual(ValidationResult.PASS, replay_check(report).result)
        self.assertTrue(report.build_gate_passed)

    def test_metadata_cannot_confirm_when_later_token_is_t2(self):
        events, context = copy.deepcopy(generated_pair(2102))
        issued = [e for e in events if e.event_type == EventType.TOKEN_ISSUED]
        replay, rejection = replay_events(events)
        replay.token_id = issued[-1].token_id
        rejection.token_id = issued[-1].token_id

        report = validate((events, context))
        self.assertEqual(ValidationResult.FAIL, replay_check(report).result)
        self.assertTrue(context.injection_records[0]["transformation_applied"])
        self.assertFalse(report.build_gate_passed)

    def test_repeated_t1_before_replacement_is_not_scoped_violation(self):
        events, context = copy.deepcopy(generated_pair(2103))
        replay, rejection = replay_events(events)
        events.remove(replay)
        events.remove(rejection)
        replacement_index = next(
            index for index, event in enumerate(events)
            if event.event_type == EventType.TOKEN_ISSUED
            and event.supersedes_token_id
        )
        previous = events[replacement_index - 1]
        replacement = events[replacement_index]
        replay.timestamp = (previous.timestamp + replacement.timestamp) / 2.0
        events.insert(replacement_index, replay)

        report = validate((events, context))
        self.assertEqual(ValidationResult.FAIL, replay_check(report).result)

    def test_distinct_old_or_expired_token_is_not_replay(self):
        events, context = copy.deepcopy(generated_pair(2104))
        replay, rejection = replay_events(events)
        replay.token_id = "distinct-expired-token"
        replay.token_expiry = replay.timestamp - 1.0
        rejection.token_id = replay.token_id

        report = validate((events, context))
        self.assertEqual(ValidationResult.FAIL, replay_check(report).result)

    def test_failed_replacement_does_not_establish_supersession(self):
        events, context = copy.deepcopy(generated_pair(2105))
        replacement = next(
            event for event in events
            if event.event_type == EventType.TOKEN_ISSUED
            and event.supersedes_token_id
        )
        replacement.result = EventResult.FAILURE
        replacement.failure_reason = "forced_replacement_failure"

        report = validate((events, context))
        self.assertEqual(ValidationResult.FAIL, replay_check(report).result)

    def test_different_protected_sessions_are_outside_supported_scope(self):
        events, context = copy.deepcopy(generated_pair(2106))
        replay, rejection = replay_events(events)
        replay.protected_session_id = "different-protected-session"
        rejection.protected_session_id = replay.protected_session_id

        report = validate((events, context))
        self.assertEqual(ValidationResult.FAIL, replay_check(report).result)

    def test_stealth_intent_without_actual_reuse_remains_normal(self):
        events, context = generated_pair(2107, stealth=1.0)
        report = validate((events, context))

        self.assertTrue(context.attack_scenario_intent)
        self.assertFalse(context.injection_records)
        self.assertFalse(context.observable_anomaly)
        self.assertTrue(all(event.anomaly_label is None for event in events))
        self.assertEqual("validated_normal", report.trace_ground_truth_statuses[context.trace_id])
        self.assertTrue(report.build_gate_passed)

    def test_rejected_replayed_token_still_confirms_replay_behavior(self):
        events, context = generated_pair(2108)
        replay, rejection = replay_events(events)

        self.assertLess(replay.timestamp, rejection.timestamp)
        self.assertEqual(EventResult.FAILURE, rejection.result)
        self.assertEqual("rejected", rejection.token_validation_result)
        self.assertEqual(replay.token_id, rejection.token_id)
        self.assertNotEqual(rejection.token_id, rejection.current_token_id)
        report = validate((events, context))
        self.assertEqual(ValidationResult.PASS, replay_check(report).result)


if __name__ == "__main__":
    unittest.main()
