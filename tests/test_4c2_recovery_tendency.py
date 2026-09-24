"""4C.2 recovery tendency, fixed-slot, composition, and boundary tests."""

from __future__ import annotations

import json
import math
import random
import unittest
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import FIELD_MANIFEST_VERSION
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.device_profiles.population import build_profile_population
from simulator.device_profiles.profile import LifecycleWeights
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import NORMAL_FLOW, NORMAL_FLOW_WITH_RETRY
from simulator.event_model import EventResult, EventType
from simulator.planning.models import (
    ApplicableChoices,
    ApplicableRecoveryChoices,
    ConstructionRole,
    LifecycleDepth,
    LifecycleSelection,
    RecoveryChoice,
    TraceConstructionRequest,
    TracePlanningError,
    TracePlanningErrorReason,
)
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
    adapt_trace_plan,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    select_profile_conditioned_recovery,
)
from simulator.planning.randomness import (
    PLANNING_DECISION_SLOT_ORDER_VERSION,
    PlanningDecisionSlot,
    PlanningRandomness,
    derive_planning_randomness,
)
from simulator.planning.repertoire import (
    AUTHENTICATION_RECOVERY_REPERTOIRE,
    LIFECYCLE_REPERTOIRE_VERSION,
    RECOVERY_REPERTOIRE_VERSION,
    compose_ordinary_lifecycle,
)
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.runner import run_simulation_with_population
from simulator.security_context import PersistentDeviceContext


def lifecycle_applicable() -> ApplicableChoices:
    return ApplicableChoices(tuple(LifecycleDepth), LIFECYCLE_REPERTOIRE_VERSION)


def recovery_applicable(
    depth: LifecycleDepth = LifecycleDepth.AUTH,
    *choices: RecoveryChoice,
) -> ApplicableRecoveryChoices:
    return ApplicableRecoveryChoices(
        lifecycle_depth=depth,
        recovery_choices=choices or (RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY),
        supported_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
    )


def request(key: str) -> TraceConstructionRequest:
    return TraceConstructionRequest.profile_conditioned_lifecycle(
        trace_construction_key=key,
        scenario_id=f"scenario-{key}",
    )


def randomness(key: str = "trace", device: str = "device", seed: int = 4102):
    return derive_planning_randomness(
        planning_root_seed=seed,
        population_device_key=device,
        trace_construction_key=key,
    )


class FixedSlotRandomness:
    def __init__(self, recovery_value: float):
        self.recovery_value = recovery_value
        self.requested_slots: list[int] = []

    def variate_at(self, slot: int) -> float:
        self.requested_slots.append(int(slot))
        if int(slot) != int(PlanningDecisionSlot.RECOVERY):
            raise AssertionError("recovery may read only fixed slot 2")
        return self.recovery_value


def selection(depth: LifecycleDepth) -> LifecycleSelection:
    return LifecycleSelection(
        applicable_lifecycle_depths=tuple(LifecycleDepth),
        normalized_weights=tuple((item, 1 / 3) for item in LifecycleDepth),
        selected_lifecycle_depth=depth,
        random_variate=0.25,
    )


class RecoveryMechanismTests(unittest.TestCase):
    def test_exact_repertoire_and_compositional_event_sequences(self):
        self.assertEqual(
            {"DIRECT", "ONE_RETRY"}, {choice.value for choice in RecoveryChoice},
        )
        self.assertEqual(2, len(AUTHENTICATION_RECOVERY_REPERTOIRE))
        self.assertEqual(tuple(NORMAL_FLOW[:11]), compose_ordinary_lifecycle(
            LifecycleDepth.AUTH, RecoveryChoice.DIRECT,
        ))
        self.assertEqual(tuple(NORMAL_FLOW_WITH_RETRY[:16]), compose_ordinary_lifecycle(
            LifecycleDepth.AUTH, RecoveryChoice.ONE_RETRY,
        ))
        self.assertEqual(tuple(NORMAL_FLOW[:15]), compose_ordinary_lifecycle(
            LifecycleDepth.SESSION, RecoveryChoice.DIRECT,
        ))
        self.assertEqual(tuple(NORMAL_FLOW_WITH_RETRY[:20]), compose_ordinary_lifecycle(
            LifecycleDepth.SESSION, RecoveryChoice.ONE_RETRY,
        ))
        self.assertEqual(tuple(NORMAL_FLOW[:17]), compose_ordinary_lifecycle(
            LifecycleDepth.ACCESS, RecoveryChoice.DIRECT,
        ))
        self.assertEqual(tuple(NORMAL_FLOW_WITH_RETRY[:22]), compose_ordinary_lifecycle(
            LifecycleDepth.ACCESS, RecoveryChoice.ONE_RETRY,
        ))

    def test_recovery_boundaries_and_half_open_rule(self):
        fixtures = (
            (0.0, 0.0, RecoveryChoice.DIRECT),
            (0.0, 0.9, RecoveryChoice.DIRECT),
            (1.0, 0.0, RecoveryChoice.ONE_RETRY),
            (1.0, math.nextafter(1.0, 0.0), RecoveryChoice.ONE_RETRY),
            (0.4, math.nextafter(0.4, 0.0), RecoveryChoice.ONE_RETRY),
            (0.4, 0.4, RecoveryChoice.DIRECT),
            (0.4, math.nextafter(0.4, 1.0), RecoveryChoice.DIRECT),
        )
        for tendency, draw, expected in fixtures:
            with self.subTest(tendency=tendency, draw=draw):
                source = FixedSlotRandomness(draw)
                result = select_profile_conditioned_recovery(
                    tendency, recovery_applicable(), source,
                )
                self.assertIs(expected, result.selected_recovery_choice)
                self.assertEqual([2], source.requested_slots)
                self.assertTrue(result.stochastic_decision_used)

    def test_malformed_recovery_tendency_is_rejected_without_coercion(self):
        malformed = (True, -0.01, 1.01, math.nan, math.inf, -math.inf, None, "0.5")
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(TracePlanningError) as raised:
                select_profile_conditioned_recovery(
                    value, recovery_applicable(), FixedSlotRandomness(0.5),
                )
            self.assertIs(
                TracePlanningErrorReason.MALFORMED_RECOVERY_TENDENCY,
                raised.exception.reason,
            )

    def test_singleton_empty_and_unsupported_applicability(self):
        singleton = ApplicableRecoveryChoices(
            LifecycleDepth.AUTH, (RecoveryChoice.DIRECT,), RECOVERY_REPERTOIRE_VERSION,
        )
        source = FixedSlotRandomness(0.0)
        chosen = select_profile_conditioned_recovery(1.0, singleton, source)
        self.assertIs(RecoveryChoice.DIRECT, chosen.selected_recovery_choice)
        self.assertFalse(chosen.stochastic_decision_used)
        self.assertEqual([], source.requested_slots)

        empty = ApplicableRecoveryChoices(
            LifecycleDepth.AUTH, (), RECOVERY_REPERTOIRE_VERSION, ("blocked",),
        )
        with self.assertRaises(TracePlanningError) as raised:
            select_profile_conditioned_recovery(0.5, empty, source)
        self.assertIs(TracePlanningErrorReason.NO_APPLICABLE_RECOVERY_CHOICE,
                      raised.exception.reason)

        duplicate = ApplicableRecoveryChoices(
            LifecycleDepth.AUTH,
            (RecoveryChoice.DIRECT, RecoveryChoice.DIRECT),
            RECOVERY_REPERTOIRE_VERSION,
        )
        with self.assertRaises(TracePlanningError) as raised:
            select_profile_conditioned_recovery(0.5, duplicate, source)
        self.assertIs(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
            raised.exception.reason,
        )


class ApplicabilityAndSlotTests(unittest.TestCase):
    def test_recovery_applicability_is_profile_and_rng_independent(self):
        resolver = ApplicabilityResolver()
        context = PersistentDeviceContext(device_id="device")
        for depth in LifecycleDepth:
            before = resolver.resolve_recovery(context, ConstructionRole.ORDINARY, depth)
            for _recovery_tendency in (0.0, 0.2, 0.8, 1.0):
                self.assertEqual(
                    before,
                    resolver.resolve_recovery(context, ConstructionRole.ORDINARY, depth),
                )
                self.assertEqual(
                    (RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY),
                    before.recovery_choices,
                )
        blocked = PersistentDeviceContext(device_id="blocked", blocked=True)
        self.assertEqual(
            (), resolver.resolve_recovery(
                blocked, ConstructionRole.ORDINARY, LifecycleDepth.AUTH,
            ).recovery_choices,
        )

    def test_fixed_slots_preserve_lifecycle_and_do_not_collapse(self):
        source = randomness("fixed-slots")
        stream = source.new_rng()
        first = stream.random()
        second = stream.random()
        third = stream.random()
        self.assertEqual(first, source.variate_at(PlanningDecisionSlot.LIFECYCLE))
        self.assertEqual(second, source.variate_at(PlanningDecisionSlot.RECOVERY))
        self.assertEqual(third, source.variate_at(3))

        weights = LifecycleWeights(0.2, 0.3, 0.5)
        before = reconstruct_profile_conditioned_lifecycle(
            weights, lifecycle_applicable(), source,
        )
        reconstruct_profile_conditioned_recovery(
            0.6, recovery_applicable(before.selected_lifecycle_depth), source,
        )
        after = reconstruct_profile_conditioned_lifecycle(
            weights, lifecycle_applicable(), source,
        )
        self.assertEqual(before, after)
        self.assertEqual(third, source.variate_at(3))

        singleton = recovery_applicable(
            before.selected_lifecycle_depth, RecoveryChoice.DIRECT,
        )
        reconstruct_profile_conditioned_recovery(1.0, singleton, source)
        self.assertEqual(third, source.variate_at(3))

    def test_frozen_4c1_characterization_lifecycle_counts_remain_exact(self):
        population = build_profile_population(
            [(f"campaign-device-{i}", f"campaign-population-{i:03d}") for i in range(8)],
            PopulationConfiguration(population_seed=4101),
        )
        expected = (
            (2, 20, 2), (9, 5, 10), (5, 11, 8), (14, 1, 9),
            (19, 2, 3), (13, 6, 5), (1, 15, 8), (5, 10, 9),
        )
        for assignment, frozen in zip(population.assignments, expected):
            counts = Counter()
            for index in range(24):
                key = f"{assignment.population_device_key}-trace-{index:03d}"
                source = randomness(key, assignment.population_device_key)
                selected = reconstruct_profile_conditioned_lifecycle(
                    assignment.profile.lifecycle, lifecycle_applicable(), source,
                ).selected_lifecycle_depth
                counts[selected] += 1
                # Recovery owns slot 2 but cannot mutate or advance slot 1.
                reconstruct_profile_conditioned_recovery(
                    assignment.profile.recovery,
                    recovery_applicable(selected), source,
                )
                self.assertIs(
                    selected,
                    reconstruct_profile_conditioned_lifecycle(
                        assignment.profile.lifecycle, lifecycle_applicable(), source,
                    ).selected_lifecycle_depth,
                )
            self.assertEqual(
                frozen,
                tuple(counts[depth] for depth in LifecycleDepth),
            )


class CompositionIntegrityTests(unittest.TestCase):
    def _plan(self, depth: LifecycleDepth, tendency: float, key: str):
        source = randomness(key)
        return DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_and_recovery(
            request(key), selection(depth), recovery_applicable(depth), tendency, source,
        )

    def test_direct_and_retry_compose_with_all_lifecycle_endpoints(self):
        expected_endpoints = {
            LifecycleDepth.AUTH: EventType.AUTHENTICATION_SUCCESS,
            LifecycleDepth.SESSION: EventType.SESSION_OPENED,
            LifecycleDepth.ACCESS: EventType.ACCESS_GRANTED,
        }
        for index, depth in enumerate(LifecycleDepth):
            for tendency, expected_recovery in (
                (0.0, RecoveryChoice.DIRECT), (1.0, RecoveryChoice.ONE_RETRY),
            ):
                with self.subTest(depth=depth, tendency=tendency):
                    plan = self._plan(depth, tendency, f"compose-{index}-{tendency}")
                    context = PersistentDeviceContext(device_id=f"device-{index}-{tendency}")
                    random.seed(5200 + index)
                    pair = EventEngine(
                        device_id=context.device_id, gateway_id="gateway",
                        auth_server_id="auth", broker_id="broker",
                        source_ip="10.0.0.1", battery_level=80,
                        firmware_version="1", persistent_context=context,
                        start_time=1000.0,
                    ).execute(adapt_trace_plan(plan))
                    events, trace = pair
                    bundle = build_synchronized_bundle([pair])
                    self.assertIs(expected_recovery, plan.recovery_choice)
                    self.assertIs(expected_endpoints[depth], events[-1].event_type)
                    self.assertEqual("validated_normal",
                                     bundle["validation"]["trace_ground_truth_statuses"][trace.trace_id])
                    self.assertTrue(bundle["validation"]["build_gate"]["passed"])
                    failures = [event for event in events
                                if event.event_type is EventType.AUTHENTICATION_FAILURE]
                    retries = [event for event in events if event.event_type is EventType.RETRY]
                    if expected_recovery is RecoveryChoice.ONE_RETRY:
                        self.assertEqual(2, len(trace.auth_attempt_ids))
                        self.assertEqual(1, len(failures))
                        self.assertEqual(1, len(retries))
                        self.assertIs(EventResult.FAILURE, failures[0].result)
                        successes = [event for event in events
                                     if event.event_type is EventType.AUTHENTICATION_SUCCESS]
                        self.assertNotEqual(failures[0].auth_attempt_id,
                                            successes[0].auth_attempt_id)
                        nonces = [event for event in events
                                  if event.event_type is EventType.NONCE_RECEIVED]
                        self.assertEqual(2, len(nonces))
                        self.assertNotEqual(nonces[0].nonce, nonces[1].nonce)
                        self.assertNotEqual(nonces[0].auth_attempt_id,
                                            nonces[1].auth_attempt_id)
                    else:
                        self.assertEqual(1, len(trace.auth_attempt_ids))
                        self.assertEqual([], failures)
                        self.assertEqual([], retries)

    def test_mismatched_lifecycle_recovery_applicability_fails(self):
        with self.assertRaises(TracePlanningError) as raised:
            DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_and_recovery(
                request("mismatch"), selection(LifecycleDepth.ACCESS),
                recovery_applicable(LifecycleDepth.AUTH), 0.5, randomness("mismatch"),
            )
        self.assertIs(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
            raised.exception.reason,
        )


class RunnerProvenanceBoundaryTests(unittest.TestCase):
    def _bundle(self):
        requests = [request(f"runner-{index:03d}") for index in range(24)]
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345,
                "num_devices": 1,
                "num_sessions_normal": 0,
                "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            sequences, population = run_simulation_with_population(
                planning_requests=requests, planning_root_seed=700,
            )
            bundle = build_synchronized_bundle(
                sequences, run_population=population,
            )
        return sequences, population, bundle

    def test_runner_profile_persistence_provenance_and_information_boundary(self):
        sequences, population, bundle = self._bundle()
        assignment = population.behavioral_profiles.assignments[0]
        profile = assignment.profile
        persistent_context = population.semantic_contexts[assignment.device_id]
        self.assertIs(profile, population.behavioral_profiles.profile_for_device(
            assignment.device_id,
        ))
        self.assertEqual(24, persistent_context.completed_trace_count)
        self.assertTrue(persistent_context.enrolled)
        self.assertFalse(persistent_context.blocked)
        self.assertFalse(persistent_context.revoked)
        previous_end = None
        for events, _trace in sequences:
            if previous_end is not None:
                self.assertGreaterEqual(events[0].timestamp, previous_end)
            previous_end = events[-1].timestamp
        self.assertEqual(previous_end, persistent_context.semantic_time_cursor)
        self.assertIs(profile, population.behavioral_profiles.profile_for_device(
            assignment.device_id,
        ))
        records = [trace["trace_planning"] for trace in bundle["provenance"]["traces"]]
        self.assertEqual(24, len(records))
        self.assertEqual(
            {"DIRECT", "ONE_RETRY"},
            {record["selected_recovery_choice"] for record in records},
        )
        for record in records:
            self.assertEqual("4c5-trace-planning-provenance-v6",
                             record["planning_provenance_schema_version"])
            self.assertEqual(PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
                             record["planner_version"])
            self.assertEqual(RECOVERY_REPERTOIRE_VERSION,
                             record["supported_recovery_repertoire_version"])
            self.assertEqual(["DIRECT", "ONE_RETRY"],
                             record["applicable_recovery_choices"])
            self.assertEqual(2, record["recovery_decision_slot"])
            self.assertTrue(record["recovery_decision_stochastic"])
            self.assertTrue(record["profile_conditioned_recovery_selection"])
            self.assertIn(record["planning_draw_count"], (2, 3, 4))
            self.assertEqual(PLANNING_DECISION_SLOT_ORDER_VERSION,
                             record["planning_decision_slot_order_version"])
            for forbidden in (
                "recovery_tendency", "recovery_probability", "recovery_variate",
                "behavioral_profile_id", "population_device_key",
            ):
                self.assertNotIn(forbidden, record)

        self.assertEqual({"validated_normal"}, set(
            bundle["validation"]["trace_ground_truth_statuses"].values(),
        ))
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        for surface in ("detector_observations", "ground_truth", "debug"):
            serialized = json.dumps(bundle[surface])
            for forbidden in (
                "selected_recovery_choice", "applicable_recovery_choices",
                "recovery_decision_slot", "profile_conditioned_recovery_selection",
                "planning_rng_reference", "behavioral_profile_id",
                "population_device_key",
            ):
                self.assertNotIn(forbidden, serialized)
        companion = json.dumps(bundle["device_profile_provenance"])
        self.assertNotIn("trace_planning", companion)
        self.assertNotIn("selected_recovery_choice", companion)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        self.assertEqual("sync-v1.1.0+4c4.activity-field-manifest-v5",
                         FIELD_MANIFEST_VERSION)
        self.assertEqual(24, len(sequences))


if __name__ == "__main__":
    unittest.main()
