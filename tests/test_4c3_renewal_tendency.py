"""4C.3 renewal tendency, applicability, composition, and boundary tests."""

from __future__ import annotations

import math
import random
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import FIELD_MANIFEST_VERSION
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.device_profiles.population import build_profile_population
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import NORMAL_FLOW_WITH_RENEWAL
from simulator.event_model import EventType
from simulator.planning.models import (
    ApplicableRenewalChoices,
    ConstructionRole,
    LifecycleDepth,
    LifecycleSelection,
    RecoveryChoice,
    RecoverySelection,
    RenewalChoice,
    TraceConstructionRequest,
    TracePlanningError,
    TracePlanningErrorReason,
)
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    PROFILE_CONDITIONED_RENEWAL_PLANNER_VERSION,
    adapt_trace_plan,
    reconstruct_profile_conditioned_renewal,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    select_profile_conditioned_renewal,
)
from simulator.planning.randomness import (
    PLANNING_DECISION_SLOT_ORDER_VERSION,
    PlanningDecisionSlot,
    derive_planning_randomness,
)
from simulator.planning.repertoire import (
    RENEWAL_REPERTOIRE_VERSION,
    compose_ordinary_lifecycle,
    compose_ordinary_trace,
)
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.runner import run_simulation_with_population
from simulator.security_context import PersistentDeviceContext


def request(key: str = "trace") -> TraceConstructionRequest:
    return TraceConstructionRequest.profile_conditioned_lifecycle(
        trace_construction_key=key, scenario_id=f"scenario-{key}",
    )


def planning_randomness(key: str = "trace"):
    return derive_planning_randomness(
        planning_root_seed=4102,
        population_device_key="population-device-000",
        trace_construction_key=key,
    )


def lifecycle_selection(depth: LifecycleDepth) -> LifecycleSelection:
    return LifecycleSelection(
        applicable_lifecycle_depths=tuple(LifecycleDepth),
        normalized_weights=tuple((item, 1 / 3) for item in LifecycleDepth),
        selected_lifecycle_depth=depth,
        random_variate=0.25,
    )


def recovery_selection(choice: RecoveryChoice) -> RecoverySelection:
    return RecoverySelection(
        applicable_recovery_choices=(RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY),
        selected_recovery_choice=choice,
        decision_slot=2,
        stochastic_decision_used=True,
    )


def renewal_applicable(
    depth: LifecycleDepth = LifecycleDepth.ACCESS,
    recovery: RecoveryChoice = RecoveryChoice.DIRECT,
    choices: tuple[RenewalChoice, ...] = (
        RenewalChoice.NO_RENEWAL, RenewalChoice.RENEWAL,
    ),
) -> ApplicableRenewalChoices:
    return ApplicableRenewalChoices(
        lifecycle_depth=depth,
        recovery_choice=recovery,
        renewal_choices=choices,
        supported_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
    )


class FixedSlotRandomness:
    def __init__(self, value: float):
        self.value = value
        self.requested_slots: list[int] = []

    def variate_at(self, slot: int) -> float:
        self.requested_slots.append(int(slot))
        if int(slot) != 3:
            raise AssertionError("renewal may read only fixed slot 3")
        return self.value


class RenewalMechanismTests(unittest.TestCase):
    def test_exact_choices_suffix_and_composition(self):
        self.assertEqual({"NO_RENEWAL", "RENEWAL"}, {item.value for item in RenewalChoice})
        for recovery in RecoveryChoice:
            base = compose_ordinary_lifecycle(LifecycleDepth.ACCESS, recovery)
            self.assertEqual(base, compose_ordinary_trace(
                LifecycleDepth.ACCESS, recovery, RenewalChoice.NO_RENEWAL,
            ))
            renewed = compose_ordinary_trace(
                LifecycleDepth.ACCESS, recovery, RenewalChoice.RENEWAL,
            )
            self.assertEqual(tuple(NORMAL_FLOW_WITH_RENEWAL[17:]), renewed[len(base):])
        with self.assertRaises(KeyError):
            compose_ordinary_trace(
                LifecycleDepth.SESSION, RecoveryChoice.DIRECT, RenewalChoice.RENEWAL,
            )

    def test_boundaries_and_half_open_rule_use_slot_three(self):
        fixtures = (
            (0.0, 0.0, RenewalChoice.NO_RENEWAL),
            (1.0, math.nextafter(1.0, 0.0), RenewalChoice.RENEWAL),
            (0.4, math.nextafter(0.4, 0.0), RenewalChoice.RENEWAL),
            (0.4, 0.4, RenewalChoice.NO_RENEWAL),
            (0.4, math.nextafter(0.4, 1.0), RenewalChoice.NO_RENEWAL),
        )
        for tendency, draw, expected in fixtures:
            source = FixedSlotRandomness(draw)
            result = select_profile_conditioned_renewal(
                tendency, renewal_applicable(), source,
            )
            self.assertIs(expected, result.selected_renewal_choice)
            self.assertEqual([3], source.requested_slots)

    def test_malformed_tendency_rejected_without_coercion(self):
        malformed = (True, -0.01, 1.01, math.nan, math.inf, -math.inf, None, "0.5")
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(TracePlanningError) as raised:
                select_profile_conditioned_renewal(
                    value, renewal_applicable(), FixedSlotRandomness(0.5),
                )
            self.assertIs(
                TracePlanningErrorReason.MALFORMED_RENEWAL_TENDENCY,
                raised.exception.reason,
            )

    def test_singleton_is_deterministic_but_still_owns_slot_three(self):
        source = FixedSlotRandomness(0.0)
        result = select_profile_conditioned_renewal(
            1.0,
            renewal_applicable(
                LifecycleDepth.AUTH,
                choices=(RenewalChoice.NO_RENEWAL,),
            ),
            source,
        )
        self.assertIs(RenewalChoice.NO_RENEWAL, result.selected_renewal_choice)
        self.assertFalse(result.stochastic_decision_used)
        self.assertEqual(3, result.decision_slot)
        self.assertEqual([], source.requested_slots)


class ApplicabilityAndCompositionTests(unittest.TestCase):
    def test_applicability_is_independent_of_n_and_restricted_by_depth(self):
        resolver = ApplicabilityResolver()
        context = PersistentDeviceContext(device_id="device")
        for depth in LifecycleDepth:
            base = compose_ordinary_lifecycle(depth, RecoveryChoice.DIRECT)
            before = resolver.resolve_renewal(
                context, ConstructionRole.ORDINARY, depth, RecoveryChoice.DIRECT, base,
            )
            expected = (
                (RenewalChoice.NO_RENEWAL, RenewalChoice.RENEWAL)
                if depth is LifecycleDepth.ACCESS
                else (RenewalChoice.NO_RENEWAL,)
            )
            self.assertEqual(expected, before.renewal_choices)
            for _n in (0.0, 0.3, 1.0):
                self.assertEqual(before, resolver.resolve_renewal(
                    context, ConstructionRole.ORDINARY, depth,
                    RecoveryChoice.DIRECT, base,
                ))
        malformed = resolver.resolve_renewal(
            context, ConstructionRole.ORDINARY, LifecycleDepth.ACCESS,
            RecoveryChoice.DIRECT, (),
        )
        self.assertEqual((), malformed.renewal_choices)

    def _plan(self, recovery: RecoveryChoice, tendency: float, key: str):
        depth = LifecycleDepth.ACCESS
        source = planning_randomness(key)
        applicable = renewal_applicable(depth, recovery)
        return DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_recovery_and_renewal(
            request(key), lifecycle_selection(depth), recovery_selection(recovery),
            applicable, tendency, source,
        )

    def test_direct_and_retry_compose_with_and_without_renewal(self):
        for recovery in RecoveryChoice:
            for tendency, expected in (
                (0.0, RenewalChoice.NO_RENEWAL), (1.0, RenewalChoice.RENEWAL),
            ):
                key = f"{recovery.value}-{tendency}"
                plan = self._plan(recovery, tendency, key)
                context = PersistentDeviceContext(device_id=key)
                random.seed(5200)
                pair = EventEngine(
                    device_id=key, gateway_id="gateway", auth_server_id="auth",
                    broker_id="broker", source_ip="10.0.0.1", battery_level=80,
                    firmware_version="1", persistent_context=context, start_time=1000.0,
                ).execute(adapt_trace_plan(plan))
                events, trace = pair
                bundle = build_synchronized_bundle([pair])
                self.assertIs(expected, plan.renewal_choice)
                self.assertEqual("validated_normal",
                    bundle["validation"]["trace_ground_truth_statuses"][trace.trace_id])
                self.assertTrue(bundle["validation"]["build_gate"]["passed"])
                if expected is RenewalChoice.RENEWAL:
                    self.assertIs(EventType.SESSION_CLOSED, events[-1].event_type)
                    self.assertEqual(1, sum(
                        event.event_type is EventType.SESSION_OPENED for event in events
                    ))
                    self.assertEqual(1, sum(
                        event.event_type is EventType.ACCESS_GRANTED for event in events
                    ))
                    self.assertEqual(1, len(trace.renewal_auth_attempt_ids))
                    self.assertEqual(
                        2 if recovery is RecoveryChoice.DIRECT else 3,
                        len(trace.auth_attempt_ids),
                    )
                    issued = [event for event in events
                              if event.event_type is EventType.TOKEN_ISSUED]
                    self.assertEqual([1, 2], [event.token_generation for event in issued])
                    self.assertNotEqual(issued[0].token_id, issued[1].token_id)
                    self.assertEqual(issued[0].token_id, issued[1].supersedes_token_id)
                    session_ids = {event.protected_session_id for event in events
                                   if event.protected_session_id}
                    self.assertEqual(1, len(session_ids))
                    nonces = [event.nonce for event in events
                              if event.event_type is EventType.NONCE_RECEIVED]
                    self.assertEqual(len(nonces), len(set(nonces)))
                    after_t2 = events[events.index(issued[1]) + 1:]
                    self.assertNotIn(issued[0].token_id,
                                     [event.token_id for event in after_t2])
                    self.assertFalse(any(
                        event.event_type is EventType.ACCESS_REQUEST
                        for event in after_t2
                    ))

    def test_provenance_and_information_boundary(self):
        plan = self._plan(RecoveryChoice.DIRECT, 1.0, "provenance")
        record = plan.provenance_record()
        self.assertEqual("4c5-trace-planning-provenance-v6",
                         record["planning_provenance_schema_version"])
        self.assertEqual(PROFILE_CONDITIONED_RENEWAL_PLANNER_VERSION,
                         record["planner_version"])
        self.assertEqual(RENEWAL_REPERTOIRE_VERSION,
                         record["supported_renewal_repertoire_version"])
        self.assertEqual(["NO_RENEWAL", "RENEWAL"],
                         record["applicable_renewal_choices"])
        self.assertEqual("RENEWAL", record["selected_renewal_choice"])
        self.assertEqual(3, record["renewal_decision_slot"])
        self.assertEqual("access_granted", record["base_lifecycle_endpoint"])
        self.assertEqual("session_closed", record["final_planned_endpoint"])
        self.assertEqual(PLANNING_DECISION_SLOT_ORDER_VERSION,
                         record["planning_decision_slot_order_version"])
        for forbidden in (
            "renewal_tendency", "renewal_probability", "renewal_variate",
            "behavioral_profile_id", "population_device_key",
        ):
            self.assertNotIn(forbidden, record)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        self.assertEqual("sync-v1.1.0+4c4.activity-field-manifest-v5",
                         FIELD_MANIFEST_VERSION)

    def test_planning_slots_reconstruct_independently(self):
        source = planning_randomness("fixed-slots")
        stream = source.new_rng()
        draws = [stream.random() for _ in range(3)]
        self.assertEqual(draws[0], source.variate_at(PlanningDecisionSlot.LIFECYCLE))
        self.assertEqual(draws[1], source.variate_at(PlanningDecisionSlot.RECOVERY))
        self.assertEqual(draws[2], source.variate_at(PlanningDecisionSlot.RENEWAL))
        result = reconstruct_profile_conditioned_renewal(
            0.5, renewal_applicable(), source,
        )
        expected = RenewalChoice.RENEWAL if draws[2] < 0.5 else RenewalChoice.NO_RENEWAL
        self.assertIs(expected, result.selected_renewal_choice)

    def test_192_trace_upstream_choices_and_slot_three_reconstruct(self):
        population = build_profile_population(
            [(f"campaign-device-{index}", f"campaign-population-{index:03d}")
             for index in range(8)],
            PopulationConfiguration(population_seed=4101),
        )
        resolver = ApplicabilityResolver()
        lifecycle_applicable = resolver.resolve(
            PersistentDeviceContext(device_id="applicability"),
            ConstructionRole.ORDINARY,
        )
        checked = 0
        for assignment in population.assignments:
            profile = assignment.profile
            context = PersistentDeviceContext(device_id=assignment.device_id)
            for index in range(24):
                key = f"{assignment.population_device_key}-trace-{index:03d}"
                source = derive_planning_randomness(
                    planning_root_seed=4102,
                    population_device_key=assignment.population_device_key,
                    trace_construction_key=key,
                )
                lifecycle = reconstruct_profile_conditioned_lifecycle(
                    profile.lifecycle, lifecycle_applicable, source,
                )
                recovery_applicable = resolver.resolve_recovery(
                    context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                )
                recovery = reconstruct_profile_conditioned_recovery(
                    profile.recovery, recovery_applicable, source,
                )
                renewal_choices = resolver.resolve_renewal(
                    context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                    compose_ordinary_lifecycle(
                        lifecycle.selected_lifecycle_depth,
                        recovery.selected_recovery_choice,
                    ),
                )
                plan = DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_recovery_and_renewal(
                    request(key), lifecycle, recovery, renewal_choices,
                    profile.renewal, source,
                )
                self.assertIs(lifecycle.selected_lifecycle_depth, plan.lifecycle_depth)
                self.assertIs(recovery.selected_recovery_choice, plan.recovery_choice)
                reconstructed = reconstruct_profile_conditioned_renewal(
                    profile.renewal, renewal_choices, source,
                )
                self.assertIs(reconstructed.selected_renewal_choice,
                              plan.renewal_choice)
                self.assertEqual(source.reference, plan.planning_rng_reference)
                checked += 1
        self.assertEqual(192, checked)

    def test_runner_persistence_provenance_and_detector_boundary(self):
        requests = [request(f"runner-{index:03d}") for index in range(36)]
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
                planning_requests=requests, planning_root_seed=4102,
            )
            bundle = build_synchronized_bundle(sequences, run_population=population)
        assignment = population.behavioral_profiles.assignments[0]
        profile = assignment.profile
        self.assertIs(profile, population.behavioral_profiles.profile_for_device(
            assignment.device_id,
        ))
        self.assertEqual(36, population.semantic_contexts[
            assignment.device_id
        ].completed_trace_count)
        persistent_context = population.semantic_contexts[assignment.device_id]
        self.assertTrue(persistent_context.enrolled)
        self.assertFalse(persistent_context.blocked)
        self.assertFalse(persistent_context.revoked)
        previous_end = None
        seen_token_ids: set[str] = set()
        seen_attempt_ids: set[str] = set()
        for events, trace in sequences:
            if previous_end is not None:
                self.assertGreaterEqual(events[0].timestamp, previous_end)
            previous_end = events[-1].timestamp
            trace_tokens = {event.token_id for event in events if event.token_id}
            trace_attempts = set(trace.auth_attempt_ids)
            self.assertTrue(seen_token_ids.isdisjoint(trace_tokens))
            self.assertTrue(seen_attempt_ids.isdisjoint(trace_attempts))
            seen_token_ids.update(trace_tokens)
            seen_attempt_ids.update(trace_attempts)
        self.assertEqual(previous_end, persistent_context.semantic_time_cursor)
        records = [trace["trace_planning"] for trace in bundle["provenance"]["traces"]]
        self.assertTrue(any(
            record["selected_lifecycle_depth"] == "ACCESS" for record in records
        ))
        for record in records:
            self.assertEqual(3, record["renewal_decision_slot"])
            if record["selected_lifecycle_depth"] in ("AUTH", "SESSION"):
                self.assertEqual(["NO_RENEWAL"], record["applicable_renewal_choices"])
                self.assertEqual("NO_RENEWAL", record["selected_renewal_choice"])
                self.assertFalse(record["renewal_decision_stochastic"])
            else:
                self.assertEqual(["NO_RENEWAL", "RENEWAL"],
                                 record["applicable_renewal_choices"])
                self.assertTrue(record["renewal_decision_stochastic"])
        forbidden = (
            "behavioral_profile_id", "population_device_key", "renewal_tendency",
            "applicable_renewal_choices", "selected_renewal_choice",
            "renewal_decision_slot", "planning_rng_reference",
        )
        for surface in ("detector_observations", "ground_truth", "debug"):
            serialized = str(bundle[surface])
            for name in forbidden:
                self.assertNotIn(name, serialized)
        self.assertEqual({"validated_normal"}, set(
            bundle["validation"]["trace_ground_truth_statuses"].values(),
        ))
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
