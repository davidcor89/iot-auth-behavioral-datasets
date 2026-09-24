"""4C.INF lifecycle planning, neutrality, provenance, and boundary tests."""

from __future__ import annotations

import json
import random
import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import FieldRole, field_manifest
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.device_profiles.population import build_profile_population
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import (
    AUTH_PREFIX,
    CLEAN_ACCESS_CYCLE,
    NORMAL_FLOW,
    NORMAL_FLOW_WITH_RETRY,
    ScenarioEngine,
    ScenarioSpec,
)
from simulator.event_model import EventType
from simulator.planning.models import (
    ApplicableChoices,
    CompletionMode,
    ConstructionRole,
    LifecycleDepth,
    PlanningMode,
    TraceConstructionRequest,
    TracePlanningError,
    TracePlanningErrorReason,
)
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    adapt_trace_plan,
)
from simulator.planning.randomness import derive_planning_randomness
from simulator.planning.repertoire import (
    LIFECYCLE_REPERTOIRE_VERSION,
    ORDINARY_LIFECYCLE_REPERTOIRE,
)
from simulator.provenance import build_generation_provenance
from simulator.runner import run_simulation_with_population
from simulator.security_context import PersistentDeviceContext
from tests.hn4_stress_audit import _structure


def profile(seed: int = 1):
    return build_profile_population(
        [("emitted-device", "stable-device-key")],
        PopulationConfiguration(population_seed=seed),
    ).assignments[0].profile


def request(depth: LifecycleDepth, key: str = "trace-0001") -> TraceConstructionRequest:
    return TraceConstructionRequest.explicit_lifecycle(
        trace_construction_key=key,
        scenario_id=f"scenario-{key}",
        lifecycle_depth=depth,
    )


def plan_for(
    depth: LifecycleDepth,
    context: PersistentDeviceContext,
    *,
    key: str = "trace-0001",
):
    resolver = ApplicabilityResolver()
    randomness = derive_planning_randomness(
        planning_root_seed=91,
        population_device_key="stable-device-key",
        trace_construction_key=key,
    )
    return DeviceAwareTracePlanner().plan(
        request(depth, key), resolver.resolve(context, ConstructionRole.ORDINARY),
        profile(), randomness,
    )


def execute_plan(plan, context, *, seed=300, start_time=10_000.0):
    random.seed(seed)
    return EventEngine(
        device_id=context.device_id,
        gateway_id="gateway-1",
        auth_server_id="auth-1",
        broker_id="broker-1",
        source_ip="10.0.1.8",
        battery_level=80.0,
        firmware_version="1.0.0",
        persistent_context=context,
        start_time=start_time,
    ).execute(adapt_trace_plan(plan))


class LifecycleRepertoireTests(unittest.TestCase):
    def test_exact_lifecycle_ids_templates_and_intentional_endpoints(self):
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, {item.value for item in LifecycleDepth})
        self.assertNotIn("PREAUTH", {item.value for item in LifecycleDepth})
        self.assertEqual(
            tuple(NORMAL_FLOW[:11]),
            ORDINARY_LIFECYCLE_REPERTOIRE[LifecycleDepth.AUTH].event_steps,
        )
        self.assertEqual(
            tuple(AUTH_PREFIX),
            ORDINARY_LIFECYCLE_REPERTOIRE[LifecycleDepth.SESSION].event_steps,
        )
        self.assertEqual(
            tuple([*AUTH_PREFIX, *CLEAN_ACCESS_CYCLE]),
            ORDINARY_LIFECYCLE_REPERTOIRE[LifecycleDepth.ACCESS].event_steps,
        )
        expected = {
            LifecycleDepth.AUTH: EventType.AUTHENTICATION_SUCCESS,
            LifecycleDepth.SESSION: EventType.SESSION_OPENED,
            LifecycleDepth.ACCESS: EventType.ACCESS_GRANTED,
        }
        for depth, endpoint in expected.items():
            with self.subTest(depth=depth):
                template = ORDINARY_LIFECYCLE_REPERTOIRE[depth]
                self.assertIs(endpoint, template.intentional_endpoint)
                self.assertIs(endpoint, template.event_steps[-1])
                self.assertNotIn(EventType.AUTHENTICATION_FAILURE, template.event_steps)
                self.assertNotIn(EventType.RENEWAL_REQUEST, template.event_steps)

    def test_all_endpoints_validate_as_normal_and_access_has_one_clean_operation(self):
        for index, depth in enumerate(LifecycleDepth):
            with self.subTest(depth=depth):
                context = PersistentDeviceContext(device_id=f"device-{index}")
                plan = plan_for(depth, context, key=f"endpoint-{index}")
                self.assertIs(
                    CompletionMode.INTENTIONAL_LIFECYCLE_ENDPOINT,
                    plan.completion_mode,
                )
                events, trace = execute_plan(plan, context, seed=510 + index)
                bundle = build_synchronized_bundle([(events, trace)])
                self.assertEqual(
                    "validated_normal",
                    bundle["validation"]["trace_ground_truth_statuses"][trace.trace_id],
                )
                self.assertTrue(bundle["validation"]["build_gate"]["passed"])
                self.assertEqual(0, bundle["validation"]["result_counts"]["FAIL"])
                self.assertEqual(0, bundle["validation"]["result_counts"]["ERROR"])
                self.assertGreater(
                    len(bundle["provenance"]["dataset"]["capabilities"]
                        ["explicitly_unsupported_or_out_of_scope_capabilities"]),
                    0,
                )
                self.assertIs(ORDINARY_LIFECYCLE_REPERTOIRE[depth].intentional_endpoint,
                              events[-1].event_type)
                if depth is LifecycleDepth.ACCESS:
                    self.assertEqual(1, sum(e.event_type is EventType.ACCESS_REQUEST for e in events))
                    self.assertEqual(1, sum(e.event_type is EventType.ACCESS_GRANTED for e in events))

    def test_intentional_endpoints_are_separate_from_legacy_partial_generation(self):
        with patch(
            "simulator.engines.scenario_engine.build_partial_flow",
            side_effect=AssertionError("lifecycle registry must not call normal_partial"),
        ):
            for depth in LifecycleDepth:
                plan_for(depth, PersistentDeviceContext(device_id=f"device-{depth.value}"))
        random.seed(44)
        partial = ScenarioEngine().normal_partial()
        self.assertGreaterEqual(len(partial.normal_steps), 3)
        self.assertLessEqual(len(partial.normal_steps), 12)
        self.assertIs(CompletionMode.VALID_PREFIX, CompletionMode.VALID_PREFIX)


class ApplicabilityAndFailureTests(unittest.TestCase):
    def setUp(self):
        self.resolver = ApplicabilityResolver()
        self.planner = DeviceAwareTracePlanner()
        self.randomness = derive_planning_randomness(
            planning_root_seed=8,
            population_device_key="stable-device-key",
            trace_construction_key="request-one",
        )

    def test_profile_values_do_not_change_applicability_or_selection(self):
        context = PersistentDeviceContext(device_id="device")
        choices = self.resolver.resolve(context, ConstructionRole.ORDINARY)
        self.assertEqual(tuple(LifecycleDepth), choices.lifecycle_depths)
        first = self.planner.plan(request(LifecycleDepth.SESSION), choices, profile(1), self.randomness)
        second = self.planner.plan(request(LifecycleDepth.SESSION), choices, profile(2), self.randomness)
        self.assertEqual(first, second)
        # An opaque object proves no profile ID or latent field is inspected.
        third = self.planner.plan(request(LifecycleDepth.SESSION), choices, object(), self.randomness)
        self.assertEqual(first, third)

    def test_blocked_and_revoked_contexts_have_no_fallback(self):
        for field in ("blocked", "revoked"):
            context = PersistentDeviceContext(device_id="device", **{field: True})
            choices = self.resolver.resolve(context, ConstructionRole.ORDINARY)
            self.assertEqual((), choices.lifecycle_depths)
            with self.assertRaises(TracePlanningError) as raised:
                self.planner.plan(request(LifecycleDepth.AUTH), choices, profile(), self.randomness)
            self.assertIs(
                TracePlanningErrorReason.NO_SUPPORTED_APPLICABLE_LIFECYCLE,
                raised.exception.reason,
            )

    def test_unsupported_lifecycle_and_invalid_role_fail_explicitly(self):
        restricted = ApplicableChoices(
            lifecycle_depths=(LifecycleDepth.AUTH,),
            supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
        )
        with self.assertRaises(TracePlanningError) as raised:
            self.planner.plan(request(LifecycleDepth.ACCESS), restricted, profile(), self.randomness)
        self.assertIs(TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                      raised.exception.reason)

        invalid = replace(request(LifecycleDepth.AUTH), construction_role=ConstructionRole.ANOMALY)
        with self.assertRaises(TracePlanningError) as raised:
            self.resolver.resolve(PersistentDeviceContext(device_id="device"), invalid.construction_role)
        self.assertIs(TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                      raised.exception.reason)

    def test_applicability_is_rng_free_and_independent_of_enrollment_prefix(self):
        random.seed(991)
        state = random.getstate()
        fresh = self.resolver.resolve(
            PersistentDeviceContext(device_id="fresh"), ConstructionRole.ORDINARY,
        )
        enrolled = self.resolver.resolve(
            PersistentDeviceContext(device_id="enrolled", enrolled=True),
            ConstructionRole.ORDINARY,
        )
        self.assertEqual(fresh, enrolled)
        self.assertEqual(state, random.getstate())


class PlanningRandomnessTests(unittest.TestCase):
    def test_stable_inputs_isolation_and_key_separation(self):
        random.seed(771)
        state = random.getstate()
        first = derive_planning_randomness(
            planning_root_seed=12,
            population_device_key="population-device-00000001",
            trace_construction_key="trace-4",
        )
        same = derive_planning_randomness(
            planning_root_seed=12,
            population_device_key="population-device-00000001",
            trace_construction_key="trace-4",
        )
        changed_trace = derive_planning_randomness(
            planning_root_seed=12,
            population_device_key="population-device-00000001",
            trace_construction_key="trace-5",
        )
        changed_device = derive_planning_randomness(
            planning_root_seed=12,
            population_device_key="population-device-00000002",
            trace_construction_key="trace-4",
        )
        self.assertEqual(first, same)
        self.assertNotEqual(first.reference, changed_trace.reference)
        self.assertNotEqual(first.reference, changed_device.reference)
        isolated = first.new_rng()
        self.assertEqual(state, random.getstate())
        isolated.random()
        self.assertEqual(state, random.getstate())
        self.assertTrue(first.reference.startswith("sha256:"))

    def test_profile_lookup_and_applicability_do_not_consume_planning_stream(self):
        population = build_profile_population(
            [("uuid-a", "stable-a")], PopulationConfiguration(population_seed=3),
        )
        randomness = derive_planning_randomness(
            planning_root_seed=12,
            population_device_key="stable-a",
            trace_construction_key="trace-a",
        )
        planning_rng = randomness.new_rng()
        state = planning_rng.getstate()
        assignment = population.assignment_for_device("uuid-a")
        choices = ApplicabilityResolver().resolve(
            PersistentDeviceContext(device_id="uuid-a"), ConstructionRole.ORDINARY,
        )
        DeviceAwareTracePlanner().plan(
            request(LifecycleDepth.AUTH), choices, assignment.profile, randomness,
        )
        self.assertEqual(state, planning_rng.getstate())


class PersistenceAndNeutralityTests(unittest.TestCase):
    def test_auth_and_session_continuations_reuse_enrollment_and_validate(self):
        for first_depth in (LifecycleDepth.AUTH, LifecycleDepth.SESSION):
            with self.subTest(first_depth=first_depth):
                context = PersistentDeviceContext(device_id=f"device-{first_depth.value}")
                first = execute_plan(plan_for(first_depth, context, key="first"), context, seed=811)
                first_end = context.semantic_time_cursor
                second = execute_plan(
                    plan_for(LifecycleDepth.AUTH, context, key="second"),
                    context,
                    seed=812,
                    start_time=10_000.0,
                )
                self.assertTrue(context.enrolled)
                self.assertGreaterEqual(second[0][0].timestamp, first_end)
                self.assertIs(EventType.AUTHENTICATION_REQUEST, second[0][0].event_type)
                self.assertFalse(any(event.event_type is EventType.DISCOVERY for event in second[0]))
                bundle = build_synchronized_bundle([first, second])
                self.assertTrue(bundle["validation"]["build_gate"]["passed"])
                self.assertEqual(
                    {"validated_normal"},
                    set(bundle["validation"]["trace_ground_truth_statuses"].values()),
                )

    def test_legacy_equivalent_adapter_preserves_normalized_semantics_and_rng_state(self):
        spec = ScenarioSpec(
            scenario_id="legacy-scenario",
            scenario_type="normal",
            is_anomaly=False,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            normal_steps=list(NORMAL_FLOW_WITH_RETRY),
        )
        legacy_request = TraceConstructionRequest.legacy_equivalent(
            trace_construction_key="legacy-trace",
            lifecycle_depth=LifecycleDepth.ACCESS,
            scenario_spec=spec,
        )
        context_for_plan = PersistentDeviceContext(device_id="device-neutral")
        choices = ApplicabilityResolver().resolve(
            context_for_plan, ConstructionRole.ORDINARY,
        )
        randomness = derive_planning_randomness(
            planning_root_seed=4,
            population_device_key="stable-neutral-device",
            trace_construction_key="legacy-trace",
        )
        random.seed(901)
        state_before = random.getstate()
        plan = DeviceAwareTracePlanner().plan(
            legacy_request, choices, profile(), randomness,
        )
        planned_spec = adapt_trace_plan(plan)
        self.assertEqual(state_before, random.getstate())
        self.assertEqual(spec.normal_steps, planned_spec.normal_steps)
        self.assertEqual(spec.generation_profile, planned_spec.generation_profile)

        old_context = PersistentDeviceContext(device_id="device-neutral")
        new_context = PersistentDeviceContext(device_id="device-neutral")
        random.seed(902)
        old = EventEngine(
            device_id="device-neutral", gateway_id="gateway", auth_server_id="auth",
            broker_id="broker", source_ip="10.0.1.9", battery_level=60.0,
            firmware_version="1.0.0", persistent_context=old_context, start_time=1_000.0,
        ).execute(spec)
        random.seed(902)
        new = EventEngine(
            device_id="device-neutral", gateway_id="gateway", auth_server_id="auth",
            broker_id="broker", source_ip="10.0.1.9", battery_level=60.0,
            firmware_version="1.0.0", persistent_context=new_context, start_time=1_000.0,
        ).execute(planned_spec)
        self.assertEqual(_structure(*old), _structure(*new))


class RunnerAndProvenanceTests(unittest.TestCase):
    def test_opt_in_runner_is_device_before_plan_and_provenance_is_privileged(self):
        requests = [
            request(LifecycleDepth.AUTH, "runner-auth"),
            request(LifecycleDepth.SESSION, "runner-session"),
            request(LifecycleDepth.ACCESS, "runner-access"),
        ]
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345,
                "num_devices": 1,
                "num_sessions_normal": 99,
                "num_sessions_attack": 99,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            sequences, population = run_simulation_with_population(
                planning_requests=requests,
                planning_root_seed=700,
            )
            provenance = build_generation_provenance(
                cfg, run_id="4c-inf-run", generator_git_dirty=True,
            )
            bundle = build_synchronized_bundle(
                sequences, provenance=provenance, run_population=population,
            )

        self.assertEqual(3, len(sequences))
        self.assertEqual(3, len(population.trace_planning_records))
        self.assertEqual(
            [EventType.AUTHENTICATION_SUCCESS, EventType.SESSION_OPENED, EventType.ACCESS_GRANTED],
            [events[-1].event_type for events, _ in sequences],
        )
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        planning_records = [trace["trace_planning"] for trace in bundle["provenance"]["traces"]]
        self.assertTrue(all(record is not None for record in planning_records))
        self.assertEqual(
            {"AUTH", "SESSION", "ACCESS"},
            {record["selected_lifecycle_depth"] for record in planning_records},
        )
        self.assertTrue(all(record["profile_conditioned_selection"] is False
                            for record in planning_records))
        forbidden_latents = {
            "timing", "activity", "recovery", "renewal", "lifecycle",
            "T", "A", "R", "N", "L", "center_modifier", "dispersion_modifier",
        }
        for record in planning_records:
            self.assertFalse(forbidden_latents & set(record))
            self.assertNotIn("behavioral_profile_id", record)
            self.assertEqual("provenance_trace.device_id",
                             record["profile_association_reference"])
        for surface in ("detector_observations", "ground_truth", "debug"):
            serialized = json.dumps(bundle[surface])
            self.assertNotIn("trace_planning", serialized)
            self.assertNotIn("behavioral_profile_id", serialized)
            self.assertNotIn("planning_rng_reference", serialized)
        planning_definition = next(
            item for item in field_manifest()["fields"]
            if item["export_view"] == "provenance_traces"
            and item["field_name"] == "trace_planning"
        )
        self.assertEqual(FieldRole.PROVENANCE.value, planning_definition["role"])
        self.assertTrue(planning_definition["privileged"])


if __name__ == "__main__":
    unittest.main()
