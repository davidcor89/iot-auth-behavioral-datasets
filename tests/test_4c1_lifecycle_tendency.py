"""4C.1 masked lifecycle-tendency consumption and boundary regressions."""

from __future__ import annotations

import json
import math
import random
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import FIELD_MANIFEST_VERSION, field_manifest
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.device_profiles.population import (
    build_profile_population,
    population_device_key_from_index,
)
from simulator.device_profiles.profile import LifecycleWeights
from simulator.engines.event_engine import EventEngine
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
    PROFILE_CONDITIONED_PLANNER_VERSION,
    PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
    adapt_trace_plan,
    normalized_applicable_lifecycle_weights,
    reconstruct_profile_conditioned_lifecycle,
    select_profile_conditioned_lifecycle,
)
from simulator.planning.randomness import PlanningRandomness, derive_planning_randomness
from simulator.planning.repertoire import LIFECYCLE_REPERTOIRE_VERSION
from simulator.provenance import FEATURE_SCHEMA_VERSION, build_generation_provenance
from simulator.runner import run_simulation_with_population
from simulator.security_context import PersistentDeviceContext


def choices(*depths: LifecycleDepth) -> ApplicableChoices:
    return ApplicableChoices(
        lifecycle_depths=depths or tuple(LifecycleDepth),
        supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
    )


def profile_request(key: str) -> TraceConstructionRequest:
    return TraceConstructionRequest.profile_conditioned_lifecycle(
        trace_construction_key=key,
        scenario_id=f"scenario-{key}",
    )


def planning_randomness(
    key: str,
    *,
    device_key: str = "device-a",
    root_seed: int = 42,
):
    return derive_planning_randomness(
        planning_root_seed=root_seed,
        population_device_key=device_key,
        trace_construction_key=key,
    )


class FixedCountingRandom:
    def __init__(self, value: float):
        self.value = value
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return self.value


class LifecycleSelectionMechanismTests(unittest.TestCase):
    def test_one_hot_profiles_select_the_only_positive_depth(self):
        fixtures = (
            (LifecycleWeights(1.0, 0.0, 0.0), LifecycleDepth.AUTH),
            (LifecycleWeights(0.0, 1.0, 0.0), LifecycleDepth.SESSION),
            (LifecycleWeights(0.0, 0.0, 1.0), LifecycleDepth.ACCESS),
        )
        for weights, expected in fixtures:
            with self.subTest(expected=expected):
                rng = FixedCountingRandom(0.731)
                selection = select_profile_conditioned_lifecycle(weights, choices(), rng)
                self.assertIs(expected, selection.selected_lifecycle_depth)
                self.assertEqual(1, rng.draw_count)

    def test_masking_and_renormalization_ignore_inapplicable_positive_mass(self):
        weights = LifecycleWeights(0.20, 0.50, 0.30)
        applicable = choices(LifecycleDepth.AUTH, LifecycleDepth.ACCESS)
        distribution = dict(normalized_applicable_lifecycle_weights(weights, applicable))
        self.assertEqual({LifecycleDepth.AUTH, LifecycleDepth.ACCESS}, set(distribution))
        self.assertNotIn(LifecycleDepth.SESSION, distribution)
        self.assertAlmostEqual(0.40, distribution[LifecycleDepth.AUTH])
        self.assertAlmostEqual(0.60, distribution[LifecycleDepth.ACCESS])
        self.assertIs(
            LifecycleDepth.AUTH,
            select_profile_conditioned_lifecycle(
                weights, applicable, FixedCountingRandom(0.399999),
            ).selected_lifecycle_depth,
        )
        self.assertIs(
            LifecycleDepth.ACCESS,
            select_profile_conditioned_lifecycle(
                weights, applicable, FixedCountingRandom(0.400001),
            ).selected_lifecycle_depth,
        )

    def test_zero_applicable_weight_fails_without_fallback(self):
        weights = LifecycleWeights(0.0, 1.0, 0.0)
        with self.assertRaises(TracePlanningError) as raised:
            normalized_applicable_lifecycle_weights(
                weights, choices(LifecycleDepth.AUTH, LifecycleDepth.ACCESS),
            )
        self.assertIs(TracePlanningErrorReason.ZERO_APPLICABLE_WEIGHT,
                      raised.exception.reason)

    def test_malformed_lifecycle_weights_are_rejected(self):
        malformed = (
            SimpleNamespace(auth=0.5, session=0.5),
            SimpleNamespace(auth=-0.1, session=0.6, access=0.5),
            SimpleNamespace(auth=math.nan, session=0.5, access=0.5),
            SimpleNamespace(auth=0.5, session=0.5, access=0.5),
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(TracePlanningError) as raised:
                normalized_applicable_lifecycle_weights(value, choices())
            self.assertIs(TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
                          raised.exception.reason)

    def test_profile_conditioned_request_is_ordinary_intentional_and_selects_depth(self):
        request = profile_request("request-1")
        self.assertIs(ConstructionRole.ORDINARY, request.construction_role)
        self.assertIs(CompletionMode.INTENTIONAL_LIFECYCLE_ENDPOINT,
                      request.completion_mode)
        self.assertIs(PlanningMode.PROFILE_CONDITIONED_LIFECYCLE,
                      request.planning_mode)
        self.assertIsNone(request.requested_lifecycle_depth)
        plan = DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle(
            request,
            choices(),
            LifecycleWeights(1.0, 0.0, 0.0),
            planning_randomness("request-1"),
        )
        self.assertIs(LifecycleDepth.AUTH, plan.lifecycle_depth)
        self.assertEqual(1, plan.planning_draw_count)
        self.assertTrue(plan.profile_conditioned_selection)
        self.assertEqual(PROFILE_CONDITIONED_PLANNER_VERSION, plan.planner_version)

    def test_profile_conditioned_auth_session_access_traces_validate(self):
        fixtures = (
            (LifecycleWeights(1.0, 0.0, 0.0), LifecycleDepth.AUTH),
            (LifecycleWeights(0.0, 1.0, 0.0), LifecycleDepth.SESSION),
            (LifecycleWeights(0.0, 0.0, 1.0), LifecycleDepth.ACCESS),
        )
        for index, (weights, expected) in enumerate(fixtures):
            with self.subTest(expected=expected):
                context = PersistentDeviceContext(device_id=f"lifecycle-device-{index}")
                plan = DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle(
                    profile_request(f"one-hot-{index}"), choices(), weights,
                    planning_randomness(f"one-hot-{index}"),
                )
                random.seed(850 + index)
                pair = EventEngine(
                    device_id=context.device_id,
                    gateway_id="gateway",
                    auth_server_id="auth",
                    broker_id="broker",
                    source_ip="10.0.1.10",
                    battery_level=70.0,
                    firmware_version="1.0.0",
                    persistent_context=context,
                    start_time=10_000.0,
                ).execute(adapt_trace_plan(plan))
                bundle = build_synchronized_bundle([pair])
                self.assertIs(expected, plan.lifecycle_depth)
                self.assertEqual(
                    "validated_normal",
                    bundle["validation"]["trace_ground_truth_statuses"][pair[1].trace_id],
                )
                self.assertTrue(bundle["validation"]["build_gate"]["passed"])


class LifecyclePlanningRngTests(unittest.TestCase):
    def test_reproducibility_key_separation_uuid_independence_and_global_isolation(self):
        weights = LifecycleWeights(0.2, 0.3, 0.5)
        applicable = choices()
        random.seed(778)
        global_state = random.getstate()
        first_randomness = planning_randomness("trace-08")
        first = reconstruct_profile_conditioned_lifecycle(
            weights, applicable, first_randomness,
        )
        same = reconstruct_profile_conditioned_lifecycle(
            weights, applicable, planning_randomness("trace-08"),
        )
        different_trace = reconstruct_profile_conditioned_lifecycle(
            weights, applicable, planning_randomness("trace-12"),
        )
        different_device = reconstruct_profile_conditioned_lifecycle(
            weights, applicable, planning_randomness("trace-08", device_key="device-b"),
        )
        self.assertEqual(first, same)
        self.assertIs(LifecycleDepth.AUTH, first.selected_lifecycle_depth)
        self.assertIs(LifecycleDepth.SESSION, different_trace.selected_lifecycle_depth)
        self.assertNotEqual(first_randomness.reference,
                            planning_randomness("trace-08", device_key="device-b").reference)
        self.assertNotEqual(first.random_variate, different_device.random_variate)
        # No emitted UUID is an input; changing it cannot alter this stream.
        self.assertEqual(first, reconstruct_profile_conditioned_lifecycle(
            weights, applicable, first_randomness,
        ))
        self.assertEqual(
            first,
            reconstruct_profile_conditioned_lifecycle(
                weights, applicable,
                PlanningRandomness.from_reference(first_randomness.reference),
            ),
        )
        self.assertEqual(global_state, random.getstate())

    def test_selection_consumes_exactly_one_draw_after_rng_free_masking(self):
        rng = FixedCountingRandom(0.75)
        weights = LifecycleWeights(0.2, 0.3, 0.5)
        applicable = choices()
        normalized_applicable_lifecycle_weights(weights, applicable)
        self.assertEqual(0, rng.draw_count)
        select_profile_conditioned_lifecycle(weights, applicable, rng)
        self.assertEqual(1, rng.draw_count)

    def test_planner_consumes_exactly_one_draw_from_derived_stream(self):
        rng = FixedCountingRandom(0.75)
        randomness = planning_randomness("planner-draw")
        with patch.object(PlanningRandomness, "new_rng", return_value=rng):
            plan = DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle(
                profile_request("planner-draw"), choices(),
                LifecycleWeights(0.2, 0.3, 0.5), randomness,
            )
        self.assertEqual(1, rng.draw_count)
        self.assertEqual(1, plan.planning_draw_count)


class ApplicabilityAndOverlapTests(unittest.TestCase):
    def test_profiles_cannot_change_applicable_membership(self):
        resolver = ApplicabilityResolver()
        context = PersistentDeviceContext(device_id="device")
        before = resolver.resolve(context, ConstructionRole.ORDINARY)
        profiles = (
            LifecycleWeights(1.0, 0.0, 0.0),
            LifecycleWeights(0.0, 0.0, 1.0),
        )
        after = [resolver.resolve(context, ConstructionRole.ORDINARY) for _ in profiles]
        self.assertTrue(all(item == before for item in after))

    def test_distinct_overlapping_profiles_can_realize_common_depth(self):
        first = LifecycleWeights(0.7, 0.2, 0.1)
        second = LifecycleWeights(0.2, 0.7, 0.1)
        randomness = planning_randomness("trace-08")  # deterministic u ~= 0.043
        first_choice = reconstruct_profile_conditioned_lifecycle(first, choices(), randomness)
        second_choice = reconstruct_profile_conditioned_lifecycle(second, choices(), randomness)
        self.assertNotEqual(first, second)
        self.assertIs(LifecycleDepth.AUTH, first_choice.selected_lifecycle_depth)
        self.assertIs(first_choice.selected_lifecycle_depth,
                      second_choice.selected_lifecycle_depth)


class RunnerPersistenceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.requests = [profile_request(f"runner-{index}") for index in range(12)]

    def _run(self):
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
                planning_requests=self.requests,
                planning_root_seed=700,
            )
            provenance = build_generation_provenance(
                cfg, run_id="4c1-lifecycle-run", generator_git_dirty=True,
            )
            self.assertEqual(
                "4c-inf-planning-sha256-v1",
                provenance.randomness_contract["trace_planning_stream"]
                    ["derivation_version"],
            )
            bundle = build_synchronized_bundle(
                sequences, provenance=provenance, run_population=population,
            )
        return sequences, population, bundle

    def test_persistent_profile_stochastic_traces_validate_and_reconstruct(self):
        sequences, population, bundle = self._run()
        assignment = population.behavioral_profiles.assignments[0]
        original_profile = assignment.profile
        planning_records = [
            trace["trace_planning"] for trace in bundle["provenance"]["traces"]
        ]
        selected = [record["selected_lifecycle_depth"] for record in planning_records]
        self.assertEqual(12, len(set(record["trace_construction_key"]
                                     for record in planning_records)))
        self.assertGreater(len(set(selected)), 1)
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, set(selected))
        self.assertIs(original_profile,
                      population.behavioral_profiles.profile_for_device(assignment.device_id))
        self.assertEqual(original_profile.lifecycle,
                         population.behavioral_profiles.profile_for_device(
                             assignment.device_id).lifecycle)
        self.assertEqual(
            {"validated_normal"},
            set(bundle["validation"]["trace_ground_truth_statuses"].values()),
        )
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        self.assertEqual(12, len(sequences))

        applicable = choices()
        for record in planning_records:
            reconstructed = reconstruct_profile_conditioned_lifecycle(
                original_profile.lifecycle,
                applicable,
                PlanningRandomness.from_reference(record["planning_rng_reference"]),
            )
            self.assertEqual(record["selected_lifecycle_depth"],
                             reconstructed.selected_lifecycle_depth.value)

    def test_trace_provenance_versions_and_information_boundaries(self):
        _, _, bundle = self._run()
        records = [trace["trace_planning"] for trace in bundle["provenance"]["traces"]]
        for record in records:
            self.assertEqual("4c5-trace-planning-provenance-v6",
                             record["planning_provenance_schema_version"])
            self.assertEqual(PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
                             record["planner_version"])
            self.assertEqual("profile_conditioned_lifecycle", record["planning_mode"])
            self.assertEqual(["AUTH", "SESSION", "ACCESS"],
                             record["applicable_lifecycle_depths"])
            self.assertIn(record["planning_draw_count"], (2, 3, 4))
            self.assertTrue(record["profile_conditioned_selection"])
            self.assertTrue(record["profile_conditioned_recovery_selection"])
            self.assertEqual(2, record["recovery_decision_slot"])
            self.assertTrue(record["recovery_decision_stochastic"])
            self.assertEqual("4c-inf-planning-sha256-v1",
                             record["planning_rng_derivation_version"])
            self.assertEqual("device-aware-trace-planning",
                             record["planning_rng_namespace"])
            for forbidden in ("lifecycle_weights", "timing", "activity", "recovery",
                              "renewal", "behavioral_profile_id", "population_device_key"):
                self.assertNotIn(forbidden, record)

        for surface in ("detector_observations", "ground_truth", "debug"):
            serialized = json.dumps(bundle[surface])
            for forbidden in (
                "applicable_lifecycle_depths", "selected_lifecycle_depth",
                "planning_mode", "planning_rng_reference",
                "profile_conditioned_selection", "behavioral_profile_id",
                "population_device_key",
            ):
                self.assertNotIn(forbidden, serialized)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        self.assertEqual("sync-v1.1.0+4c4.activity-field-manifest-v5",
                         FIELD_MANIFEST_VERSION)
        definition = next(
            item for item in field_manifest()["fields"]
            if item["export_view"] == "provenance_traces"
            and item["field_name"] == "trace_planning"
        )
        self.assertTrue(definition["privileged"])
        self.assertEqual("provenance", definition["role"])
        randomness_metadata = bundle["provenance"]["dataset"]["randomness_metadata"]
        self.assertTrue(randomness_metadata["derived_stream_metadata_available"])
        self.assertEqual(
            "legacy_global_event_stream_plus_isolated_trace_planning_"
            "and_relation_timing_derivation",
            randomness_metadata["stream_model"],
        )
        self.assertEqual(["4c-inf-planning-sha256-v1"],
                         randomness_metadata["planning_rng_derivation_versions"])
        serialized_profile_companion = json.dumps(bundle["device_profile_provenance"])
        self.assertNotIn("trace_planning", serialized_profile_companion)
        self.assertNotIn("selected_lifecycle_depth", serialized_profile_companion)


if __name__ == "__main__":
    unittest.main()
