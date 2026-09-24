"""DP.1 dormant Device Profile shape, persistence, RNG, and export checks."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict, fields, replace
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.device_profiles.population import (
    build_profile_population,
    population_device_key_from_index,
)
from simulator.device_profiles.profile import (
    DeviceProfile,
    LifecycleWeights,
    RelativeTimingTendency,
    TimingTendencies,
    LATENT_COMPONENTS,
    LIFECYCLE_DEPTHS,
    TIMING_RELATION_FAMILIES,
)
from simulator.runner import build_run_population, run_simulation
from simulator.data.synchronized_views import build_synchronized_bundle
from tests.hn4_stress_audit import _structure


PRE_DP1_RUNNER_SEMANTIC_SHA256 = (
    "657cb57cb28515120b233704b618e64c8ec1a37d61ef129a1fbabd3060d7808c"
)


def bindings(count: int) -> list[tuple[str, str]]:
    return [(f"emitted-uuid-{index}", population_device_key_from_index(index))
            for index in range(count)]


def keyed_profiles(population) -> dict[str, DeviceProfile]:
    return {item.population_device_key: item.profile for item in population.assignments}


class DeviceProfileSchemaTests(unittest.TestCase):
    def setUp(self):
        self.configuration = PopulationConfiguration(population_seed=101)
        self.profile = build_profile_population(bindings(1), self.configuration).assignments[0].profile

    def test_exact_five_latent_components_and_no_deferred_or_security_tendency(self):
        self.assertEqual(("T", "A", "R", "N", "L"), LATENT_COMPONENTS)
        self.assertEqual(
            {"behavioral_profile_id", "contract_version", "timing", "activity",
             "recovery", "renewal", "lifecycle"},
            {item.name for item in fields(DeviceProfile)},
        )
        self.assertEqual(
            ("AUTH", "TRANSITION", "ACCESS", "RENEWAL"), TIMING_RELATION_FAMILIES,
        )
        self.assertEqual(
            {"auth", "transition", "access", "renewal"},
            {item.name for item in fields(TimingTendencies)},
        )
        self.assertEqual(
            {"center_modifier", "dispersion_modifier"},
            {item.name for item in fields(RelativeTimingTendency)},
        )
        self.assertEqual(("AUTH", "SESSION", "ACCESS"), LIFECYCLE_DEPTHS)
        self.assertEqual(
            {"auth", "session", "access"},
            {item.name for item in fields(LifecycleWeights)},
        )
        flattened = json.dumps(asdict(self.profile)).lower()
        for forbidden in (
            "preference", "preauth", "identity_tendency", "binding_tendency",
            "authorization_tendency", "anomaly_tendency", "failure_rate",
            "retry_count", "message_rate", "packet_rate", "lifetime_s",
            "latency_ms", "request_spacing",
        ):
            self.assertNotIn(forbidden, flattened)

    def test_domains_normalization_and_immutability(self):
        for value in (self.profile.activity, self.profile.recovery, self.profile.renewal):
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        lifecycle = self.profile.lifecycle
        self.assertTrue(all(value >= 0 for value in (
            lifecycle.auth, lifecycle.session, lifecycle.access,
        )))
        self.assertAlmostEqual(1.0, lifecycle.auth + lifecycle.session + lifecycle.access)
        for name in ("activity", "recovery", "renewal"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(self.profile, **{name: -0.1})
        with self.assertRaises(ValueError):
            LifecycleWeights(0.2, 0.2, 0.2)
        with self.assertRaises(ValueError):
            LifecycleWeights(-0.1, 0.5, 0.6)
        with self.assertRaises(FrozenInstanceError):
            self.profile.activity = 0.5

    def test_population_configuration_is_separate_and_uncalibrated(self):
        self.assertEqual("synthetic_controlled_uncalibrated", self.configuration.calibration_status)
        self.assertEqual("device-profile-v1", self.configuration.contract_version)
        self.assertNotIn("profile_population", asdict(cfg))
        self.assertNotIn("device_profile_population", asdict(cfg))


class ProfilePopulationTests(unittest.TestCase):
    def test_runner_creates_one_sibling_profile_per_device(self):
        random.seed(42)
        with patch.object(cfg.simulation, "num_devices", 5):
            population = build_run_population()
        self.assertEqual(5, len(population.devices))
        self.assertEqual(5, len(population.devices_by_id))
        self.assertEqual(5, len(population.semantic_contexts))
        self.assertEqual(5, len(population.behavioral_profiles.assignments))
        self.assertEqual(5, len(population.behavioral_profiles.behavioral_profiles))
        for index, device in enumerate(population.devices):
            profile = population.behavioral_profiles.profile_for_device(device.device_id)
            self.assertIs(profile, population.behavioral_profiles.profile_for_device(device.device_id))
            self.assertEqual(
                population_device_key_from_index(index),
                population.behavioral_profiles.assignment_for_device(device.device_id).population_device_key,
            )
            self.assertEqual(device.device_type,
                             population.semantic_contexts[device.device_id].legacy_device_type_id)
            self.assertNotIn("behavioral_profile", asdict(population.semantic_contexts[device.device_id]))
            self.assertNotIn("behavioral_profile_id", asdict(population.semantic_contexts[device.device_id]))
            self.assertNotIn("behavioral_profile", vars(device))

    def test_profile_identity_reproducibility_and_append_stability(self):
        config = PopulationConfiguration(population_seed=77)
        first = build_profile_population(bindings(8), config)
        same_keys_new_uuids = [(f"different-uuid-{index}", key) for index, (_, key) in enumerate(bindings(8))]
        second = build_profile_population(same_keys_new_uuids, config)
        self.assertEqual(keyed_profiles(first), keyed_profiles(second))
        reversed_bindings = list(reversed(bindings(8)))
        self.assertEqual(
            keyed_profiles(first),
            keyed_profiles(build_profile_population(reversed_bindings, config)),
        )
        extended = build_profile_population(
            [*bindings(8), ("extra-uuid", "additional-key")], config
        )
        self.assertEqual(
            keyed_profiles(first),
            {key: keyed_profiles(extended)[key] for key in keyed_profiles(first)},
        )
        changed_seed = build_profile_population(bindings(8), replace(config, population_seed=78))
        self.assertTrue(any(
            keyed_profiles(first)[key] != keyed_profiles(changed_seed)[key]
            for key in keyed_profiles(first)
        ))
        for assignment in first.assignments:
            self.assertNotEqual(assignment.device_id, assignment.profile.behavioral_profile_id)
            self.assertNotEqual(assignment.population_device_key, assignment.profile.behavioral_profile_id)
            self.assertTrue(assignment.profile_seed_reference.startswith("sha256:"))
        internal = first.internal_reconstruction_record()
        self.assertEqual(8, len(internal["assignments"]))
        self.assertEqual(config.population_seed, internal["configuration"]["population_seed"])
        self.assertIn("seed_derivation_version", internal)

    def test_independent_marginal_strata_without_named_classes(self):
        config = PopulationConfiguration(population_seed=99, strata_count=4)
        profiles = [item.profile for item in build_profile_population(bindings(64), config).assignments]
        for name in ("activity", "recovery", "renewal"):
            occupied = {min(int(getattr(profile, name) * 4), 3) for profile in profiles}
            self.assertEqual({0, 1, 2, 3}, occupied)
        combinations = {(int(profile.activity * 4), int(profile.recovery * 4),
                         int(profile.renewal * 4)) for profile in profiles}
        self.assertGreater(len(combinations), 4)
        self.assertNotIn("device_types", asdict(config))
        self.assertNotIn("correlation", json.dumps(asdict(config)).lower())

    def test_global_random_state_is_unchanged_by_profile_population(self):
        random.seed(123)
        before = random.getstate()
        build_profile_population(bindings(16), PopulationConfiguration(population_seed=456))
        self.assertEqual(before, random.getstate())


class TraceNeutralityTests(unittest.TestCase):
    def test_runner_matches_pre_dp1_normalized_semantics_and_exports_no_profile(self):
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345, "num_devices": 4,
                "num_sessions_normal": 12, "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            sequences = run_simulation()
            bundle = build_synchronized_bundle(sequences)
        devices = {}
        rows = []
        for events, context in sequences:
            devices.setdefault(context.device_id, len(devices) + 1)
            rows.append({"device_ordinal": devices[context.device_id],
                         **_structure(events, context)})
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(PRE_DP1_RUNNER_SEMANTIC_SHA256, hashlib.sha256(payload).hexdigest())
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        for observation in bundle["detector_observations"]:
            names = set(observation)
            self.assertFalse(names & {
                "behavioral_profile_id", "population_device_key", "timing",
                "activity", "recovery", "renewal", "lifecycle", "center_modifier",
                "dispersion_modifier", "auth_weight", "session_weight", "access_weight",
            })
        for view in ("ground_truth", "provenance", "debug", "field_manifest"):
            serialized = json.dumps(bundle[view]).lower()
            self.assertNotIn("behavioral_profile_id", serialized)
            self.assertNotIn("population_device_key", serialized)


if __name__ == "__main__":
    unittest.main()
