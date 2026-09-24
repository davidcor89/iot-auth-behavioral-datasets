"""DP.2 fingerprint, round-trip, referential and export-authority checks."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import FieldRole, field_manifest
from simulator.data.synchronized_views import (
    SynchronizedOutputViews,
    build_synchronized_bundle,
)
from simulator.device_profiles.population import build_profile_population
from simulator.device_profiles.provenance import (
    POPULATION_FINGERPRINT_VERSION,
    PROFILE_PROVENANCE_AUTHORITY,
    PROFILE_PROVENANCE_SCHEMA_VERSION,
    PROFILE_SAMPLER_VERSION,
    assert_trace_profile_association,
    load_device_profile_provenance,
    population_fingerprint,
    to_device_profile_provenance,
    validate_device_profile_provenance,
)
from simulator.provenance import build_generation_provenance
from simulator.runner import RunPopulation, run_simulation_with_population
from tests.hn4_stress_audit import _structure
from tests.test_dp1_profile_infrastructure import PRE_DP1_RUNNER_SEMANTIC_SHA256


def test_population(
    configuration: PopulationConfiguration,
    bindings: list[tuple[str, str]],
) -> RunPopulation:
    devices = [SimpleNamespace(device_id=device_id) for device_id, _ in bindings]
    return RunPopulation(
        devices=devices,
        devices_by_id={device.device_id: device for device in devices},
        semantic_contexts={device.device_id: object() for device in devices},
        behavioral_profiles=build_profile_population(bindings, configuration),
    )


def bindings(count: int, *, prefix: str = "uuid") -> list[tuple[str, str]]:
    return [(f"{prefix}-{index}", f"stable-population-key-{index:03d}")
            for index in range(count)]


class PopulationFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.configuration = PopulationConfiguration(population_seed=19)
        self.first = to_device_profile_provenance(
            test_population(self.configuration, bindings(5)), run_id="run-one",
        )

    def fingerprint(self, artifact):
        return artifact["population"]["population_fingerprint"]

    def test_normalized_population_identity_ignores_uuid_and_record_order(self):
        same = to_device_profile_provenance(
            test_population(self.configuration, bindings(5, prefix="other-uuid")),
            run_id="run-two",
        )
        reordered = to_device_profile_provenance(
            test_population(self.configuration, list(reversed(bindings(5)))),
            run_id="run-three",
        )
        self.assertEqual(self.fingerprint(self.first), self.fingerprint(same))
        self.assertEqual(self.fingerprint(self.first), self.fingerprint(reordered))
        self.assertNotEqual(self.first["devices"][0]["device_id"], same["devices"][0]["device_id"])
        record_order = list(reversed(self.first["devices"]))
        self.assertEqual(
            self.fingerprint(self.first),
            population_fingerprint(self.first["population"]["configuration"], record_order),
        )

    def test_seed_configuration_and_latent_changes_change_fingerprint(self):
        changed_seed = to_device_profile_provenance(
            test_population(replace(self.configuration, population_seed=20), bindings(5)),
            run_id="run-four",
        )
        changed_configuration = to_device_profile_provenance(
            test_population(replace(self.configuration, strata_count=5), bindings(5)),
            run_id="run-five",
        )
        changed_profile = copy.deepcopy(self.first)
        changed_profile["devices"][0]["latent_tendencies"]["A"] += 0.01
        changed_value_fingerprint = population_fingerprint(
            changed_profile["population"]["configuration"], changed_profile["devices"],
        )
        self.assertNotEqual(self.fingerprint(self.first), self.fingerprint(changed_seed))
        self.assertNotEqual(self.fingerprint(self.first), self.fingerprint(changed_configuration))
        self.assertNotEqual(self.fingerprint(self.first), changed_value_fingerprint)
        self.assertTrue(self.fingerprint(self.first).startswith("sha256:"))
        self.assertEqual(POPULATION_FINGERPRINT_VERSION,
                         "dp2-normalized-population-sha256-v1")

    def test_identity_uniqueness_and_reconstruction_are_enforced(self):
        validate_device_profile_provenance(self.first)
        for field in ("device_id", "population_device_key", "behavioral_profile_id"):
            altered = copy.deepcopy(self.first)
            altered["devices"][1][field] = altered["devices"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_device_profile_provenance(altered)
        altered = copy.deepcopy(self.first)
        altered["devices"][0]["latent_tendencies"]["A"] += 0.01
        altered["population"]["population_fingerprint"] = population_fingerprint(
            altered["population"]["configuration"], altered["devices"],
        )
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            validate_device_profile_provenance(altered)

    def test_reload_normalizes_device_record_order(self):
        reordered = copy.deepcopy(self.first)
        reordered["devices"].reverse()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reordered_profile_provenance.json"
            path.write_text(json.dumps(reordered), encoding="utf-8")
            self.assertEqual(self.first, load_device_profile_provenance(path))


class ProfileExportTests(unittest.TestCase):
    def setUp(self):
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345, "num_devices": 1,
                "num_sessions_normal": 4, "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            self.sequences, self.population = run_simulation_with_population()
            self.generation = build_generation_provenance(
                cfg, run_id="dp2-round-trip", generator_git_dirty=False,
            )

    def test_two_level_schema_and_trace_association(self):
        artifact = to_device_profile_provenance(
            self.population, run_id=self.generation.run_id, sequences=self.sequences,
        )
        self.assertEqual(PROFILE_PROVENANCE_AUTHORITY, artifact["authority"])
        self.assertEqual(PROFILE_PROVENANCE_SCHEMA_VERSION,
                         artifact["provenance_schema_version"])
        self.assertEqual(self.generation.run_id, artifact["run_id"])
        self.assertEqual(
            PROFILE_SAMPLER_VERSION,
            artifact["population"]["configuration"]["sampler_version"],
        )
        self.assertEqual(1, artifact["population"]["device_count"])
        self.assertEqual(1, len(artifact["devices"]))
        self.assertEqual({"T", "A", "R", "N", "L"},
                         set(artifact["devices"][0]["latent_tendencies"]))
        self.assertEqual(4, len(self.sequences))
        self.assertEqual(1, len({context.device_id for _, context in self.sequences}))
        assert_trace_profile_association(self.sequences, artifact)
        for _, context in self.sequences:
            self.assertEqual(
                self.population.behavioral_profiles.profile_for_device(context.device_id)
                    .behavioral_profile_id,
                artifact["devices"][0]["behavioral_profile_id"],
            )

    def test_separate_json_round_trip_and_authority_separation(self):
        baseline = build_synchronized_bundle(self.sequences, provenance=self.generation)
        with_profile = build_synchronized_bundle(
            self.sequences, provenance=self.generation, run_population=self.population,
        )
        self.assertEqual(set(baseline) | {"device_profile_provenance"}, set(with_profile))
        for view in baseline:
            if view == "provenance":
                baseline_provenance = copy.deepcopy(baseline[view])
                profile_provenance = copy.deepcopy(with_profile[view])
                self.assertEqual(
                    "absent", baseline_provenance["dataset"]["capabilities"]
                        ["device_profile_provenance"]["status"],
                )
                self.assertEqual(
                    "present", profile_provenance["dataset"]["capabilities"]
                        ["device_profile_provenance"]["status"],
                )
                del baseline_provenance["dataset"]["capabilities"]["device_profile_provenance"]
                del profile_provenance["dataset"]["capabilities"]["device_profile_provenance"]
                self.assertEqual(baseline_provenance, profile_provenance)
            else:
                self.assertEqual(baseline[view], with_profile[view], view)
        detector_fields = set(baseline["detector_observations"][0])
        profile_fields = (
            set(with_profile["device_profile_provenance"]["devices"][0])
            | set(with_profile["device_profile_provenance"]["population"]["configuration"])
            | set(with_profile["device_profile_provenance"]["devices"][0]["latent_tendencies"])
        )
        self.assertFalse(detector_fields & profile_fields)
        manifest = field_manifest()
        detector_definitions = {
            item["field_name"] for item in manifest["fields"]
            if item["role"] in (FieldRole.OBSERVABLE.value, FieldRole.DERIVED_OBSERVABLE.value)
        }
        self.assertFalse(detector_definitions & profile_fields)

        with tempfile.TemporaryDirectory() as directory:
            paths = SynchronizedOutputViews(directory).save(
                self.sequences, stem="dp2-test", provenance=self.generation,
                run_population=self.population,
            )
            self.assertEqual("dp2-test_device_profile_provenance.json",
                             paths["device_profile_provenance"].name)
            loaded = load_device_profile_provenance(paths["device_profile_provenance"])
            self.assertEqual(with_profile["device_profile_provenance"], loaded)
            self.assertEqual(
                with_profile["device_profile_provenance"]["population"]["population_fingerprint"],
                loaded["population"]["population_fingerprint"],
            )
            assert_trace_profile_association(self.sequences, loaded)
            self.assertNotIn("behavioral_profile_id", Path(paths["provenance"]).read_text())
            self.assertNotIn("behavioral_profile_id", Path(paths["debug"]).read_text())
            self.assertNotIn("behavioral_profile_id", Path(paths["ground_truth_events"]).read_text())
            self.assertNotIn("behavioral_profile_id", Path(paths["detector_observations"]).read_text())

    def test_non_runner_traces_have_no_fabricated_profile(self):
        bundle = build_synchronized_bundle(self.sequences, provenance=self.generation)
        self.assertNotIn("device_profile_provenance", bundle)
        wrong = to_device_profile_provenance(
            test_population(PopulationConfiguration(), [("unrelated-device", "key-1")]),
            run_id="wrong-run",
        )
        with self.assertRaisesRegex(ValueError, "no profile record"):
            assert_trace_profile_association(self.sequences, wrong)


class CapturedRunNeutralityTests(unittest.TestCase):
    def test_population_capture_preserves_dp1_reference_trace_semantics(self):
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
            sequences, population = run_simulation_with_population()
        self.assertEqual(4, len(population.behavioral_profiles.assignments))
        ordinals = {}
        rows = []
        for events, context in sequences:
            ordinals.setdefault(context.device_id, len(ordinals) + 1)
            rows.append({"device_ordinal": ordinals[context.device_id],
                         **_structure(events, context)})
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(
            PRE_DP1_RUNNER_SEMANTIC_SHA256, hashlib.sha256(payload).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
