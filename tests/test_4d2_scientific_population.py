"""4D.2 fixed-cohort scientific population coverage and compatibility tests."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.population_timing import PopulationTimingConfiguration
from simulator.config.scientific_population import (
    LIFECYCLE_SIMPLEX_PARTITION_VERSION,
    SCIENTIFIC_POPULATION_RNG_VERSION,
    SCIENTIFIC_POPULATION_SAMPLER_VERSION,
    ScientificLifecycleSpace,
    ScientificPopulationConfiguration,
    ScientificScalarSpace,
)
from simulator.config.settings import cfg
from simulator.data.publication import load_dataset_profile_association
from simulator.data.synchronized_views import SynchronizedOutputViews, build_synchronized_bundle
from simulator.device_profiles.population import (
    build_profile_population,
    population_device_key_from_index,
)
from simulator.device_profiles.provenance import (
    PROFILE_PROVENANCE_SCHEMA_VERSION,
    PROFILE_SAMPLER_VERSION,
    SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION,
    load_device_profile_provenance,
    to_device_profile_provenance,
    validate_device_profile_provenance,
)
from simulator.device_profiles.scientific_population import (
    SCALAR_DIMENSIONS,
    SCIENTIFIC_POPULATION_FINGERPRINT_VERSION,
    build_scientific_profile_population,
    lifecycle_point_from_reference,
    lifecycle_simplex_cells,
    scientific_population_fingerprint,
    scientific_scalar_from_reference,
)
from simulator.planning.models import TraceConstructionRequest
from simulator.provenance import FEATURE_SCHEMA_VERSION, build_generation_provenance
from simulator.runner import RunPopulation, run_simulation_with_population


def bindings(count: int, *, reverse: bool = False):
    values = [(f"uuid-{index}", f"scientific-device-{index:03d}") for index in range(count)]
    return list(reversed(values)) if reverse else values


def keyed(population):
    return {item.population_device_key: item for item in population.assignments}


def components(profile):
    return {
        "T": profile.timing, "A": profile.activity, "R": profile.recovery,
        "N": profile.renewal, "L": profile.lifecycle,
    }


def run_population(population) -> RunPopulation:
    devices = [SimpleNamespace(device_id=item.device_id) for item in population.assignments]
    return RunPopulation(
        devices=devices,
        devices_by_id={item.device_id: item for item in devices},
        semantic_contexts={item.device_id: object() for item in devices},
        behavioral_profiles=population,
    )


class ScientificCoverageTests(unittest.TestCase):
    def setUp(self):
        self.configuration = ScientificPopulationConfiguration(device_count=64, population_seed=4101)
        self.population = build_scientific_profile_population(
            bindings(64), self.configuration,
        )

    def test_balanced_scalar_coverage_and_strict_endpoints(self):
        diagnostics = self.population.diagnostics["scalar_marginals"]
        self.assertEqual(set(SCALAR_DIMENSIONS), set(diagnostics))
        for name, record in diagnostics.items():
            with self.subTest(name=name):
                self.assertEqual([16, 16, 16, 16], record["stratum_occupancy"])
                self.assertEqual(0, record["endpoint_violations"])
                self.assertLess(record["minimum"], record["maximum"])
        for assignment in self.population.assignments:
            profile = assignment.profile
            self.assertTrue(0 < profile.activity < 1)
            self.assertTrue(0 < profile.recovery < 1)
            self.assertTrue(0 < profile.renewal < 1)
            for tendency in (
                profile.timing.auth, profile.timing.transition,
                profile.timing.access, profile.timing.renewal,
            ):
                self.assertTrue(0.75 < tendency.center_modifier < 1.25)
                self.assertTrue(0.75 < tendency.dispersion_modifier < 1.25)

    def test_simplex_partition_is_explicit_balanced_symmetric_and_interior(self):
        cells = lifecycle_simplex_cells(4)
        self.assertEqual(16, len(cells))
        self.assertEqual(16, len({cell.cell_id for cell in cells}))
        occupancy = self.population.diagnostics["lifecycle"]["cell_occupancy"]
        self.assertEqual({4}, set(occupancy.values()))
        vertex_sets = {
            frozenset(tuple(round(value, 12) for value in vertex) for vertex in cell.vertices)
            for cell in cells
        }
        for cell in cells:
            for permutation in ((1, 0, 2), (2, 1, 0)):
                permuted = frozenset(
                    tuple(round(vertex[index], 12) for index in permutation)
                    for vertex in cell.vertices
                )
                self.assertIn(permuted, vertex_sets)
        by_cell = {cell.cell_id: cell for cell in cells}
        for assignment in self.population.assignments:
            point = (
                assignment.profile.lifecycle.auth,
                assignment.profile.lifecycle.session,
                assignment.profile.lifecycle.access,
            )
            self.assertTrue(all(value > 0 for value in point))
            self.assertAlmostEqual(1.0, sum(point), places=15)
            cell = by_cell[assignment.lifecycle_cell_id]
            # Independent two-coordinate barycentric membership calculation.
            v0, v1, v2 = cell.vertices
            ax, ay = v1[0] - v0[0], v1[1] - v0[1]
            bx, by = v2[0] - v0[0], v2[1] - v0[1]
            px, py = point[0] - v0[0], point[1] - v0[1]
            determinant = ax * by - ay * bx
            alpha = (px * by - py * bx) / determinant
            beta = (ax * py - ay * px) / determinant
            self.assertGreater(alpha, 0)
            self.assertGreater(beta, 0)
            self.assertGreater(1 - alpha - beta, 0)

    def test_recorded_references_independently_reconstruct_values(self):
        cells = {cell.cell_id: cell for cell in lifecycle_simplex_cells(4)}
        spaces = {
            "T.AUTH.center": self.configuration.timing.auth_center,
            "T.AUTH.dispersion": self.configuration.timing.auth_dispersion,
            "T.TRANSITION.center": self.configuration.timing.transition_center,
            "T.TRANSITION.dispersion": self.configuration.timing.transition_dispersion,
            "T.ACCESS.center": self.configuration.timing.access_center,
            "T.ACCESS.dispersion": self.configuration.timing.access_dispersion,
            "T.RENEWAL.center": self.configuration.timing.renewal_center,
            "T.RENEWAL.dispersion": self.configuration.timing.renewal_dispersion,
            "A": self.configuration.activity, "R": self.configuration.recovery,
            "N": self.configuration.renewal,
        }
        for assignment in self.population.assignments:
            profile = assignment.profile
            actual = {
                "T.AUTH.center": profile.timing.auth.center_modifier,
                "T.AUTH.dispersion": profile.timing.auth.dispersion_modifier,
                "T.TRANSITION.center": profile.timing.transition.center_modifier,
                "T.TRANSITION.dispersion": profile.timing.transition.dispersion_modifier,
                "T.ACCESS.center": profile.timing.access.center_modifier,
                "T.ACCESS.dispersion": profile.timing.access.dispersion_modifier,
                "T.RENEWAL.center": profile.timing.renewal.center_modifier,
                "T.RENEWAL.dispersion": profile.timing.renewal.dispersion_modifier,
                "A": profile.activity, "R": profile.recovery, "N": profile.renewal,
            }
            for name in SCALAR_DIMENSIONS:
                self.assertEqual(
                    actual[name], scientific_scalar_from_reference(
                        spaces[name], assignment.scalar_strata[name],
                        assignment.component_references[name],
                    ),
                )
            self.assertEqual(
                (profile.lifecycle.auth, profile.lifecycle.session, profile.lifecycle.access),
                lifecycle_point_from_reference(
                    cells[assignment.lifecycle_cell_id],
                    assignment.component_references["L"],
                ),
            )
        self.assertEqual(
            self.population.population_fingerprint,
            scientific_population_fingerprint(
                self.configuration, self.population.assignments,
            ),
        )

    def test_diagnostics_are_descriptive_and_conformant(self):
        diagnostics = self.population.diagnostics
        self.assertTrue(diagnostics["descriptive_only_no_resampling"])
        self.assertIsNone(
            diagnostics["cross_dimension"]["acceptance_threshold"],
        )
        self.assertGreaterEqual(
            diagnostics["cross_dimension"]["maximum_absolute_off_diagonal_pearson"], 0,
        )
        self.assertGreaterEqual(
            diagnostics["cross_dimension"]["maximum_absolute_off_diagonal_spearman"], 0,
        )
        self.assertEqual(0, diagnostics["identity"]["duplicate_exact_profile_count"])
        self.assertGreater(diagnostics["identity"]["pairwise_distance_minimum"], 0)
        self.assertGreater(diagnostics["lifecycle"]["minimum_weight"], 0)
        self.assertLessEqual(diagnostics["lifecycle"]["maximum_normalization_error"], 1e-12)


class ScientificReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.configuration = ScientificPopulationConfiguration(device_count=64, population_seed=4101)
        self.first = build_scientific_profile_population(bindings(64), self.configuration)

    def assert_only_component_changes(self, changed_configuration, changed_name):
        changed = build_scientific_profile_population(bindings(64), changed_configuration)
        first, second = keyed(self.first), keyed(changed)
        observed_change = False
        for key in first:
            before, after = components(first[key].profile), components(second[key].profile)
            for name in before:
                if name == changed_name:
                    observed_change |= before[name] != after[name]
                else:
                    self.assertEqual(before[name], after[name], (changed_name, name, key))
        self.assertTrue(observed_change)

    def test_component_specific_configuration_isolation(self):
        timing = replace(
            self.configuration.timing,
            auth_center=ScientificScalarSpace((0.80, 1.20), 5),
        )
        self.assert_only_component_changes(replace(self.configuration, timing=timing), "T")
        self.assert_only_component_changes(
            replace(self.configuration, activity=ScientificScalarSpace((0.0, 1.0), 5)), "A",
        )
        self.assert_only_component_changes(
            replace(self.configuration, recovery=ScientificScalarSpace((0.0, 1.0), 5)), "R",
        )
        self.assert_only_component_changes(
            replace(self.configuration, renewal=ScientificScalarSpace((0.0, 1.0), 5)), "N",
        )
        self.assert_only_component_changes(
            replace(self.configuration, lifecycle=ScientificLifecycleSpace(subdivision=5)), "L",
        )

    def test_cohort_order_invariance_duplicate_rejection_and_cohort_change(self):
        reordered = build_scientific_profile_population(
            bindings(64, reverse=True), self.configuration,
        )
        self.assertEqual(self.first.population_fingerprint, reordered.population_fingerprint)
        self.assertEqual(
            {key: item.profile for key, item in keyed(self.first).items()},
            {key: item.profile for key, item in keyed(reordered).items()},
        )
        duplicate = bindings(64)
        duplicate[-1] = (duplicate[-1][0], duplicate[0][1])
        with self.assertRaisesRegex(ValueError, "stable keys must be unique"):
            build_scientific_profile_population(duplicate, self.configuration)
        larger_configuration = replace(self.configuration, device_count=65)
        larger = build_scientific_profile_population(bindings(65), larger_configuration)
        self.assertNotEqual(self.first.population_fingerprint, larger.population_fingerprint)
        self.assertTrue(any(
            keyed(self.first)[key].profile != keyed(larger)[key].profile
            for key in keyed(self.first)
        ))


class PopulationModeProvenanceTests(unittest.TestCase):
    def test_legacy_sampler_and_fingerprint_golden_remain_unchanged(self):
        old_bindings = [(f"uuid-{index}", f"stable-{index:03d}") for index in range(8)]
        legacy = build_profile_population(
            old_bindings, PopulationConfiguration(population_seed=19),
        )
        artifact = to_device_profile_provenance(
            run_population(legacy), run_id="legacy-golden",
        )
        self.assertEqual(PROFILE_PROVENANCE_SCHEMA_VERSION, artifact["provenance_schema_version"])
        self.assertEqual(PROFILE_SAMPLER_VERSION,
                         artifact["population"]["configuration"]["sampler_version"])
        self.assertEqual(
            "sha256:fc7b733047ef90b72576202b575d3a7f83c939cdfe028cbbff1588fde3eb5a9d",
            artifact["population"]["population_fingerprint"],
        )
        validate_device_profile_provenance(artifact)

    def test_scientific_provenance_round_trip_and_exact_dispatch(self):
        configuration = ScientificPopulationConfiguration(device_count=16, population_seed=4101)
        population = build_scientific_profile_population(bindings(16), configuration)
        artifact = to_device_profile_provenance(
            run_population(population), run_id="scientific-round-trip",
        )
        self.assertEqual(
            SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION,
            artifact["provenance_schema_version"],
        )
        self.assertEqual(SCIENTIFIC_POPULATION_SAMPLER_VERSION,
                         artifact["population"]["sampler_version"])
        self.assertEqual(SCIENTIFIC_POPULATION_RNG_VERSION,
                         artifact["population"]["rng_derivation_version"])
        self.assertEqual(SCIENTIFIC_POPULATION_FINGERPRINT_VERSION,
                         artifact["population"]["fingerprint_version"])
        validate_device_profile_provenance(artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(artifact, load_device_profile_provenance(path))
        altered = copy.deepcopy(artifact)
        altered["devices"][0]["scalar_strata"]["A"] += 1
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            validate_device_profile_provenance(altered)


class ScientificRunnerIntegrationTests(unittest.TestCase):
    def test_explicit_scientific_mode_preserves_five_dimensional_ordinary_path(self):
        configuration = ScientificPopulationConfiguration(device_count=16, population_seed=4101)
        requests = [TraceConstructionRequest.profile_conditioned_lifecycle(
            trace_construction_key=f"scientific-trace-{index:03d}",
            scenario_id=f"scientific-scenario-{index:03d}",
        ) for index in range(32)]
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345, "num_devices": 16,
                "num_sessions_normal": 0, "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            sequences, population = run_simulation_with_population(
                planning_requests=requests,
                planning_root_seed=4102,
                profile_configuration=configuration,
            )
        self.assertEqual(16, len(population.behavioral_profiles.assignments))
        records = population.trace_planning_records
        self.assertEqual(32, len(records))
        for record in records.values():
            self.assertIn("selected_lifecycle_depth", record)
            self.assertIn("selected_recovery_choice", record)
            self.assertIn("selected_renewal_choice", record)
            self.assertIn("selected_protected_operation_count", record)
            self.assertIn("timing_realization", record)
        bundle = build_synchronized_bundle(
            sequences,
            provenance=build_generation_provenance(cfg, run_id="4d2-characterization"),
            run_population=population,
        )
        self.assertEqual({"validated_normal"}, set(
            bundle["validation"]["trace_ground_truth_statuses"].values()
        ))
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        self.assertNotIn("hard_negative_qualification", bundle)
        serialized_observations = json.dumps(bundle["detector_observations"])
        for forbidden in (
            "behavioral_profile", "population_device", "scalar_strata",
            "lifecycle_cell", "population_fingerprint", "coverage_diagnostics",
        ):
            self.assertNotIn(forbidden, serialized_observations)

    def test_scientific_companion_publication_and_timing_fingerprint_separation(self):
        configuration = ScientificPopulationConfiguration(device_count=16, population_seed=4101)
        requests = [TraceConstructionRequest.profile_conditioned_lifecycle(
            trace_construction_key=f"publication-trace-{index:03d}",
            scenario_id=f"publication-scenario-{index:03d}",
        ) for index in range(4)]
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345, "num_devices": 16,
                "num_sessions_normal": 0, "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            sequences, population = run_simulation_with_population(
                planning_requests=requests, planning_root_seed=4102,
                profile_configuration=configuration,
                timing_configuration=PopulationTimingConfiguration(
                    minimum_causal_increment_s=0.000002,
                ),
            )
        fingerprint = population.behavioral_profiles.population_fingerprint
        rebuilt = build_scientific_profile_population(
            [(f"other-uuid-{index}", population_device_key_from_index(index))
             for index in range(16)],
            configuration,
        )
        self.assertEqual(fingerprint, rebuilt.population_fingerprint)
        with tempfile.TemporaryDirectory() as directory:
            provenance = build_generation_provenance(cfg, run_id="4d2-publication")
            SynchronizedOutputViews(directory).save(
                sequences, stem="scientific", provenance=provenance,
                run_population=population,
            )
            association = load_dataset_profile_association(directory, "scientific")
            self.assertEqual("present", association["status"])
            self.assertEqual(
                SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION,
                association["profile_provenance"]["provenance_schema_version"],
            )
            self.assertEqual(
                fingerprint,
                association["profile_provenance"]["population"]["population_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
