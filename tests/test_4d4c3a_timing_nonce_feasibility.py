"""Focused tests for the non-final Timing+Nonce feasibility characterization."""

from __future__ import annotations

import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.composition.timing_nonce_feasibility import (
    FEASIBILITY_ROLES,
    PLANNING_HYPOTHESIS,
    characterize_timing_nonce_feasibility,
    write_characterization_artifacts,
)
from simulator.config.integrated_role_stress import IntegratedRoleStressConfiguration


class TimingNonceFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = IntegratedRoleStressConfiguration(
            device_count=16, opportunities_per_device=2,
        )
        cls.state = random.getstate()
        with patch("simulator.role_integration.integrated_stress._integrate") as adapter:
            cls.report = characterize_timing_nonce_feasibility(cls.configuration)
            cls.adapter_calls = adapter.call_count

    def test_exact_attempt_count_and_no_replacement(self):
        admission = self.report["canonical_admission"]
        self.assertEqual(32, admission["attempted"])
        self.assertEqual(32, admission["admitted"] + admission["rejected"])
        self.assertFalse(admission["replacement_or_resampling"])
        for rejection in admission["rejections"]:
            self.assertIn("rejection_reference", rejection)
            self.assertIn("normalized_failures", rejection)
            self.assertNotIn("trace_ground_truth_statuses", rejection)

    def test_existing_applicability_builds_complete_device_role_support(self):
        matrix = self.report["device_role_support"]
        self.assertEqual(16, len(matrix))
        expected = {role.key for role in FEASIBILITY_ROLES}
        self.assertTrue(all(set(row["eligible_base_counts"]) == expected for row in matrix))
        summary = self.report["role_opportunity_summary"]
        for role in FEASIBILITY_ROLES:
            self.assertEqual(
                summary[role.key]["eligible_bases"],
                sum(row["eligible_base_counts"][role.key] for row in matrix),
            )
        relationships = self.report["applicability_predicate_relationships"]
        self.assertFalse(relationships["semantic_logic_redefined"])
        self.assertEqual(2, len(relationships["same_predicate_role_groups"]))

    def test_characterization_is_privileged_and_does_not_transform_or_export(self):
        self.assertEqual(0, self.adapter_calls)
        integrity = self.report["integrity"]
        self.assertTrue(integrity["profile_metadata_privileged"])
        self.assertFalse(integrity["detector_schema_modified"])
        self.assertFalse(integrity["adapters_invoked"])
        self.assertFalse(integrity["final_candidate_generated"])
        self.assertFalse(
            self.report["planning_hypothesis_dry_run"]["transformations_invoked"]
        )

    def test_pool_and_global_rng_are_immutable(self):
        integrity = self.report["integrity"]
        self.assertTrue(integrity["population_immutable"])
        self.assertTrue(integrity["pool_immutable"])
        self.assertTrue(integrity["bases_immutable"])
        self.assertTrue(integrity["global_rng_unchanged"])
        self.assertEqual(self.state, random.getstate())

    def test_same_configuration_is_deterministic(self):
        repeated = characterize_timing_nonce_feasibility(self.configuration)
        self.assertEqual(
            self.report["characterization_fingerprint"],
            repeated["characterization_fingerprint"],
        )
        self.assertEqual(self.report["fingerprints"], repeated["fingerprints"])
        self.assertEqual(self.report["device_role_support"], repeated["device_role_support"])

    def test_dry_run_uses_hypothesis_without_building_candidate(self):
        dry_run = self.report["planning_hypothesis_dry_run"]
        self.assertEqual(
            sum(PLANNING_HYPOTHESIS.values()), sum(dry_run["requested_counts"].values())
        )
        self.assertFalse(dry_run["final_candidate_generated"])
        self.assertIsInstance(dry_run["status"], str)

    def test_artifact_writer_emits_analysis_only_surfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_characterization_artifacts(self.report, Path(directory))
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual({".json", ".md"}, {path.suffix for path in paths})

    def test_checked_characterization_uses_full_requested_campaign(self):
        path = Path("audits/4d4c3a-timing-nonce-feasibility.json")
        if not path.exists():
            self.skipTest("checked characterization artifact not generated")
        report = json.loads(path.read_text(encoding="utf-8"))
        configuration = report["configuration"]["integrated_configuration"]
        self.assertEqual(64, configuration["device_count"])
        self.assertEqual(8, configuration["opportunities_per_device"])
        self.assertEqual(512, report["canonical_admission"]["attempted"])
        self.assertEqual(64, len(report["device_role_support"]))
        self.assertFalse(report["planning_hypothesis_dry_run"]["transformations_invoked"])


if __name__ == "__main__":
    unittest.main()
