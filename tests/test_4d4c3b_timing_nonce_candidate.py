"""Integrity checks for the frozen controlled Timing+Nonce candidate."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from simulator.composition.build import CompositionBuildError
from simulator.composition.timing_nonce_candidate import (
    CANDIDATE_SCHEMA_VERSION,
    _execute_frozen_build,
)
from simulator.composition.timing_nonce_feasibility import PLANNING_HYPOTHESIS
from simulator.role_integration.models import stable_reference


ARTIFACT_ROOT = Path("artifacts/icds-timing-nonce-candidate")


class TimingNonceCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(
            (ARTIFACT_ROOT / "scientific-candidate.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (ARTIFACT_ROOT / "build-manifest.json").read_text(encoding="utf-8")
        )
        cls.rows = cls.candidate["samples"]

    def test_exact_composition_and_one_role_per_base(self):
        self.assertEqual(CANDIDATE_SCHEMA_VERSION, self.candidate["schema_version"])
        self.assertEqual(336, len(self.rows))
        self.assertEqual(PLANNING_HYPOTHESIS, dict(Counter(
            row["privileged_metadata"]["role"]["key"] for row in self.rows
        )))
        bases = [row["privileged_metadata"]["base"]["base_trace_reference"]
                 for row in self.rows]
        self.assertEqual(336, len(set(bases)))

    def test_no_out_of_scope_family_or_rejected_base(self):
        allowed = set(PLANNING_HYPOTHESIS)
        self.assertTrue(all(
            row["privileged_metadata"]["role"]["key"] in allowed for row in self.rows
        ))
        rejected = set(self.manifest["base_pool"]["rejected_keys"])
        self.assertFalse(rejected & {
            row["privileged_metadata"]["base"]["base_construction_key"]
            for row in self.rows
        })

    def test_normal_and_hn_validation(self):
        hns = [row for row in self.rows
               if row["privileged_metadata"]["role"]["kind"] == "hard_negative"]
        self.assertEqual(144, len(hns))
        self.assertTrue(all(
            row["privileged_metadata"]["detector_truth"] == "normal"
            and row["privileged_metadata"]["hn_qualification"] == "qualified"
            and not row["privileged_metadata"]["complete_validated_invariant_ids"]
            for row in hns
        ))
        ordinary = [row for row in self.rows
                    if row["privileged_metadata"]["role"]["kind"] == "ordinary_normal"]
        self.assertEqual(96, len(ordinary))
        self.assertTrue(all(row["privileged_metadata"]["canonical_reload_valid"]
                            for row in ordinary))

    def test_anomaly_invariants_survive_canonical_reload(self):
        timing = [row for row in self.rows if row["privileged_metadata"]["role"]["variant_id"]
                  == "timestamp_inconsistency"]
        nonce = [row for row in self.rows if row["privileged_metadata"]["role"]["variant_id"]
                 == "nonce_reuse"]
        self.assertEqual(48, len(timing))
        self.assertEqual(48, len(nonce))
        self.assertTrue(all(
            "INV-TIME-OBSERVED-CAUSAL"
            in row["privileged_metadata"]["complete_validated_invariant_ids"]
            for row in timing
        ))
        self.assertTrue(all(
            "INV-REPLAY-NONCE-CROSS-ATTEMPT"
            in row["privileged_metadata"]["complete_validated_invariant_ids"]
            for row in nonce
        ))

    def test_strict_duplicate_integrity(self):
        self.assertEqual(336, self.manifest["integrity"]["unique_detector_evidence_hashes"])
        self.assertEqual(0, self.manifest["integrity"]["same_truth_duplicate_groups"])
        self.assertEqual(0, self.manifest["integrity"]["conflicting_truth_duplicate_groups"])

    def test_privileged_metadata_is_outside_o03_rows(self):
        forbidden = {
            "scientific_role", "behavioral_profile_id", "invariant_ids",
            "hn_qualification", "applicability", "control_membership",
            "population_device_key", "detector_truth", "ground_truth",
        }
        for sample in self.rows:
            for observation in sample["detector_observations"]:
                self.assertFalse(forbidden & set(observation))

    def test_c0_c1_are_metadata_projections(self):
        projections = self.manifest["projections"]
        self.assertEqual(192, projections["C0"]["count"])
        self.assertEqual(336, projections["C1"]["count"])
        all_ids = {row["sample_id"] for row in self.rows}
        self.assertEqual(all_ids, set(projections["C1"]["sample_ids"]))
        self.assertTrue(set(projections["C0"]["sample_ids"]) < all_ids)

    def test_candidate_fingerprint_and_repeat_build_evidence(self):
        payload = dict(self.candidate)
        expected = payload.pop("candidate_fingerprint")
        self.assertEqual(expected, stable_reference(payload))
        self.assertTrue(self.manifest["reproducibility"]["canonically_identical"])
        self.assertEqual(2, self.manifest["reproducibility"]["complete_builds_compared"])

    def test_opaque_runtime_material_is_stably_pseudonymized(self):
        for sample in self.rows:
            for observation in sample["detector_observations"]:
                for field in ("token_id", "nonce"):
                    value = observation[field]
                    if value is not None:
                        self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")
        self.assertIn(
            "trace-namespaced stable pseudonyms",
            self.manifest["integrity"]["opaque_runtime_material_policy"],
        )

    def test_failed_build_is_not_retried_or_reassigned(self):
        population, pool, plan = Mock(), Mock(), Mock()
        with patch(
            "simulator.composition.timing_nonce_candidate.build_composition",
            side_effect=CompositionBuildError("planned failure"),
        ) as build:
            with self.assertRaises(CompositionBuildError):
                _execute_frozen_build(population, pool, plan, 4705)
        build.assert_called_once_with(population, pool, plan, transformation_seed=4705)


if __name__ == "__main__":
    unittest.main()
