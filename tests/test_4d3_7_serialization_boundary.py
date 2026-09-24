"""Regression for canonical six-decimal admission at the 4D.3.7 boundary."""

from __future__ import annotations

import json
import random
import unittest

from simulator.config.integrated_role_stress import IntegratedRoleStressConfiguration
from simulator.role_integration.direct_timing import _validation, materialize_base
from simulator.role_integration.integrated_stress import build_stress_population_and_pool


class CanonicalSerializationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = IntegratedRoleStressConfiguration()
        cls.rng_state = random.getstate()
        cls.population, cls.pool = build_stress_population_and_pool(cls.configuration)

    def test_original_precision_collapse_is_rejected_without_replacement(self):
        self.assertEqual(256, self.pool.diagnostics["attempted_bases"])
        self.assertEqual(255, len(self.pool.bases))
        self.assertEqual(1, len(self.pool.rejected))
        rejected = self.pool.rejected[0]
        self.assertEqual(
            "base/population-device-00000037/opportunity-3",
            rejected.base_construction_key,
        )
        self.assertEqual("canonical_serialized_admission", rejected.rejection_stage)
        source = json.loads(rejected.source_timestamp_evidence_json)
        canonical = [json.loads(item) for item in rejected.canonical_event_records_json]
        collapsed = [
            (left, right, canonical[index], canonical[index + 1])
            for index, (left, right) in enumerate(zip(source, source[1:]))
            if (
                left["semantic_timestamp"] < right["semantic_timestamp"]
                and canonical[index]["timestamp"]
                == canonical[index + 1]["timestamp"]
            )
        ]
        self.assertTrue(collapsed)
        left, right, serialized_left, serialized_right = collapsed[0]
        self.assertEqual("access_granted", left["event_type"])
        self.assertEqual("access_request", right["event_type"])
        self.assertEqual(1710166007.8187835, left["semantic_timestamp"])
        self.assertEqual(1710166007.8187845, right["semantic_timestamp"])
        self.assertEqual(1710166007.818784, serialized_left["timestamp"])
        self.assertEqual(serialized_left["timestamp"], serialized_right["timestamp"])
        evidence = json.loads(rejected.validation_evidence_json)
        self.assertFalse(evidence["build_gate"]["passed"])
        self.assertTrue(any(
            failure["validation_id"] == "c1.7.temporal.supported_subset"
            and any(
                issue.get("issue") == "semantic_time_not_strictly_monotonic"
                for issue in failure["issues"]
            )
            for failure in evidence["failures"]
        ))
        # One failed attempt remains one rejection: it is not replaced.
        self.assertEqual(
            256,
            len(self.pool.bases) + len(self.pool.rejected),
        )

    def test_every_admitted_base_reloads_as_independently_valid(self):
        for base in self.pool.bases:
            events, context = materialize_base(base)
            report = _validation(events, context)
            self.assertTrue(report.build_gate_passed)
            self.assertEqual(
                "validated_normal",
                report.trace_ground_truth_statuses[context.trace_id],
            )
            self.assertEqual(
                (), report.trace_validated_invariant_ids[context.trace_id],
            )

    def test_canonical_admission_does_not_perturb_global_rng(self):
        self.assertEqual(self.rng_state, random.getstate())


if __name__ == "__main__":
    unittest.main()
