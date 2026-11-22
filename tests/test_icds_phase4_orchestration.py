"""Pre-execution safeguards for the frozen Phase-4 orchestrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.evaluation.phase4_experiment import (
    EXPECTED_GROUPED_COHORT_FINGERPRINT,
    EXPECTED_PIPELINE_SPEC_FINGERPRINT,
    Phase4ExecutionError,
    distribution_summary,
    grouped_run_plan,
    primary_run_plan,
    validate_prediction_campaign,
    verify_phase4_inputs,
)
from ml.evaluation.pipeline import assemble_matrix, model_seed


ROOT = Path(__file__).resolve().parents[1]


class Phase4OrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.pipeline_spec, cls.grouped = verify_phase4_inputs(ROOT)

    def test_frozen_inputs_and_plans(self):
        self.assertEqual(
            self.pipeline_spec["specification_fingerprint"],
            EXPECTED_PIPELINE_SPEC_FINGERPRINT,
        )
        self.assertEqual(
            self.grouped["artifact_fingerprint"], EXPECTED_GROUPED_COHORT_FINGERPRINT,
        )
        self.assertEqual(len(primary_run_plan()), 45)
        self.assertEqual(len(set(primary_run_plan())), 45)
        self.assertEqual(len(grouped_run_plan()), 9)
        self.assertEqual(len(set(grouped_run_plan())), 9)

    def test_distribution_summary_retains_all_repetitions(self):
        summary = distribution_summary([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(summary["values_by_repetition"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(summary["median"], 0.5)
        self.assertEqual(summary["q1"], 0.25)
        self.assertEqual(summary["q3"], 0.75)
        self.assertEqual(summary["iqr"], 0.5)
        self.assertEqual(summary["minimum"], 0.0)
        self.assertEqual(summary["maximum"], 1.0)

    def test_grouped_c2_is_direct_c1_membership_representation_alias(self):
        self.assertEqual(self.grouped["C1"]["counts"]["total"], 142)
        c1 = assemble_matrix(
            self.inputs, self.grouped["C1"]["training_trace_ids"], "C1",
        )
        c2 = assemble_matrix(
            self.inputs, self.grouped["C1"]["training_trace_ids"], "C2",
        )
        self.assertEqual(c1.trace_ids, c2.trace_ids)
        self.assertEqual(c1.y.tolist(), c2.y.tolist())
        self.assertEqual(c1.X.shape, (142, 90))
        self.assertEqual(c2.X.shape, (142, 87))
        self.assertEqual(model_seed(1, "LR"), 51001)
        self.assertEqual(model_seed(1, "DT"), 52001)
        self.assertEqual(model_seed(1, "RF"), 53001)

    def test_prediction_campaign_validation_rejects_duplicates(self):
        base = {
            "scientific_scope": "fixture",
            "repetition": 1,
            "condition": "C0",
            "model": "LR",
            "split": "test",
            "trace_id": "trace-one",
            "true_target": 0,
            "predicted_class": 0,
            "predicted_anomaly_probability": 0.1,
            "scientific_role": "ordinary_normal",
        }
        with self.assertRaises(Phase4ExecutionError):
            validate_prediction_campaign(
                [base, dict(base)], expected_run_count=1, expected_predictions_per_run=2,
            )

    def test_output_overwrite_guard_is_declared_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertTrue(path.exists())
            # execute_phase4's first operation rejects an existing output root;
            # do not invoke it here because this is a no-fit pre-execution test.
            source = (ROOT / "IoT_auth_simulator/ml/evaluation/phase4_experiment.py").read_text()
            self.assertIn("refusing to overwrite Phase-4 output", source)


if __name__ == "__main__":
    unittest.main()
