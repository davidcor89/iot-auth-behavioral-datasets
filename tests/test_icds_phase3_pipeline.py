"""Phase-3 leakage, model, persistence, metric, and grouped-cohort safeguards."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.evaluation.experiment_manifest import (
    FRESH_ATTEMPT_HN,
    NONCE_ANOMALY,
    ORDINARY,
    SAME_ATTEMPT_HN,
    TIMING_ANOMALY,
    TIMING_HN,
)
from ml.evaluation.grouped_cohorts import build_grouped_cohorts
from ml.evaluation.metrics import reconstruct_metrics
from ml.evaluation.pipeline import (
    CONDITIONS,
    DT_PARAMETERS,
    EXPECTED_PHASE2_FILE_SHA256,
    EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
    LR_PARAMETERS,
    MODEL_SEED_BASE,
    PIPELINE_SPEC_VERSION,
    RF_PARAMETERS,
    MatrixBundle,
    PipelineContractError,
    assemble_matrix,
    build_estimator,
    execute_run,
    feature_names_for_profile,
    feature_profile_for_condition,
    feature_type_inventory,
    fit_estimator,
    fitted_scaler_state,
    load_frozen_inputs,
    model_seed,
    pipeline_specification,
    predict_estimator,
    validate_comparison_contract,
)
from ml.evaluation.trace_features import C1_FEATURE_NAMES, C2_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SPEC_PATH = ROOT / "IoT_auth_simulator/ml/icds/icds_pipeline_spec_v1.json"
GROUPED_PATH = ROOT / "artifacts/icds-ml-phase3/icds_grouped_cohorts_v1.json"


class Phase3PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_frozen_inputs(ROOT)
        cls.rep1 = cls.inputs.experiment_manifest["repetitions"][0]
        cls.c0_lr = execute_run(1, "C0", "LR", inputs=cls.inputs)

    def test_upstream_freeze_and_feature_inventory(self):
        self.assertEqual(
            self.inputs.experiment_manifest["manifest_fingerprint"],
            EXPECTED_PHASE2_MANIFEST_FINGERPRINT,
        )
        self.assertEqual(
            self.inputs.phase1_audit["missing_or_non_applicable_counts_c1"], {},
        )
        inventory = feature_type_inventory(self.inputs)
        self.assertEqual(inventory["feature_count_C1"], 90)
        self.assertEqual(inventory["feature_count_C2"], 87)
        self.assertEqual(
            inventory["declared_output_types_C1"],
            {"integer": 49, "number": 35, "number|null": 6},
        )
        self.assertEqual(inventory["candidate_missing_values"], 0)
        self.assertEqual(inventory["candidate_non_numeric_values"], 0)
        self.assertEqual(inventory["candidate_non_finite_values"], 0)
        self.assertFalse(inventory["imputation_required"])
        self.assertFalse(inventory["categorical_encoding_required"])

    def test_exact_feature_profiles_and_matrix_isolation(self):
        self.assertEqual(feature_names_for_profile("C1"), C1_FEATURE_NAMES)
        self.assertEqual(feature_names_for_profile("C2"), C2_FEATURE_NAMES)
        self.assertEqual(set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES), {
            "semantic_timing_min_signed_causal_margin",
            "semantic_nonce_cross_attempt_value_count",
            "semantic_nonce_max_attempt_sharing_count",
        })
        forbidden = {
            "sample_id", "trace_id", "device_id", "population_device_key",
            "scientific_role", "detector_truth", "anomaly_label", "provenance",
        }
        self.assertTrue(forbidden.isdisjoint(C1_FEATURE_NAMES))
        for condition in CONDITIONS:
            profile = feature_profile_for_condition(condition)
            matrix = assemble_matrix(
                self.inputs, self.rep1[condition]["training_trace_ids"], profile,
            )
            expected_width = 87 if condition == "C2" else 90
            self.assertEqual(matrix.X.shape, (144, expected_width))
            self.assertTrue(np.isfinite(matrix.X).all())
            self.assertEqual(matrix.X.dtype, np.float64)

    def test_comparison_invariants_abort_before_fit_when_tampered(self):
        for repetition in range(1, 6):
            validate_comparison_contract(self.inputs.experiment_manifest, repetition)
        tampered = deepcopy(self.inputs.experiment_manifest)
        tampered["repetitions"][0]["C1"]["training_trace_ids"][0] = (
            tampered["repetitions"][0]["C0"]["training_trace_ids"][0]
        )
        with self.assertRaises(PipelineContractError):
            validate_comparison_contract(tampered, 1)

    def test_model_contract_and_seed_policy(self):
        self.assertEqual(MODEL_SEED_BASE, {"LR": 51000, "DT": 52000, "RF": 53000})
        for repetition in range(1, 6):
            self.assertEqual(model_seed(repetition, "LR"), 51000 + repetition)
            self.assertEqual(model_seed(repetition, "DT"), 52000 + repetition)
            self.assertEqual(model_seed(repetition, "RF"), 53000 + repetition)
        lr = build_estimator("LR", 51001)
        self.assertIsInstance(lr, Pipeline)
        self.assertIsInstance(lr.named_steps["model"], LogisticRegression)
        for name, value in LR_PARAMETERS.items():
            self.assertEqual(lr.named_steps["model"].get_params()[name], value)
        dt = build_estimator("DT", 52001)
        self.assertIsInstance(dt, DecisionTreeClassifier)
        for name, value in DT_PARAMETERS.items():
            self.assertEqual(dt.get_params()[name], value)
        rf = build_estimator("RF", 53001)
        self.assertIsInstance(rf, RandomForestClassifier)
        for name, value in RF_PARAMETERS.items():
            self.assertEqual(rf.get_params()[name], value)
        self.assertIsNot(build_estimator("LR", 51001), build_estimator("LR", 51001))

    def test_lr_preprocessing_is_training_only_and_run_local(self):
        train = assemble_matrix(
            self.inputs, self.rep1["C0"]["training_trace_ids"], "C1",
        )
        test = assemble_matrix(
            self.inputs, self.rep1["C0"]["common_test_trace_ids"], "C1",
        )
        estimator = fit_estimator("LR", 51001, train)
        before = fitted_scaler_state(estimator)
        altered_test = MatrixBundle(
            test.trace_ids, test.feature_names, test.X + 1_000_000.0, test.y,
            test.scientific_roles, test.device_keys,
        )
        predict_estimator(estimator, altered_test)
        self.assertEqual(before, fitted_scaler_state(estimator))
        self.assertEqual(before["samples_seen"], [144])

        c1_train = assemble_matrix(
            self.inputs, self.rep1["C1"]["training_trace_ids"], "C1",
        )
        c0_estimator = fit_estimator("LR", 51001, train)
        c1_estimator = fit_estimator("LR", 51001, c1_train)
        self.assertIsNot(
            c0_estimator.named_steps["scaler"], c1_estimator.named_steps["scaler"],
        )
        self.assertFalse(np.array_equal(
            c0_estimator.named_steps["scaler"].mean_,
            c1_estimator.named_steps["scaler"].mean_,
        ))

    def test_prediction_contract_and_persistence(self):
        output = self.c0_lr
        expected_ids = set(self.rep1["C0"]["common_test_trace_ids"])
        self.assertEqual(len(output.predictions), 84)
        self.assertEqual({row["trace_id"] for row in output.predictions}, expected_ids)
        self.assertEqual(len({row["trace_id"] for row in output.predictions}), 84)
        for record in output.predictions:
            self.assertEqual(
                record["true_target"],
                1 if record["scientific_role"] in {TIMING_ANOMALY, NONCE_ANOMALY} else 0,
            )
            self.assertTrue(math.isfinite(record["predicted_anomaly_probability"]))
            self.assertGreaterEqual(record["predicted_anomaly_probability"], 0.0)
            self.assertLessEqual(record["predicted_anomaly_probability"], 1.0)
            self.assertEqual(record["split"], "test")
            self.assertEqual(record["execution_mode"], "ENGINEERING_SMOKE_TEST")
            self.assertEqual(record["repository_commit"], output.provenance["repository_commit"])
            self.assertIsNotNone(record["repository_commit"])
        self.assertFalse(output.provenance["test_used_for_fit_or_preprocessing"])
        self.assertFalse(output.provenance["threshold_tuning_performed"])
        with tempfile.TemporaryDirectory() as directory:
            persisted = execute_run(
                1, "C0", "LR", inputs=self.inputs, output_root=Path(directory),
            )
            run_dir = Path(directory) / PIPELINE_SPEC_VERSION / "repetition-1/C0/LR"
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            records = [json.loads(line) for line in
                       (run_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 84)
            self.assertTrue((run_dir / "run-provenance.json").is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())

    def test_prediction_determinism_for_all_fixed_models(self):
        for model in ("LR", "DT", "RF"):
            with self.subTest(model=model):
                first = execute_run(1, "C0", model, inputs=self.inputs)
                second = execute_run(1, "C0", model, inputs=self.inputs)
                normalized_first = [
                    (row["trace_id"], row["predicted_class"], row["predicted_anomaly_probability"])
                    for row in first.predictions
                ]
                normalized_second = [
                    (row["trace_id"], row["predicted_class"], row["predicted_anomaly_probability"])
                    for row in second.predictions
                ]
                self.assertEqual(normalized_first, normalized_second)

    def test_metric_reconstruction_from_prediction_records(self):
        roles_and_values = [
            (ORDINARY, 0, 0, 0.1), (ORDINARY, 0, 1, 0.8),
            (TIMING_HN, 0, 1, 0.7), (SAME_ATTEMPT_HN, 0, 0, 0.2),
            (FRESH_ATTEMPT_HN, 0, 0, 0.3),
            (TIMING_ANOMALY, 1, 1, 0.9), (TIMING_ANOMALY, 1, 0, 0.4),
            (NONCE_ANOMALY, 1, 1, 0.9), (NONCE_ANOMALY, 1, 1, 0.8),
        ]
        records = [{
            "trace_id": f"fixture-{index}", "repetition": 1, "condition": "C0",
            "model": "LR", "scientific_role": role, "true_target": truth,
            "predicted_class": prediction, "predicted_anomaly_probability": score,
        } for index, (role, truth, prediction, score) in enumerate(roles_and_values)]
        metrics = reconstruct_metrics(records)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 3, "fp": 2, "fn": 1, "tp": 3})
        self.assertAlmostEqual(metrics["anomaly_precision"], 0.6)
        self.assertAlmostEqual(metrics["anomaly_recall"], 0.75)
        self.assertAlmostEqual(metrics["anomaly_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.675)
        self.assertEqual(
            metrics["normal_role_diagnostics"][TIMING_HN]["false_positive_rate"], 1.0,
        )
        self.assertEqual(
            metrics["anomaly_role_diagnostics"][TIMING_ANOMALY]["recall"], 0.5,
        )

    def test_grouped_matched_cohorts_are_frozen_and_device_disjoint(self):
        artifact = json.loads(GROUPED_PATH.read_text(encoding="utf-8"))
        rebuilt = build_grouped_cohorts(self.inputs.candidate, self.inputs.experiment_manifest)
        self.assertEqual(artifact, rebuilt)
        self.assertEqual(artifact["C0"]["counts"]["binary"], {"normal": 71, "anomaly": 71})
        self.assertEqual(artifact["C1"]["counts"]["binary"], {"normal": 71, "anomaly": 71})
        self.assertEqual(artifact["C0"]["counts"]["total"], 142)
        self.assertEqual(artifact["C1"]["counts"]["total"], 142)
        common_anomalies = set(artifact["common_anomaly_trace_ids"])
        self.assertTrue(common_anomalies <= set(artifact["C0"]["training_trace_ids"]))
        self.assertTrue(common_anomalies <= set(artifact["C1"]["training_trace_ids"]))
        for condition in ("C0", "C1"):
            actual_anomalies = {
                sample_id for sample_id in artifact[condition]["training_trace_ids"]
                if self.inputs.roles[sample_id] in {TIMING_ANOMALY, NONCE_ANOMALY}
            }
            self.assertEqual(actual_anomalies, common_anomalies)
        self.assertTrue(set(artifact["train_device_ids"]).isdisjoint(artifact["test_device_ids"]))
        self.assertFalse(artifact["features_inspected_for_selection"])
        self.assertFalse(artifact["device_profiles_inspected_for_selection"])
        self.assertFalse(artifact["models_executed"])

    def test_pipeline_spec_and_scientific_execution_guard(self):
        checked_in = json.loads(PIPELINE_SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, pipeline_specification())
        self.assertNotIn("GridSearchCV", json.dumps(checked_in))
        self.assertNotIn("RandomizedSearchCV", json.dumps(checked_in))
        with self.assertRaises(PipelineContractError):
            execute_run(
                1, "C0", "LR", inputs=self.inputs,
                execution_mode="SCIENTIFIC_EXECUTION",
            )


if __name__ == "__main__":
    unittest.main()
