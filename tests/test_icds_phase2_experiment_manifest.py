"""Phase-2 split, cohort, safety-boundary, and determinism safeguards."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ml.icds.experiment_manifest import (
    ANOMALY_ROLES,
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_CANDIDATE_FINGERPRINT,
    EXPECTED_FEATURE_SPEC_FILE_SHA256,
    EXPECTED_ROLE_COUNTS,
    FRESH_ATTEMPT_HN,
    HN_ROLES,
    NONCE_ANOMALY,
    ORDINARY,
    ROLES,
    SAME_ATTEMPT_HN,
    SPLIT_SEEDS,
    TEST_ROLE_COUNTS,
    TIMING_ANOMALY,
    TIMING_HN,
    build_experiment_manifest,
    candidate_control_records,
    device_aware_select,
    file_sha256,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "artifacts/icds-timing-nonce-candidate/scientific-candidate.json"
BUILD_MANIFEST_PATH = ROOT / "artifacts/icds-timing-nonce-candidate/build-manifest.json"
FEATURE_SPEC_PATH = ROOT / "IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json"
PHASE1_AUDIT_PATH = ROOT / "artifacts/icds-ml-phase1/structural-extraction-audit.json"
PHASE2_MANIFEST_PATH = ROOT / "artifacts/icds-ml-phase2/icds_experiment_manifest_v1.json"


def role_counts(ids: set[str], role_index: dict[str, str]) -> Counter[str]:
    return Counter(role_index[sample_id] for sample_id in ids)


class ExperimentManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(PHASE2_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.build_manifest = json.loads(BUILD_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.phase1_audit = json.loads(PHASE1_AUDIT_PATH.read_text(encoding="utf-8"))

    def fresh_manifest(self, candidate=None):
        return build_experiment_manifest(
            self.candidate if candidate is None else candidate,
            self.build_manifest,
            self.phase1_audit,
            candidate_file_sha256=file_sha256(CANDIDATE_PATH),
            feature_spec_file_sha256=file_sha256(FEATURE_SPEC_PATH),
        )

    def test_frozen_candidate_and_phase1_contract(self):
        self.assertEqual(file_sha256(CANDIDATE_PATH), EXPECTED_CANDIDATE_FILE_SHA256)
        self.assertEqual(file_sha256(FEATURE_SPEC_PATH), EXPECTED_FEATURE_SPEC_FILE_SHA256)
        self.assertEqual(self.candidate["candidate_fingerprint"], EXPECTED_CANDIDATE_FINGERPRINT)
        records = candidate_control_records(self.candidate)
        self.assertEqual(len(records), 336)
        self.assertEqual(len({record.device_key for record in records}), 64)
        self.assertEqual(Counter(record.role for record in records), Counter(EXPECTED_ROLE_COUNTS))

    def test_checked_in_manifest_is_exact_deterministic_build(self):
        self.assertEqual(self.fresh_manifest(), self.manifest)
        validate_manifest(self.manifest)
        self.assertEqual(
            [item["split_seed"] for item in self.manifest["repetitions"]],
            list(SPLIT_SEEDS),
        )

    def test_primary_split_and_common_test(self):
        for item in self.manifest["repetitions"]:
            with self.subTest(repetition=item["repetition"]):
                primary = item["primary_split"]
                train = set(primary["training_pool_trace_ids"])
                test = set(primary["common_test_trace_ids"])
                self.assertEqual(len(train), 252)
                self.assertEqual(len(test), 84)
                self.assertTrue(train.isdisjoint(test))
                self.assertEqual(train | test, set(self.manifest["candidate_trace_ids"]))
                self.assertEqual(primary["common_test_counts"]["roles"], TEST_ROLE_COUNTS)
                self.assertEqual(
                    primary["common_test_counts"]["binary"],
                    {"normal": 60, "anomaly": 24, "total": 84},
                )
                self.assertEqual(
                    item["C0"]["common_test_trace_ids"],
                    item["C1"]["common_test_trace_ids"],
                )
                self.assertEqual(
                    item["C1"]["common_test_trace_ids"],
                    item["C2"]["common_test_trace_ids"],
                )

    def test_c0_c1_controlled_replacement(self):
        role_index = self.manifest["trace_role_index"]
        for item in self.manifest["repetitions"]:
            with self.subTest(repetition=item["repetition"]):
                c0 = set(item["C0"]["training_trace_ids"])
                c1 = set(item["C1"]["training_trace_ids"])
                test = set(item["primary_split"]["common_test_trace_ids"])
                retained = set(item["C1"]["retained_ordinary_normal_trace_ids"])
                replaced = set(item["C1"]["replaced_ordinary_normal_trace_ids"])
                introduced = set(item["C1"]["introduced_hard_negative_trace_ids"])
                self.assertEqual(len(c0), 144)
                self.assertEqual(len(c1), 144)
                self.assertTrue(c0.isdisjoint(test) and c1.isdisjoint(test))
                self.assertEqual(role_counts(c0, role_index), Counter({
                    ORDINARY: 72, TIMING_ANOMALY: 36, NONCE_ANOMALY: 36,
                }))
                self.assertEqual(role_counts(c1, role_index), Counter({
                    ORDINARY: 36, TIMING_HN: 12, SAME_ATTEMPT_HN: 12,
                    FRESH_ATTEMPT_HN: 12, TIMING_ANOMALY: 36, NONCE_ANOMALY: 36,
                }))
                self.assertEqual((len(retained), len(replaced), len(introduced)), (36, 36, 36))
                self.assertTrue(retained <= c0 and retained <= c1)
                self.assertTrue(replaced <= c0 and replaced.isdisjoint(c1))
                self.assertTrue(introduced <= c1 and introduced.isdisjoint(c0))
                self.assertEqual(
                    {sample_id for sample_id in c0 if role_index[sample_id] in ANOMALY_ROLES},
                    {sample_id for sample_id in c1 if role_index[sample_id] in ANOMALY_ROLES},
                )
                self.assertEqual(
                    Counter(role_index[sample_id] for sample_id in introduced),
                    Counter({role: 12 for role in HN_ROLES}),
                )

    def test_c2_is_membership_alias_only(self):
        for item in self.manifest["repetitions"]:
            with self.subTest(repetition=item["repetition"]):
                self.assertEqual(item["C2"]["feature_profile"], "C2")
                self.assertEqual(item["C1"]["feature_profile"], "C1")
                self.assertEqual(
                    item["C2"]["training_trace_ids"], item["C1"]["training_trace_ids"]
                )
                self.assertEqual(
                    item["C2"]["common_test_trace_ids"], item["C1"]["common_test_trace_ids"]
                )

    def test_device_aware_selection_is_deterministic_and_rng_isolated(self):
        records = tuple(
            record for record in candidate_control_records(self.candidate)
            if record.role == ORDINARY
        )
        global_state = random.getstate()
        first = device_aware_select(records, 36, seed=42001, namespace="audit")
        second = device_aware_select(tuple(reversed(records)), 36, seed=42001, namespace="audit")
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), global_state)
        per_device = Counter(record.device_key for record in first)
        self.assertLessEqual(max(per_device.values()) - min(per_device.values()), 1)

    def test_builder_reads_control_metadata_not_observations_or_profiles(self):
        self.assertEqual(self.manifest["safety_boundary"], {
            "metadata_read": [
                "sample_id", "scientific_role", "detector_truth_for_integrity",
                "population_device_key",
            ],
            "o03_features_read": False,
            "device_profile_parameters_read": False,
            "validator_or_transformation_details_read": False,
            "classifier_information_read": False,
        })
        altered = deepcopy(self.candidate)
        for sample in altered["samples"]:
            sample["detector_observations"] = [{"ignored": "changed"}]
            sample["privileged_metadata"]["device_profile"] = {"ignored": "changed"}
        rebuilt = self.fresh_manifest(altered)
        self.assertEqual(rebuilt["repetitions"], self.manifest["repetitions"])
        self.assertEqual(
            rebuilt["device_grouped_sensitivity"],
            self.manifest["device_grouped_sensitivity"],
        )

    def test_grouped_sensitivity_is_device_disjoint_and_structurally_adequate(self):
        grouped = self.manifest["device_grouped_sensitivity"]
        accepted = grouped["accepted_split"]
        self.assertEqual(grouped["status"], "STRUCTURALLY_ADEQUATE")
        self.assertEqual(grouped["seeds_attempted"], [43001])
        self.assertEqual(accepted["seed"], 43001)
        self.assertEqual(len(accepted["train_device_ids"]), 48)
        self.assertEqual(len(accepted["test_device_ids"]), 16)
        self.assertTrue(
            set(accepted["train_device_ids"]).isdisjoint(accepted["test_device_ids"])
        )
        self.assertTrue(all(accepted["train_role_counts"][role] >= 30 for role in ROLES))
        self.assertTrue(all(accepted["test_role_counts"][role] >= 10 for role in ROLES))
        self.assertFalse(grouped["c0_c1_subsampling_constructed"])

    def test_manifest_file_fingerprint_is_stable(self):
        digest = hashlib.sha256(PHASE2_MANIFEST_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            digest, "f7b851266e720a25db45e5430cf7b09aa501fd84fb7a057aad2779d6d1d8aa64",
        )
        self.assertEqual(
            self.manifest["manifest_fingerprint"],
            "sha256:8e11d3a31dec7d9ad57fe59a6a658f58bfb7919e7bd744fb1600b594bcf0b8ed",
        )


if __name__ == "__main__":
    unittest.main()
