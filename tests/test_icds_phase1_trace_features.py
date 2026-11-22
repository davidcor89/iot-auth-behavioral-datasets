"""Phase-1 safeguards for deterministic O03-only trace features."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ml.evaluation.trace_features import (
    C1_FEATURE_NAMES,
    C2_FEATURE_NAMES,
    FEATURE_DEFINITIONS,
    O03_FIELDS,
    TraceFeatureContractError,
    audit_candidate,
    extract_candidate_feature_rows,
    extract_trace_features,
    feature_specification,
)
from simulator.data.dataset_contract import field_names_for_view


CANDIDATE_PATH = Path("artifacts/icds-timing-nonce-candidate/scientific-candidate.json")
SPEC_PATH = Path("IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json")


def observation(
    event_type: str,
    timestamp: float,
    *,
    nonce: str | None = None,
    result: str = "success",
    retry_count: int = 0,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "observed_timestamp": timestamp,
        "observed_delay_since_previous_event": 0.0,
        "result": result,
        "failure_reason": None,
        "retry_count": retry_count,
        "token_id": None,
        "nonce": nonce,
        "topic": None,
        "resource_id": None,
        "requested_action": None,
    }


def trace_with_two_attempts(*, repeated_nonce: bool = False) -> list[dict[str, object]]:
    first = "nonce-a"
    second = first if repeated_nonce else "nonce-b"
    rows = [
        observation("authentication_request", 1.0),
        observation("challenge_sent", 2.0, nonce=first),
        observation("nonce_received", 3.0, nonce=first),
        observation("response_sent", 3.5, nonce=first),
        observation("authentication_failure", 4.0, result="failure"),
        observation("retry", 5.0, retry_count=1),
        # Deliberately no second authentication_request: this is the frozen
        # observable retry grammar whose boundary is challenge_sent.
        observation("challenge_sent", 6.0, nonce=second, retry_count=1),
        observation("nonce_received", 7.0, nonce=second, retry_count=1),
        observation("response_sent", 6.75, nonce=second, retry_count=1),
        observation("authentication_success", 8.0, retry_count=1),
    ]
    for index in range(1, len(rows)):
        rows[index]["observed_delay_since_previous_event"] = (
            float(rows[index]["observed_timestamp"])
            - float(rows[index - 1]["observed_timestamp"])
        )
    return rows


class FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    def test_specification_is_checked_in_and_complete(self):
        self.assertEqual(O03_FIELDS, field_names_for_view("detector_observations"))
        self.assertEqual(
            feature_specification(),
            json.loads(SPEC_PATH.read_text(encoding="utf-8")),
        )
        self.assertEqual(90, len(C1_FEATURE_NAMES))
        self.assertEqual(87, len(C2_FEATURE_NAMES))
        self.assertEqual(set(C1_FEATURE_NAMES), {item.name for item in FEATURE_DEFINITIONS})
        required = {
            "name", "description", "source_o03_fields", "transformation",
            "output_type", "temporal_scope", "complete_trace_required",
            "uses_future_information_relative_to_event_time", "transformation_kind",
            "feature_class", "feature_group", "predicate_equivalent_direct_rule_proxy",
            "c1_included", "c2_included", "privileged_information_dependency",
            "assumptions", "missing_non_applicable_handling",
        }
        for record in feature_specification()["features"]:
            self.assertEqual(required, set(record))
            self.assertFalse(record["privileged_information_dependency"])
            self.assertEqual("deterministic_per_trace", record["transformation_kind"])

    def test_detector_input_is_exactly_o03_and_privileged_fields_are_rejected(self):
        rows = trace_with_two_attempts()
        rows[0]["auth_attempt_id"] = "privileged-attempt"
        with self.assertRaisesRegex(TraceFeatureContractError, "extra=.*auth_attempt_id"):
            extract_trace_features(rows)
        self.assertEqual(set(O03_FIELDS), set(trace_with_two_attempts()[0]))

    def test_gt_and_provenance_changes_do_not_change_feature_vector(self):
        sample = self.candidate["samples"][0]
        expected = extract_trace_features(sample["detector_observations"])
        altered = copy.deepcopy(sample)
        altered["sample_id"] = "different-privileged-join-id"
        altered["privileged_metadata"] = {
            "detector_truth": "changed",
            "role": "changed",
            "validator": "changed",
            "auth_attempt_id": "changed",
            "device_id": "changed",
        }
        self.assertEqual(expected, extract_trace_features(altered["detector_observations"]))

    def test_timing_relation_uses_signed_o03_margin_only(self):
        rows = trace_with_two_attempts()
        features = extract_trace_features(rows)
        self.assertEqual(2, features["semantic_timing_relation_count"])
        self.assertAlmostEqual(-0.25, features["semantic_timing_min_signed_causal_margin"])
        rows[0]["result"] = "pending"
        self.assertAlmostEqual(
            -0.25,
            extract_trace_features(rows)["semantic_timing_min_signed_causal_margin"],
        )

    def test_attempt_reconstruction_uses_observable_challenge_boundaries(self):
        fresh = extract_trace_features(trace_with_two_attempts(repeated_nonce=False))
        replay = extract_trace_features(trace_with_two_attempts(repeated_nonce=True))
        self.assertEqual(0, fresh["semantic_nonce_cross_attempt_value_count"])
        self.assertEqual(1, fresh["semantic_nonce_max_attempt_sharing_count"])
        self.assertEqual(1, replay["semantic_nonce_cross_attempt_value_count"])
        self.assertEqual(2, replay["semantic_nonce_max_attempt_sharing_count"])

    def test_generic_nonce_structure_does_not_reconstruct_attempts(self):
        features = extract_trace_features(trace_with_two_attempts(repeated_nonce=False))
        self.assertEqual(6, features["nonce_bearing_observation_count"])
        self.assertEqual(2, features["nonce_distinct_value_count"])
        self.assertEqual(2, features["nonce_repeated_value_count"])
        self.assertEqual(3, features["nonce_max_multiplicity"])
        self.assertEqual(4, features["nonce_repeated_observation_count"])

    def test_c2_removes_only_direct_relation_predicates(self):
        excluded = set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES)
        self.assertEqual({
            "semantic_timing_min_signed_causal_margin",
            "semantic_nonce_cross_attempt_value_count",
            "semantic_nonce_max_attempt_sharing_count",
        }, excluded)
        self.assertTrue(set(C2_FEATURE_NAMES) < set(C1_FEATURE_NAMES))
        for retained in (
            "observed_inter_event_delay_mean",
            "nonce_repeated_value_count",
            "semantic_timing_relation_count",
        ):
            self.assertIn(retained, C2_FEATURE_NAMES)

    def test_determinism_and_dataset_trace_order_independence(self):
        first = extract_candidate_feature_rows(self.candidate)
        second = extract_candidate_feature_rows(self.candidate)
        self.assertEqual(first, second)
        reversed_candidate = {
            **self.candidate,
            "samples": list(reversed(self.candidate["samples"])),
        }
        reversed_rows = extract_candidate_feature_rows(reversed_candidate)
        self.assertEqual(
            {item["sample_reference"]: item["features"] for item in first},
            {item["sample_reference"]: item["features"] for item in reversed_rows},
        )

    def test_frozen_candidate_structural_audit(self):
        audit = audit_candidate(self.candidate)
        self.assertEqual(336, audit["traces_processed"])
        self.assertEqual(0, audit["extraction_failures"])
        self.assertEqual({"C1": 90, "C2": 87}, audit["feature_counts"])
        self.assertEqual(336, audit["observable_grammar_support"]["traces_with_reconstructed_timing_relation"])
        self.assertEqual(336, audit["observable_grammar_support"]["traces_with_observable_challenge_attempt"])
        self.assertTrue(audit["determinism"]["repeat_extraction_identical"])
        self.assertFalse(audit["dataset_level_fitting_performed"])
        self.assertFalse(audit["labels_or_privileged_metadata_read_by_extractor"])


if __name__ == "__main__":
    unittest.main()
