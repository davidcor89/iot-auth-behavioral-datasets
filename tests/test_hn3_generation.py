"""HN.3 tests exercise the opt-in production construction and HN.2 gate."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.data.synchronized_views import SynchronizedOutputViews, build_synchronized_bundle
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.hard_negative import (
    ConstructionStatus, HN_VARIANTS, QualificationStatus,
    hard_negative_capability_report, validate_hard_negative_structure,
)
from simulator.hard_negative_generation import HardNegativeRequest, generate_hard_negative
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.event_model import EventType


def request(variant, *, candidate_id=None, boundary=None):
    return HardNegativeRequest(
        family_id=variant.family_id, variant_id=variant.variant_id,
        candidate_id=candidate_id or f"hn3-{variant.variant_id}",
        construction_parameters={"campaign": "unit_test"},
        boundary_parameters=boundary if boundary is not None else (
            {"epsilon_hn": 0.01} if variant.variant_id == "small_positive_causal_margin" else {}
        ),
    )


class HardNegativeGenerationTests(unittest.TestCase):
    def test_all_registered_variants_use_production_api_and_hn2(self):
        self.assertEqual(15, len(HN_VARIANTS))
        for variant in HN_VARIANTS:
            with self.subTest(variant=variant.variant_id):
                result = generate_hard_negative(
                    request(variant), device_id=f"device-{variant.variant_id}",
                    seed=99, start_time=1000.0,
                )
                candidate = result.candidate
                self.assertEqual(variant.family_id, candidate.primary_intended_family_id)
                self.assertEqual(variant.variant_id, candidate.intended_variant_id)
                self.assertEqual(variant.invariant_ids, candidate.intended_boundary_invariant_ids)
                self.assertEqual(result.context.trace_id, candidate.trace_id)
                self.assertEqual(result.context.scenario_id, candidate.scenario_id)
                self.assertEqual("validated_normal", result.validation.trace_ground_truth_statuses[candidate.trace_id])
                self.assertEqual((), result.validation.trace_validated_invariant_ids[candidate.trace_id])
                self.assertFalse(result.context.observable_anomaly)
                self.assertTrue(result.gated_validation.build_gate_passed)
                self.assertIsNone(result.generator_defect)
                self.assertEqual(ConstructionStatus.APPLIED, candidate.construction_status)
                self.assertEqual(QualificationStatus.QUALIFIED, candidate.qualification_status)
                self.assertEqual((variant.family_id,), candidate.qualified_family_ids)
                self.assertTrue(candidate.target_event_ids)
                self.assertTrue(candidate.evidence_event_ids)
                self.assertTrue(candidate.normal_side_predicates)
                self.assertEqual("validated_normal", candidate.trace_ground_truth_status)

    def test_replacement_is_not_fabricated_for_inapplicable_lifecycle(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "correct_replacement_token_use")
        normal = ScenarioEngine(50).normal()
        normal.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
        result = generate_hard_negative(request(variant), device_id="no-renewal", seed=50,
                                        start_time=1000.0, scenario=normal)
        self.assertEqual(ConstructionStatus.REQUESTED_NOT_APPLIED, result.candidate.construction_status)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, result.candidate.qualification_status)
        self.assertTrue(result.gated_validation.build_gate_passed)

    def test_invalid_mapping_and_missing_protected_request(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "no_grant_enforced")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            generate_hard_negative(replace(request(variant), family_id="HN-NONCE-FRESHNESS"),
                                   device_id="invalid", seed=1, start_time=1000.0)
        normal = ScenarioEngine(51).normal()
        normal.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
        normal.normal_steps = normal.normal_steps[:normal.normal_steps.index(EventType.ACCESS_REQUEST)]
        result = generate_hard_negative(request(variant), device_id="no-access", seed=51,
                                        start_time=1000.0, scenario=normal)
        self.assertEqual(ConstructionStatus.REQUESTED_NOT_APPLIED, result.candidate.construction_status)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, result.candidate.qualification_status)

    def test_invalid_epsilon_is_construction_error_and_gate_failure(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "small_positive_causal_margin")
        result = generate_hard_negative(request(variant, boundary={"epsilon_hn": 0}),
                                        device_id="bad-time", seed=52, start_time=1000.0)
        self.assertEqual(ConstructionStatus.CONSTRUCTION_ERROR, result.candidate.construction_status)
        self.assertEqual(QualificationStatus.QUALIFICATION_ERROR, result.candidate.qualification_status)
        self.assertFalse(result.gated_validation.build_gate_passed)

    def test_positive_epsilon_is_explicit_and_changes_only_observed_margin(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "small_positive_causal_margin")
        margins = []
        for epsilon in (0.002, 0.01):
            result = generate_hard_negative(
                request(variant, candidate_id=f"epsilon-{epsilon}", boundary={"epsilon_hn": epsilon}),
                device_id=f"epsilon-device-{epsilon}", seed=52, start_time=1000.0,
            )
            self.assertEqual(QualificationStatus.QUALIFIED, result.candidate.qualification_status)
            observed = [event for event in result.events if event.event_type in {
                EventType.NONCE_RECEIVED, EventType.RESPONSE_SENT,
            }]
            margin = observed[1].observed_timestamp - observed[0].observed_timestamp
            self.assertGreater(margin, 0)
            self.assertLessEqual(margin, epsilon)
            self.assertEqual(epsilon, result.candidate.boundary_parameters["epsilon_hn"])
            self.assertGreater(observed[1].timestamp, observed[0].timestamp)
            margins.append(margin)
        self.assertNotEqual(*margins)

    def test_metadata_tampering_does_not_change_qualification(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "fresh_nonce_cross_attempt")
        result = generate_hard_negative(request(variant), device_id="tamper", seed=53,
                                        start_time=1000.0)
        tampered = replace(result.candidate, construction_parameters={"false_claim": "same nonce"},
                           qualification_status=QualificationStatus.NOT_QUALIFIED_NOT_APPLIED)
        requalified = qualify_hard_negative(result.events, result.context, tampered, result.validation)
        self.assertEqual(QualificationStatus.QUALIFIED, requalified.qualification_status)

    def test_candidate_anomaly_collision_keeps_positive_gt(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "fresh_nonce_cross_attempt")
        original_execute = EventEngine.execute

        def colliding_execute(engine, _normal_spec):
            anomaly = ScenarioEngine(55).synchronized_anomaly("nonce_reuse")
            anomaly.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
            return original_execute(engine, anomaly)

        with patch.object(EventEngine, "execute", colliding_execute):
            result = generate_hard_negative(request(variant), device_id="collision", seed=55,
                                            start_time=1000.0)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION,
                         result.candidate.qualification_status)
        self.assertTrue(result.context.observable_anomaly)
        self.assertTrue(result.validation.trace_validated_invariant_ids[result.context.trace_id])
        self.assertIsNotNone(result.generator_defect)

    def test_privileged_export_is_opt_in_and_keeps_failed_attempt(self):
        variant = next(v for v in HN_VARIANTS if v.variant_id == "small_positive_causal_margin")
        result = generate_hard_negative(request(variant, boundary={"epsilon_hn": 0}),
                                        device_id="export", seed=56, start_time=1000.0)
        sequences = [(list(result.events), result.context)]
        ordinary = build_synchronized_bundle(sequences, provenance=result.provenance)
        explicit = build_synchronized_bundle(
            sequences, provenance=result.provenance,
            hard_negative_qualification_records=[result.candidate],
        )
        self.assertNotIn("hard_negative_qualification", ordinary)
        self.assertEqual(ordinary["detector_observations"], explicit["detector_observations"])
        self.assertEqual(ordinary["ground_truth"], explicit["ground_truth"])
        self.assertFalse(any(
            key.startswith("hard_negative") or key in {"epsilon_hn", "qualification_status"}
            for row in explicit["detector_observations"] for key in row
        ))
        self.assertEqual(QualificationStatus.QUALIFICATION_ERROR.value,
                         explicit["hard_negative_qualification"][0]["qualification_status"])
        self.assertFalse(explicit["validation"]["build_gate"]["passed"])
        with TemporaryDirectory() as tmp:
            paths = SynchronizedOutputViews(tmp).save(
                sequences, provenance=result.provenance,
                hard_negative_qualification_records=[result.candidate],
            )
            data = json.loads(Path(paths["hard_negative_qualification"]).read_text())
            self.assertEqual(result.candidate.hard_negative_candidate_id,
                             data[0]["hard_negative_candidate_id"])

    def test_capability_declares_explicit_only_and_duplicate_ids_fail_gate(self):
        capability = hard_negative_capability_report()
        self.assertTrue(capability["explicit_single_family_generation_available"])
        self.assertFalse(capability["random_sampling_available"])
        self.assertFalse(capability["multi_boundary_generation_available"])
        variant = next(v for v in HN_VARIANTS if v.variant_id == "fresh_nonce_cross_attempt")
        result = generate_hard_negative(request(variant), device_id="dup", seed=57,
                                        start_time=1000.0)
        gate = validate_hard_negative_structure([result.candidate, result.candidate])
        self.assertFalse(gate.passed)


if __name__ == "__main__":
    unittest.main()
