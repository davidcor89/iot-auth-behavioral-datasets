"""4D.3.1 role-independent valid-base opportunity infrastructure tests."""

from __future__ import annotations

import json
import random
import unittest
from dataclasses import FrozenInstanceError, replace

from simulator.config.opportunity_pool import OpportunityPoolConfiguration
from simulator.config.scientific_population import ScientificPopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import field_manifest
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.applicability import (
    APPLICABILITY_EVALUATOR_VERSION,
    APPLICABILITY_VOCABULARY_VERSION,
    ApplicabilityPredicate,
    evaluate_applicability,
    validate_evidence_references,
)
from simulator.role_integration.models import (
    TRACE_SECURITY_SNAPSHOT_VERSION,
    VALID_BASE_ENVELOPE_VERSION,
    BaseValidationResult,
    StructuralRealizationSummary,
    TraceSecuritySnapshot,
    ValidBaseTrace,
    canonical_json,
    stable_reference,
)
from simulator.role_integration.pool import (
    OPPORTUNITY_POOL_FINGERPRINT_VERSION,
    build_opportunity_pool,
    derive_future_role_reference,
    derive_future_transformation_reference,
    validate_pool_record,
)
from simulator.role_integration.transformation import (
    ROLE_TRANSFORMATION_INTERFACE_VERSION,
    TransformationFamily,
    TransformationOutcome,
    TransformationStatus,
)
from simulator.runner import build_run_population
from simulator.validation import VALIDATOR_VERSION, VALIDATION_SCHEMA_VERSION


def _scientific_population(seed: int = 41):
    previous = cfg.simulation.num_devices
    try:
        cfg.simulation.num_devices = 16
        random.seed(991)
        return build_run_population(ScientificPopulationConfiguration(
            device_count=16, population_seed=seed,
        ))
    finally:
        cfg.simulation.num_devices = previous


def _pool(*, population_seed: int = 41, pool_seed: int = 4301,
          role_seed: int = 4303, transformation_seed: int = 4304,
          repetitions: int = 3):
    return build_opportunity_pool(
        _scientific_population(population_seed),
        OpportunityPoolConfiguration(
            opportunities_per_device=repetitions,
            pool_seed=pool_seed,
            role_assignment_seed=role_seed,
            transformation_seed=transformation_seed,
        ),
    )


def _minimal_base(event_records: list[dict], *, lifecycle: str = "ACCESS",
                  recovery: str = "DIRECT", renewal: str = "NO_RENEWAL",
                  activity: int = 1) -> ValidBaseTrace:
    for ordinal, record in enumerate(event_records):
        record.setdefault("event_type", "discovery")
        record.setdefault("event_id", f"event-{ordinal + 1}")
        record.setdefault("timestamp", float(ordinal))
        record.setdefault("observed_timestamp", float(ordinal))
        record.setdefault("result", "success")
        record.setdefault("device_id", "runtime-device")
        record.setdefault("previous_state", "unregistered")
        record.setdefault("new_state", "unregistered")
        record.setdefault("delay_since_previous_event", 0.0 if not ordinal else 1.0)
    validation = BaseValidationResult(
        validation_reference="sha256:" + "4" * 64,
        validator_version=VALIDATOR_VERSION,
        validation_schema_version=VALIDATION_SCHEMA_VERSION,
        validation_status="validated_normal", build_gate_passed=True,
        validated_invariant_ids=(), anomaly_ground_truth_present=False,
        hard_negative_role_present=False, transformation_provenance_present=False,
    )
    base = ValidBaseTrace(
        envelope_version=VALID_BASE_ENVELOPE_VERSION,
        base_trace_reference="pending",
        base_construction_key="fixture-base",
        opportunity_randomness_reference="sha256:" + "1" * 64,
        population_reference="sha256:" + "2" * 64,
        device_id="runtime-device",
        population_device_key="population-device-000001",
        behavioral_profile_id="profile-fixture",
        persistent_context_reference="sha256:" + "3" * 64,
        trace_plan_json=canonical_json({"fixture": True}),
        planning_provenance_json=canonical_json({"fixture": True}),
        event_records_json=tuple(canonical_json(item) for item in event_records),
        session_context_json=canonical_json({"trace_id": "fixture"}),
        security_snapshot=TraceSecuritySnapshot(
            version=TRACE_SECURITY_SNAPSHOT_VERSION,
            authentication_attempt_references=(), protected_session_references=(),
            token_references=(), nonce_references=(), access_request_references=(),
            token_lineage=(), authentication_result="success",
            authenticated_identity_matches_device=True,
            token_validation_result="not_evaluated",
            authorization_decisions=(), resource_operation_outcomes=(),
            authenticated_context_active=True, token_context_active=False,
            protected_session_active=False,
        ),
        timing_provenance_json=canonical_json({"fixture": True}),
        structural_summary=StructuralRealizationSummary(
            lifecycle=lifecycle, recovery=recovery, renewal=renewal,
            activity_count=activity, timing_realization_present=True,
            population_device_key="population-device-000001",
            behavioral_profile_id="profile-fixture",
        ),
        validation=validation,
    )
    return replace(base, base_trace_reference=stable_reference(
        base.normalized_identity_record()
    ))


class ApplicabilityTests(unittest.TestCase):
    def test_vocabulary_is_complete_versioned_and_noncanonical(self):
        self.assertEqual(14, len(tuple(ApplicabilityPredicate)))
        self.assertTrue(APPLICABILITY_VOCABULARY_VERSION.startswith("4d3.1-"))
        self.assertTrue(APPLICABILITY_EVALUATOR_VERSION.startswith("4d3.1-"))

    def test_evidence_drives_positive_and_negative_results(self):
        events = [
            {"event_type": "challenge_sent", "auth_attempt_id": "a1", "nonce": "n1"},
            {"event_type": "response_sent", "auth_attempt_id": "a1", "nonce": "n1"},
            {"event_type": "authentication_success", "auth_attempt_id": "a1",
             "authenticated_identity": "runtime-device"},
            {"event_type": "token_issued", "auth_attempt_id": "a1", "token_id": "t1"},
            {"event_type": "session_opened", "token_id": "t1", "protected_session_id": "s1",
             "protected_session_active": True, "token_bound_identity": "runtime-device",
             "authenticated_identity": "runtime-device",
             "token_bound_protected_session_id": "s1"},
            {"event_type": "access_request", "token_id": "t1", "access_request_id": "r1"},
            {"event_type": "access_granted", "token_id": "t1", "access_request_id": "r1",
             "authorization_decision": "granted", "authorization_authoritative": True,
             "authorization_evidence_complete": True, "resource_operation_outcome": "success"},
        ]
        base = _minimal_base(events)
        results = {item.predicate_id: item for item in evaluate_applicability(base)}
        validate_evidence_references(base, results.values())
        self.assertTrue(results[ApplicabilityPredicate.SUCCESSFUL_AUTHENTICATION.value].applicable)
        self.assertTrue(results[ApplicabilityPredicate.PROTECTED_ACCESS.value].applicable)
        self.assertTrue(results[ApplicabilityPredicate.AUTHORITATIVE_GRANT.value].applicable)
        self.assertFalse(results[ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES.value].applicable)
        self.assertFalse(results[ApplicabilityPredicate.AUTHORITATIVE_DENIAL.value].applicable)
        self.assertFalse(results[ApplicabilityPredicate.SUCCESSFUL_TOKEN_REPLACEMENT.value].applicable)
        self.assertTrue(all(item.reason for item in results.values()))

    def test_replacement_and_two_accesses_require_actual_event_evidence(self):
        events = [
            {"event_type": "access_request", "token_id": "t1", "access_request_id": "r1"},
            {"event_type": "access_request", "token_id": "t1", "access_request_id": "r2"},
            {"event_type": "token_issued", "token_id": "t2", "supersedes_token_id": "t1"},
            {"event_type": "token_presented", "token_id": "t2"},
        ]
        base = _minimal_base(events, renewal="RENEWAL", activity=2)
        results = {item.predicate_id: item for item in evaluate_applicability(base)}
        for predicate in (
            ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES,
            ApplicabilityPredicate.SUCCESSFUL_TOKEN_REPLACEMENT,
            ApplicabilityPredicate.CURRENT_T1_USE_BEFORE_REPLACEMENT,
            ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT,
        ):
            self.assertTrue(results[predicate.value].applicable, predicate.value)


class OpportunityPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()

    def test_scientific_campaign_is_valid_role_free_and_structurally_diverse(self):
        pool = self.pool
        self.assertEqual(48, pool.diagnostics["attempted_bases"])
        self.assertEqual(48, len(pool.bases))
        self.assertEqual(0, len(pool.rejected))
        self.assertEqual(16, pool.diagnostics["devices_represented"])
        self.assertEqual({3}, set(pool.diagnostics["bases_per_device"].values()))
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, set(
            pool.diagnostics["structural_realizations"]["lifecycle"]
        ))
        self.assertIn("ONE_RETRY", pool.diagnostics["structural_realizations"]["recovery"])
        self.assertIn("RENEWAL", pool.diagnostics["structural_realizations"]["renewal"])
        self.assertTrue(all(base.validation.accepted for base in pool.bases))
        self.assertFalse(pool.diagnostics["role_assignment_present"])
        self.assertFalse(pool.diagnostics["transformation_present"])

    def test_targeted_realizations_have_evidence_based_applicability(self):
        pairs = list(zip(self.pool.bases, self.pool.applicability_matrix))
        by_summary = lambda predicate: next((pair for pair in pairs if predicate(pair[0])), None)
        fixtures = (
            by_summary(lambda b: b.structural_summary.lifecycle == "AUTH" and b.structural_summary.recovery == "DIRECT"),
            by_summary(lambda b: b.structural_summary.lifecycle == "AUTH" and b.structural_summary.recovery == "ONE_RETRY"),
            by_summary(lambda b: b.structural_summary.lifecycle == "SESSION"),
            by_summary(lambda b: b.structural_summary.lifecycle == "ACCESS" and b.structural_summary.activity_count == 1),
            by_summary(lambda b: b.structural_summary.lifecycle == "ACCESS" and b.structural_summary.activity_count >= 2),
            by_summary(lambda b: b.structural_summary.renewal == "RENEWAL"),
            by_summary(lambda b: b.structural_summary.lifecycle == "ACCESS" and b.structural_summary.renewal == "NO_RENEWAL"),
        )
        self.assertTrue(all(fixtures), "declared characterization lacks a required ordinary structure")
        for base, vector in fixtures:
            validate_evidence_references(base, vector)
        access_one = fixtures[3]
        access_many = fixtures[4]
        one = {item.predicate_id: item for item in access_one[1]}
        many = {item.predicate_id: item for item in access_many[1]}
        self.assertFalse(one[ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES.value].applicable)
        self.assertTrue(many[ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES.value].applicable)
        self.assertTrue(all(
            not {item.predicate_id: item for item in vector}[
                ApplicabilityPredicate.PROTECTED_ACCESS.value
            ].applicable
            for base, vector in pairs if base.structural_summary.lifecycle in {"AUTH", "SESSION"}
        ))

    def test_pool_roundtrip_evidence_and_fingerprint_validate(self):
        record = json.loads(json.dumps(self.pool.to_record()))
        validate_pool_record(record)
        self.assertTrue(self.pool.pool_fingerprint.startswith("sha256:"))
        self.assertEqual(OPPORTUNITY_POOL_FINGERPRINT_VERSION,
                         self.pool.fingerprint_payload()["fingerprint_version"])
        record["bases"][0]["event_records"][0]["result"] = "tampered"
        with self.assertRaisesRegex(ValueError, "normalized event evidence mismatch"):
            validate_pool_record(record)

    def test_reconstruction_and_seed_isolation(self):
        rebuilt = _pool()
        self.assertEqual(self.pool.pool_fingerprint, rebuilt.pool_fingerprint)
        self.assertEqual(
            [base.base_trace_reference for base in self.pool.bases],
            [base.base_trace_reference for base in rebuilt.bases],
        )
        role_changed = _pool(role_seed=999, transformation_seed=998)
        self.assertEqual(self.pool.pool_fingerprint, role_changed.pool_fingerprint)
        pool_changed = _pool(pool_seed=999)
        self.assertNotEqual(self.pool.pool_fingerprint, pool_changed.pool_fingerprint)
        self.assertEqual(
            self.pool.population_reference, pool_changed.population_reference,
        )
        population_changed = _pool(population_seed=42)
        self.assertNotEqual(self.pool.population_reference, population_changed.population_reference)

    def test_pool_builder_preserves_global_rng_state(self):
        population = _scientific_population()
        context_before = {
            key: (value.enrolled, value.paired, value.completed_trace_count)
            for key, value in population.semantic_contexts.items()
        }
        random.seed(771)
        before = random.getstate()
        first = build_opportunity_pool(population, OpportunityPoolConfiguration(
            opportunities_per_device=1,
        ))
        self.assertEqual(before, random.getstate())
        self.assertEqual(context_before, {
            key: (value.enrolled, value.paired, value.completed_trace_count)
            for key, value in population.semantic_contexts.items()
        })
        second = build_opportunity_pool(population, OpportunityPoolConfiguration(
            opportunities_per_device=1,
        ))
        self.assertEqual(first.pool_fingerprint, second.pool_fingerprint)

    def test_base_is_immutable_and_future_references_do_not_change_pool(self):
        base = self.pool.bases[0]
        before = self.pool.pool_fingerprint
        with self.assertRaises(FrozenInstanceError):
            base.base_trace_reference = "changed"
        role_ref = derive_future_role_reference(
            role_seed=4, pool_fingerprint=before, assignment_key="future-role-1",
        )
        transform_ref = derive_future_transformation_reference(
            transformation_seed=5, base_trace_reference=base.base_trace_reference,
            transformation_key="future-transform-1",
        )
        self.assertNotEqual(role_ref, transform_ref)
        self.assertEqual(before, self.pool.pool_fingerprint)
        outcome = TransformationOutcome(
            interface_version=ROLE_TRANSFORMATION_INTERFACE_VERSION,
            family=TransformationFamily.TIMING,
            transformation_id="dummy-interface-test-only",
            base_trace_reference=base.base_trace_reference,
            status=TransformationStatus.NOT_APPLICABLE,
            derived_event_records_json=(), transformation_provenance_json="{}",
            reason="interface test only; no transformation executed",
        )
        self.assertEqual(TransformationStatus.NOT_APPLICABLE, outcome.status)
        self.assertEqual(before, self.pool.pool_fingerprint)

    def test_privileged_fields_are_absent_from_detector_contract(self):
        manifest_text = canonical_json(field_manifest())
        for forbidden in (
            "base_trace_reference", "opportunity_pool", "applicability_vector",
            "behavioral_profile_id", "population_device_key", "transformation_id",
        ):
            self.assertNotIn(forbidden, manifest_text)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
