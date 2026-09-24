"""HN.1 infrastructure-only registry, qualification, gate, and boundary tests."""

import json
import random
import unittest
from dataclasses import replace

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import (
    FieldRole,
    dataset_capability_manifest,
    field_manifest,
    field_names_for_view,
    fields_for_view,
)
from simulator.data.synchronized_views import (
    build_synchronized_bundle,
    to_hard_negative_qualification_view,
)
from simulator.engines.event_engine import EventEngine
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.hard_negative import (
    CandidateStatus,
    ConstructionStatus,
    HARD_NEGATIVE_FAMILY_REGISTRY_VERSION,
    HARD_NEGATIVE_QUALIFICATION_CONTRACT_VERSION,
    HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION,
    HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
    HARD_NEGATIVE_SEMANTIC_CONTRACT_VERSION,
    HN_CHECKS,
    HN_FAMILIES,
    HN_VARIANTS,
    HardNegativeQualificationRecord,
    HardNegativeRunSummary,
    QualificationCheckDefinition,
    QualificationStatus,
    construction_yield,
    hard_negative_capability_report,
    qualification_yield,
    validate_hard_negative_registry,
    validate_hard_negative_structure,
)
from simulator.invariant_registry import INVARIANT_REGISTRY
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import validate_synchronized_dataset


EXPECTED_FAMILIES = {
    "HN-NONCE-FRESHNESS",
    "HN-TOKEN-CURRENTNESS",
    "HN-TIME-CAUSAL-MARGIN",
    "HN-ACCESS-ENFORCEMENT",
    "HN-AUTHZ-SEPARATION",
    "HN-IDENTITY-COHERENCE",
    "HN-BINDING-COHERENCE",
}
EXPECTED_VARIANTS = {
    "fresh_nonce_cross_attempt",
    "legitimate_same_attempt_nonce_recurrence",
    "repeated_current_token_use",
    "last_valid_use_before_replacement",
    "correct_replacement_token_use",
    "small_positive_causal_margin",
    "missing_auth_context_enforced",
    "invalid_credential_enforced",
    "inactive_session_enforced",
    "no_grant_enforced",
    "explicit_deny_enforced",
    "valid_context_legitimate_denial",
    "persistent_authenticated_identity_coherent",
    "token_identity_binding_coherent",
    "token_protected_context_binding_coherent",
}
EXPECTED_CHECKS = {
    "hn.v1.nonce.fresh_cross_attempt",
    "hn.v1.nonce.same_attempt_recurrence",
    "hn.v1.token.current_repeated_use",
    "hn.v1.token.last_use_before_replacement",
    "hn.v1.token.replacement_token_use",
    "hn.v1.time.positive_causal_margin",
    "hn.v1.access.missing_auth_enforced",
    "hn.v1.access.invalid_credential_enforced",
    "hn.v1.access.inactive_session_enforced",
    "hn.v1.access.no_grant_enforced",
    "hn.v1.access.explicit_deny_enforced",
    "hn.v1.authz.valid_context_legitimate_denial",
    "hn.v1.identity.persistent_authenticated_coherent",
    "hn.v1.binding.identity_coherent",
    "hn.v1.binding.protected_context_coherent",
}
BASE_DETECTOR_FIELDS = (
    "event_type",
    "observed_timestamp",
    "observed_delay_since_previous_event",
    "result",
    "failure_reason",
    "retry_count",
    "token_id",
    "nonce",
    "topic",
    "resource_id",
    "requested_action",
)


def qualified_record(**overrides):
    values = {
        "hard_negative_candidate_id": "hn-candidate-1",
        "trace_id": "trace-1",
        "scenario_id": "scenario-1",
        "device_id": "device-1",
        "candidate_status": CandidateStatus.REQUESTED,
        "primary_intended_family_id": "HN-NONCE-FRESHNESS",
        "intended_variant_id": "fresh_nonce_cross_attempt",
        "intended_boundary_invariant_ids": ("INV-REPLAY-NONCE-CROSS-ATTEMPT",),
        "construction_status": ConstructionStatus.APPLIED,
        "qualification_status": QualificationStatus.QUALIFIED,
        "qualified_family_ids": ("HN-NONCE-FRESHNESS",),
        "qualified_boundary_invariant_ids": ("INV-REPLAY-NONCE-CROSS-ATTEMPT",),
        "qualification_check_ids": ("hn.v1.nonce.fresh_cross_attempt",),
        "target_event_ids": ("target-event",),
        "evidence_event_ids": ("source-event", "target-event"),
        "normal_side_predicates": {
            "placeholder_fixture": {"satisfied": True, "evidence_event_ids": ["source-event"]},
        },
        "boundary_proximity_evidence": {
            "boundary_reached": {"satisfied": True, "evidence_event_ids": ["target-event"]},
        },
        "relevant_validation_check_ids": ("c1.7.nonce_reuse.within_trace_attempts",),
        "relevant_validation_outcomes": {
            "c1.7.nonce_reuse.within_trace_attempts": "NOT_APPLICABLE",
        },
        "qualification_reason": "explicit structural unit fixture",
        "qualifier_implementation_version": HARD_NEGATIVE_QUALIFIER_IMPLEMENTATION_VERSION,
        "trace_ground_truth_status": "validated_normal",
        "trace_observable_anomaly": False,
    }
    values.update(overrides)
    return HardNegativeQualificationRecord(**values)


def normal_pair(seed=3101):
    random.seed(seed)
    spec = ScenarioEngine(seed).normal()
    spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
    persistent = PersistentDeviceContext(device_id="hn-normal-device")
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="hn-gateway",
        auth_server_id="hn-auth",
        broker_id="hn-broker",
        start_time=80_000.0,
        persistent_context=persistent,
    ).execute(spec)


class HardNegativeRegistryTests(unittest.TestCase):
    def test_exactly_seven_families_and_approved_variants_are_registered(self):
        self.assertEqual(EXPECTED_FAMILIES, {item.family_id for item in HN_FAMILIES})
        self.assertEqual(EXPECTED_VARIANTS, {item.variant_id for item in HN_VARIANTS})
        self.assertEqual(EXPECTED_CHECKS, {item.check_id for item in HN_CHECKS})
        self.assertEqual((), validate_hard_negative_registry())

    def test_every_variant_has_one_family_existing_invariants_and_checks(self):
        for variant in HN_VARIANTS:
            self.assertIn(variant.family_id, EXPECTED_FAMILIES)
            self.assertTrue(variant.invariant_ids)
            self.assertTrue(set(variant.invariant_ids).issubset(INVARIANT_REGISTRY))
            self.assertTrue(variant.qualification_check_ids)
            self.assertTrue(set(variant.qualification_check_ids).issubset(EXPECTED_CHECKS))

    def test_integrity_checker_detects_duplicate_unknown_orphan_and_bad_mapping(self):
        duplicate = validate_hard_negative_registry(
            (*HN_FAMILIES, HN_FAMILIES[0]), HN_VARIANTS, HN_CHECKS
        )
        self.assertTrue(any("duplicate family" in item for item in duplicate))

        bad_variant = replace(
            HN_VARIANTS[0],
            family_id="HN-UNKNOWN",
            invariant_ids=("INV-UNKNOWN",),
            qualification_check_ids=("hn.unknown",),
        )
        errors = validate_hard_negative_registry(
            HN_FAMILIES, (bad_variant, *HN_VARIANTS[1:]), HN_CHECKS
        )
        self.assertTrue(any("unknown family" in item for item in errors))
        self.assertTrue(any("invalid family/variant mapping" in item for item in errors))

        orphan = QualificationCheckDefinition(
            "hn.v1.orphan", "HN-NONCE-FRESHNESS", "fresh_nonce_cross_attempt",
            ("c1.7.nonce_reuse.within_trace_attempts",),
        )
        self.assertTrue(any(
            "orphan qualification check" in item
            for item in validate_hard_negative_registry(HN_FAMILIES, HN_VARIANTS, (*HN_CHECKS, orphan))
        ))

    def test_capability_versions_and_deferred_boundaries_are_machine_readable(self):
        report = hard_negative_capability_report()
        self.assertEqual(HARD_NEGATIVE_SEMANTIC_CONTRACT_VERSION, report["hard_negative_semantic_contract_version"])
        self.assertEqual(HARD_NEGATIVE_FAMILY_REGISTRY_VERSION, report["hard_negative_family_registry_version"])
        self.assertEqual(HARD_NEGATIVE_QUALIFICATION_CONTRACT_VERSION, report["hard_negative_qualification_contract_version"])
        self.assertEqual(HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION, report["hard_negative_qualification_schema_version"])
        self.assertTrue(report["generation_available"])
        self.assertTrue(report["explicit_single_family_generation_available"])
        self.assertFalse(report["random_sampling_available"])
        self.assertTrue(report["semantic_qualification_available"])
        self.assertIn("cross_session_token_replay", report["deferred_or_unsupported_boundaries"])
        json.dumps(report)


class HardNegativeQualificationTests(unittest.TestCase):
    def test_status_vocabularies_preserve_all_required_distinctions(self):
        self.assertEqual(
            {"not_requested", "requested"},
            {item.value for item in CandidateStatus},
        )
        self.assertEqual(
            {"not_requested", "requested_not_applied", "applied", "partially_applied", "construction_error"},
            {item.value for item in ConstructionStatus},
        )
        self.assertEqual(
            {
                "qualified", "not_qualified_not_applied",
                "not_qualified_insufficient_evidence",
                "not_qualified_underdetermined",
                "not_qualified_relevant_unsupported",
                "not_qualified_invariant_violation",
                "not_qualified_boundary_not_reached", "qualification_error",
            },
            {item.value for item in QualificationStatus},
        )

    def test_record_round_trip_and_candidate_identity_are_independent(self):
        record = qualified_record(counterfactual_pair_id="pair-1")
        restored = HardNegativeQualificationRecord.from_dict(
            json.loads(json.dumps(record.to_dict()))
        )
        self.assertEqual(record, restored)
        self.assertNotEqual(record.hard_negative_candidate_id, record.trace_id)
        self.assertNotIn(record.hard_negative_candidate_id, record.qualification_check_ids)
        self.assertEqual((), record.structural_errors())

    def test_impossible_qualified_states_are_rejected(self):
        cases = (
            qualified_record(candidate_status=CandidateStatus.NOT_REQUESTED),
            qualified_record(construction_status=ConstructionStatus.REQUESTED_NOT_APPLIED),
            qualified_record(qualified_family_ids=()),
            qualified_record(qualification_check_ids=()),
            qualified_record(qualified_boundary_invariant_ids=("INV-UNKNOWN",)),
            qualified_record(primary_intended_family_id="HN-UNKNOWN"),
            qualified_record(intended_variant_id="unknown_variant"),
            qualified_record(qualification_check_ids=("hn.unknown",)),
        )
        for record in cases:
            with self.subTest(record=record):
                self.assertTrue(record.structural_errors())
                self.assertFalse(validate_hard_negative_structure((record,)).passed)

    def test_default_record_is_nonqualified_and_does_not_infer_normality(self):
        record = HardNegativeQualificationRecord(
            hard_negative_candidate_id="not-requested",
            trace_id="trace",
            scenario_id="scenario",
            device_id="device",
        )
        self.assertEqual(ConstructionStatus.NOT_REQUESTED, record.construction_status)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_NOT_APPLIED, record.qualification_status)
        self.assertEqual((), record.structural_errors())

    def test_structural_gate_rejects_duplicate_candidates_and_detector_leakage(self):
        record = qualified_record()
        duplicate = validate_hard_negative_structure((record, record))
        self.assertFalse(duplicate.passed)
        self.assertTrue(any("duplicate hard-negative candidate" in item for item in duplicate.errors))

        manifest = field_manifest()
        manifest["fields"].append({
            "field_name": "qualification_status",
            "export_view": "detector_observations",
            "role": "observable",
            "availability": "event_time",
        })
        leaked = validate_hard_negative_structure((), field_manifest=manifest)
        self.assertFalse(leaked.passed)
        self.assertTrue(any("detector manifest exposes" in item for item in leaked.errors))

    def test_structural_gate_rejects_malformed_and_nonserializable_records(self):
        malformed = validate_hard_negative_structure(({"not": "a typed record"},))
        self.assertFalse(malformed.passed)
        self.assertTrue(any("malformed qualification record" in item for item in malformed.errors))

        nonserializable = qualified_record(
            construction_parameters={"not_json": {"set-members-are-not-json"}},
        )
        serialization = validate_hard_negative_structure((nonserializable,))
        self.assertFalse(serialization.passed)
        self.assertTrue(any("qualification serialization error" in item for item in serialization.errors))

    def test_dedicated_surface_serializes_only_explicit_fixtures(self):
        self.assertEqual([], to_hard_negative_qualification_view([]))
        rows = to_hard_negative_qualification_view([qualified_record()])
        self.assertEqual("hn-candidate-1", rows[0]["hard_negative_candidate_id"])
        self.assertEqual("qualified", rows[0]["qualification_status"])


class HardNegativeBoundaryAndReportingTests(unittest.TestCase):
    def test_hn_fields_are_privileged_and_detector_schema_is_byte_for_byte_unchanged(self):
        self.assertEqual(BASE_DETECTOR_FIELDS, field_names_for_view("detector_observations"))
        hn_fields = fields_for_view("hard_negative_qualification")
        self.assertEqual(
            set(HardNegativeQualificationRecord.__dataclass_fields__),
            {item.field_name for item in hn_fields},
        )
        self.assertTrue(hn_fields)
        self.assertTrue(all(item.privileged for item in hn_fields))
        self.assertTrue(all(item.role not in {FieldRole.OBSERVABLE, FieldRole.DERIVED_OBSERVABLE} for item in hn_fields))

    def test_ordinary_generation_has_no_hn_record_and_anomaly_gt_is_unchanged(self):
        pair = normal_pair()
        provenance = build_generation_provenance(
            cfg, run_id="hn1-normal", generator_git_commit="hn1-test", generator_git_dirty=False
        )
        bundle = build_synchronized_bundle([pair], provenance=provenance)
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        self.assertNotIn("hard_negative_structural_gate", bundle["validation"])
        self.assertNotIn("hard_negative", bundle["ground_truth"])
        self.assertFalse(bundle["ground_truth"]["traces"][0]["observable_anomaly"])
        self.assertTrue(all(row["anomaly_label"] is None for row in bundle["ground_truth"]["events"]))
        self.assertFalse(hasattr(pair[1], "hard_negative_candidate_id"))

    def test_malformed_hn_record_fails_integrated_dataset_build_gate(self):
        pair = normal_pair(3102)
        provenance = build_generation_provenance(
            cfg, run_id="hn1-gate", generator_git_commit="hn1-test", generator_git_dirty=False
        )
        report = validate_synchronized_dataset(
            [pair],
            provenance=provenance,
            field_manifest=field_manifest(),
            dataset_manifest=dataset_capability_manifest(
                provenance, generation_profiles=(pair[1].generation_profile,)
            ),
            hard_negative_qualification_records=(
                qualified_record(construction_status=ConstructionStatus.NOT_REQUESTED),
            ),
        )
        self.assertFalse(report.build_gate_passed)
        self.assertFalse(report.hard_negative_structural_gate["passed"])
        self.assertIn("hard-negative structural gate failure", report.mandatory_failure_reasons)

    def test_run_summary_and_zero_safe_yields_use_only_explicit_fixtures(self):
        qualified = qualified_record()
        unsupported = replace(
            qualified,
            hard_negative_candidate_id="hn-candidate-2",
            trace_id="trace-2",
            qualification_status=QualificationStatus.NOT_QUALIFIED_RELEVANT_UNSUPPORTED,
            qualified_family_ids=(),
            qualified_boundary_invariant_ids=(),
            qualification_check_ids=(),
            target_event_ids=(),
            evidence_event_ids=(),
            qualification_reason="relevant evidence unsupported",
        )
        summary = HardNegativeRunSummary.from_records(
            total_traces=4,
            ordinary_normal_traces=2,
            records=(qualified, unsupported),
            candidate_anomaly_collisions=0,
        )
        self.assertEqual(2, summary.hn_intended_traces)
        self.assertEqual(2, summary.hn_construction_applied)
        self.assertEqual(1, summary.hn_qualified)
        self.assertEqual(1, summary.hn_not_qualified)
        self.assertEqual(1, summary.relevant_unsupported_cases)
        self.assertEqual(1.0, construction_yield(2, 2))
        self.assertEqual(0.5, qualification_yield(1, 2))
        self.assertIsNone(construction_yield(0, 0))
        self.assertIsNone(qualification_yield(0, 0))
        json.dumps(summary.to_dict())

    def test_provenance_and_manifest_expose_hn_versions_without_release_rewrite(self):
        provenance = build_generation_provenance(
            cfg, run_id="hn1-provenance", generator_git_commit="hn1-test", generator_git_dirty=False
        )
        manifest = dataset_capability_manifest(
            provenance, generation_profiles=(SYNCHRONIZED_GENERATION_PROFILE,)
        )
        self.assertEqual("sync-v1.1.0", provenance.generator_software_version)
        self.assertEqual("sync-v1.1.0-dataset-interface", provenance.dataset_interface_version)
        self.assertEqual(
            HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION,
            manifest["hard_negative_qualification_surface"]["schema_version"],
        )
        self.assertEqual(
            EXPECTED_FAMILIES,
            set(provenance.hard_negative_capability_contract["supported_family_ids"]),
        )
        self.assertTrue(
            provenance.hard_negative_capability_contract["semantic_qualification_available"]
        )


if __name__ == "__main__":
    unittest.main()
