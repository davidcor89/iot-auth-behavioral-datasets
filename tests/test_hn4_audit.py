"""HN.4 falsification checks over frozen HN.2/HN.3R mechanisms."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.hn4_stress_audit import run_campaign
from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest, field_names_for_view
from simulator.data.synchronized_views import SynchronizedOutputViews
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import NORMAL_FLOW, ScenarioEngine
from simulator.event_model import EventType
from simulator.hard_negative import (
    ConstructionStatus, HN_FAMILIES, HN_VARIANTS, HN_VARIANT_BY_ID,
    QualificationStatus, hard_negative_capability_report,
    validate_hard_negative_structure,
)
from simulator.hard_negative_generation import HardNegativeRequest, generate_hard_negative
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.provenance import build_generation_provenance
from simulator.config.settings import cfg
from simulator.security_context import PersistentDeviceContext
from simulator.validation import ValidationResult, validate_synchronized_dataset
from tests.test_hn2_qualification import positive_fixture


def candidate(variant_id: str, seed: int = 6101):
    variant = HN_VARIANT_BY_ID[variant_id]
    boundary = {"epsilon_hn": 0.01} if variant_id == "small_positive_causal_margin" else {}
    return generate_hard_negative(
        HardNegativeRequest(variant.family_id, variant_id, f"hn4-{variant_id}-{seed}", {}, boundary),
        device_id=f"d{seed:04d}-hn4", seed=seed, start_time=1000.0,
    )


def validate(events, context, provenance, records=()):
    return validate_synchronized_dataset(
        [(events, context)], provenance=provenance,
        field_manifest=field_manifest(),
        dataset_manifest=dataset_capability_manifest(
            provenance, generation_profiles=(context.generation_profile,),
        ),
        hard_negative_qualification_records=records,
    )


class HN4AdversarialAuditTests(unittest.TestCase):
    def test_release_capability_is_bounded_and_machine_readable(self):
        capability = hard_negative_capability_report()
        self.assertEqual("hn-v1.0.0", capability["hard_negative_release_version"])
        self.assertEqual(7, capability["registered_family_count"])
        self.assertEqual(15, capability["registered_variant_count"])
        self.assertEqual(15, capability["deterministic_qualification_check_count"])
        self.assertEqual(15, capability["explicit_generation_mechanism_count"])
        self.assertEqual([], capability["generation_blocked_variant_ids"])
        self.assertFalse(capability["random_sampling_available"])
        self.assertFalse(capability["multi_boundary_generation_available"])
        self.assertFalse(capability["scientific_sampling_prevalence_frozen"])
        self.assertFalse(capability["controlled_device_profile_variability_available"])
        self.assertFalse(capability["empirical_detector_difficulty_established"])

    def test_final_build_gate_requalifies_stale_or_fabricated_claims(self):
        result = candidate("fresh_nonce_cross_attempt")
        events = list(result.events)
        target = next(event for event in events if event.event_id == result.candidate.target_event_ids[0])
        target.nonce = None
        report = validate(events, result.context, result.provenance, (result.candidate,))
        self.assertFalse(report.build_gate_passed)
        self.assertIn("hard-negative semantic requalification failure", report.mandatory_failure_reasons)
        self.assertFalse(report.hard_negative_structural_gate["passed"])
        streamed = validate_synchronized_dataset(
            iter(((events, result.context),)), provenance=result.provenance,
            field_manifest=field_manifest(),
            dataset_manifest=dataset_capability_manifest(
                result.provenance, generation_profiles=(result.context.generation_profile,),
            ),
            hard_negative_qualification_records=iter((result.candidate,)),
        )
        self.assertFalse(streamed.build_gate_passed)

    def test_all_fifteen_requalify_deterministically_and_do_not_switch_family(self):
        for index, variant in enumerate(HN_VARIANTS):
            with self.subTest(variant=variant.variant_id):
                result = candidate(variant.variant_id, 6200 + index)
                first = qualify_hard_negative(result.events, result.context, result.candidate, result.validation)
                second = qualify_hard_negative(result.events, result.context, result.candidate, result.validation)
                self.assertEqual(first, second)
                self.assertEqual(result.candidate, first)
                self.assertEqual((variant.family_id,), first.qualified_family_ids)

    def test_small_mixed_campaign_is_semantically_reproducible(self):
        options = dict(per_variant=2, ordinary_count=20, per_anomaly=1,
                       intent_only_count=7, malformed_count=5, device_count=4)
        first = run_campaign(seed=6301, **options)
        repeat = run_campaign(seed=6301, **options)
        different = run_campaign(seed=6302, **options)
        self.assertEqual(first["semantic_sha256"], repeat["semantic_sha256"])
        self.assertEqual(first["config_hash"], repeat["config_hash"])
        self.assertEqual(first["summary"], repeat["summary"])
        self.assertNotEqual(first["semantic_sha256"], different["semantic_sha256"])
        self.assertEqual(30, different["summary"]["hn_qualified"])
        self.assertEqual(7, first["summary"]["qualification_status_counts"]["not_qualified_not_applied"])
        self.assertEqual(5, first["malformed_gate_failures"])
        self.assertGreater(first["time_margin_unique_rounded_6"], 1)

    def test_lifecycle_hn_anomaly_and_later_normal_reuse_one_persistent_device(self):
        provenance = build_generation_provenance(cfg, run_id="hn4-recovery")
        for index, variant_id in enumerate((
            "inactive_session_enforced", "correct_replacement_token_use",
            "fresh_nonce_cross_attempt",
        )):
            with self.subTest(variant=variant_id):
                device = PersistentDeviceContext(device_id=f"hn4-recovery-{index}")
                variant = HN_VARIANT_BY_ID[variant_id]
                first = generate_hard_negative(
                    HardNegativeRequest(variant.family_id, variant_id, f"recovery-first-{index}", {}, {}),
                    device_id=device.device_id, seed=7200 + index, start_time=1000.0,
                    persistent_context=device, provenance=provenance,
                )
                self.assertEqual(QualificationStatus.QUALIFIED, first.candidate.qualification_status)
                with patch.dict(STEALTH_FRACTION, {"nonce_reuse": 0.0}):
                    spec = ScenarioEngine(7300 + index).synchronized_anomaly("nonce_reuse")
                    anomaly_events, anomaly_context = EventEngine(
                        device_id=device.device_id, gateway_id="gateway", auth_server_id="auth_server",
                        broker_id="broker", start_time=2000.0, persistent_context=device,
                    ).execute(spec)
                later = generate_hard_negative(
                    HardNegativeRequest(variant.family_id, variant_id, f"recovery-later-{index}", {}, {}),
                    device_id=device.device_id, seed=7400 + index, start_time=3000.0,
                    persistent_context=device, provenance=provenance,
                )
                self.assertEqual(QualificationStatus.QUALIFIED, later.candidate.qualification_status)
                normal = ScenarioEngine(7500 + index).normal()
                normal.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
                normal.normal_steps = list(NORMAL_FLOW)
                normal_events, normal_context = EventEngine(
                    device_id=device.device_id, gateway_id="gateway", auth_server_id="auth_server",
                    broker_id="broker", start_time=4000.0, persistent_context=device,
                ).execute(normal)
                report = validate_synchronized_dataset(
                    [(list(first.events), first.context), (anomaly_events, anomaly_context),
                     (list(later.events), later.context), (normal_events, normal_context)],
                    provenance=provenance, field_manifest=field_manifest(),
                    dataset_manifest=dataset_capability_manifest(
                        provenance, generation_profiles=(SYNCHRONIZED_GENERATION_PROFILE,),
                    ),
                    hard_negative_qualification_records=(first.candidate, later.candidate),
                )
                self.assertTrue(report.build_gate_passed)
                self.assertEqual("validated_positive", report.trace_ground_truth_statuses[anomaly_context.trace_id])
                self.assertEqual("validated_normal", report.trace_ground_truth_statuses[normal_context.trace_id])
                self.assertTrue(device.enrolled)
                self.assertEqual(4, device.completed_trace_count)

    def test_intent_metadata_and_reference_tampering_cannot_create_truth(self):
        for index, family in enumerate(HN_FAMILIES):
            variant_id = family.approved_variant_ids[0]
            result = candidate(variant_id, 6400 + index)
            intent = replace(
                result.candidate, construction_status=ConstructionStatus.REQUESTED_NOT_APPLIED,
                qualification_status=QualificationStatus.NOT_QUALIFIED_NOT_APPLIED,
                qualified_family_ids=(), qualified_boundary_invariant_ids=(),
                qualification_check_ids=(), normal_side_predicates={},
                boundary_proximity_evidence={}, relevant_validation_check_ids=(),
                relevant_validation_outcomes={}, qualifier_implementation_version=None,
                trace_ground_truth_status=None, trace_observable_anomaly=None,
            )
            self.assertEqual(QualificationStatus.NOT_QUALIFIED_NOT_APPLIED,
                             qualify_hard_negative(result.events, result.context, intent,
                                                   result.validation).qualification_status)
            false_metadata = replace(result.candidate, construction_parameters={"claims_normal": False})
            self.assertEqual(QualificationStatus.QUALIFIED,
                             qualify_hard_negative(result.events, result.context, false_metadata,
                                                   result.validation).qualification_status)
            for tampered in (
                replace(result.candidate, target_event_ids=("missing-event",)),
                replace(result.candidate, evidence_event_ids=("missing-event",)),
                replace(result.candidate, trace_id="other-trace"),
                replace(result.candidate, scenario_id="other-scenario"),
                replace(result.candidate, device_id="other-device"),
            ):
                with self.subTest(family=family.family_id, tamper=tampered):
                    requalified = qualify_hard_negative(result.events, result.context,
                                                        tampered, result.validation)
                    self.assertNotEqual(QualificationStatus.QUALIFIED,
                                        requalified.qualification_status)
                    self.assertFalse(validate(list(result.events), result.context,
                                              result.provenance, (tampered,)).build_gate_passed)

    def test_registry_and_duplicate_candidate_tampering_fail_gate(self):
        result = candidate("fresh_nonce_cross_attempt", 6501)
        record = result.candidate
        other = HN_VARIANT_BY_ID["small_positive_causal_margin"]
        for tampered in (
            replace(record, primary_intended_family_id="unknown-family"),
            replace(record, intended_variant_id="unknown-variant"),
            replace(record, intended_variant_id=other.variant_id),
            replace(record, qualification_check_ids=("unknown-check",)),
            replace(record, intended_boundary_invariant_ids=("unknown-invariant",)),
            replace(record, qualified_boundary_invariant_ids=other.invariant_ids),
            replace(record, target_event_ids=record.target_event_ids * 2),
            replace(record, evidence_event_ids=record.evidence_event_ids * 2),
        ):
            with self.subTest(tamper=tampered):
                self.assertFalse(validate(list(result.events), result.context,
                                          result.provenance, (tampered,)).build_gate_passed)
        self.assertFalse(validate(list(result.events), result.context,
                                  result.provenance, (record, record)).build_gate_passed)
        manifest = field_manifest()
        manifest["fields"].append({
            "field_name": "hard_negative_family_hint",
            "export_view": "detector_observations",
        })
        self.assertFalse(validate_hard_negative_structure((record,), field_manifest=manifest).passed)

    def test_each_family_anomalous_counterpart_rejects_hn(self):
        counterparts = {
            "HN-NONCE-FRESHNESS": ("fresh_nonce_cross_attempt", "nonce_reuse"),
            "HN-TOKEN-CURRENTNESS": ("correct_replacement_token_use", "token_replay_after_replacement"),
            "HN-TIME-CAUSAL-MARGIN": ("small_positive_causal_margin", "timestamp_inconsistency"),
            "HN-ACCESS-ENFORCEMENT": ("missing_auth_context_enforced", "access_success_without_auth_context"),
            "HN-AUTHZ-SEPARATION": ("valid_context_legitimate_denial", "access_success_despite_explicit_deny"),
            "HN-IDENTITY-COHERENCE": ("persistent_authenticated_identity_coherent", "authenticated_identity_mismatch_persistent_device"),
            "HN-BINDING-COHERENCE": ("token_identity_binding_coherent", "token_bound_identity_mismatch_authenticated_context"),
        }
        for index, (family, (variant_id, anomaly_name)) in enumerate(counterparts.items()):
            with self.subTest(family=family):
                source = candidate(variant_id, 6600 + index)
                with patch.dict(STEALTH_FRACTION, {anomaly_name: 0.0}):
                    device = PersistentDeviceContext(device_id=f"d{index:04d}-collision")
                    spec = ScenarioEngine(6700 + index).synchronized_anomaly(anomaly_name)
                    events, context = EventEngine(
                        device_id=device.device_id, gateway_id="gateway",
                        auth_server_id="auth_server", broker_id="broker",
                        start_time=2000.0, persistent_context=device,
                    ).execute(spec)
                report = validate(events, context, source.provenance)
                proposed = replace(
                    source.candidate, trace_id=context.trace_id,
                    scenario_id=context.scenario_id, device_id=context.device_id,
                    target_event_ids=(events[-1].event_id,),
                    evidence_event_ids=(events[-1].event_id,),
                )
                requalified = qualify_hard_negative(events, context, proposed, report)
                self.assertEqual("validated_positive", report.trace_ground_truth_statuses[context.trace_id])
                self.assertTrue(report.trace_validated_invariant_ids[context.trace_id])
                self.assertEqual(QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION,
                                 requalified.qualification_status)

    def test_access_enforcement_is_not_authz_separation(self):
        missing = candidate("missing_auth_context_enforced", 6801)
        authz_variant = HN_VARIANT_BY_ID["valid_context_legitimate_denial"]
        switched = replace(
            missing.candidate, primary_intended_family_id=authz_variant.family_id,
            intended_variant_id=authz_variant.variant_id,
            intended_boundary_invariant_ids=authz_variant.invariant_ids,
            qualification_status=QualificationStatus.NOT_QUALIFIED_NOT_APPLIED,
            qualified_family_ids=(), qualified_boundary_invariant_ids=(),
            qualification_check_ids=(), normal_side_predicates={},
            boundary_proximity_evidence={}, relevant_validation_check_ids=(),
            relevant_validation_outcomes={}, qualifier_implementation_version=None,
            trace_ground_truth_status=None, trace_observable_anomaly=None,
        )
        outcome = qualify_hard_negative(missing.events, missing.context, switched,
                                        missing.validation)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED,
                         outcome.qualification_status)
        self.assertEqual((), outcome.qualified_family_ids)
        valid_denial = candidate(authz_variant.variant_id, 6802)
        self.assertEqual(QualificationStatus.QUALIFIED, valid_denial.candidate.qualification_status)

    def test_privileged_export_reload_keeps_evidence_links(self):
        result = candidate("inactive_session_enforced", 6901)
        with TemporaryDirectory() as directory:
            paths = SynchronizedOutputViews(directory).save(
                [(list(result.events), result.context)], provenance=result.provenance,
                hard_negative_qualification_records=[result.candidate],
            )
            exported = json.loads(Path(paths["hard_negative_qualification"]).read_text())
            provenance = json.loads(Path(paths["provenance"]).read_text())
            gt = Path(paths["ground_truth_traces"]).read_text()
            validation = json.loads(Path(paths["validation_report"]).read_text())
            record = exported[0]
            event_ids = {item["event_id"] for item in provenance["events"]}
            self.assertEqual(result.candidate.hard_negative_candidate_id,
                             record["hard_negative_candidate_id"])
            self.assertEqual(result.context.trace_id, record["trace_id"])
            self.assertEqual(result.context.device_id, record["device_id"])
            self.assertEqual(result.candidate.intended_variant_id, record["intended_variant_id"])
            self.assertEqual(list(result.candidate.qualification_check_ids), record["qualification_check_ids"])
            self.assertEqual(list(result.candidate.qualified_boundary_invariant_ids),
                             record["qualified_boundary_invariant_ids"])
            self.assertEqual("qualified", record["qualification_status"])
            self.assertTrue(set(record["target_event_ids"] + record["evidence_event_ids"]).issubset(event_ids))
            self.assertIn("validated_normal", gt)
            self.assertTrue(validation["build_gate"]["passed"])
            self.assertNotIn("hard_negative_qualification", field_names_for_view("detector_observations"))

    def test_nonce_retry_claim_requires_referenced_lifecycle_evidence(self):
        events, context, record, report = positive_fixture("hn.v1.nonce.fresh_cross_attempt")
        without_link = [event for event in events if event.event_type is not EventType.RETRY]
        claimed = replace(record, evidence_event_ids=(events[0].event_id, events[-1].event_id))
        outcome = qualify_hard_negative(without_link, context, claimed, report)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED,
                         outcome.qualification_status)
        unreferenced = qualify_hard_negative(events, context, claimed, report)
        self.assertEqual(QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE,
                         unreferenced.qualification_status)
        contradictory = replace(record, boundary_parameters={"attempt_relationship": "renewal"})
        self.assertEqual(QualificationStatus.QUALIFICATION_ERROR,
                         qualify_hard_negative(events, context, contradictory,
                                               report).qualification_status)

    def test_relevant_validator_disagreement_but_not_unrelated_unsupported_blocks(self):
        check_id = "hn.v1.access.missing_auth_enforced"
        for result, expected in (
            (ValidationResult.FAIL, QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION),
            (ValidationResult.ERROR, QualificationStatus.QUALIFICATION_ERROR),
            (ValidationResult.UNSUPPORTED, QualificationStatus.NOT_QUALIFIED_RELEVANT_UNSUPPORTED),
        ):
            events, context, record, report = positive_fixture(check_id)
            report.records[0].result = result
            self.assertEqual(expected,
                             qualify_hard_negative(events, context, record,
                                                   report).qualification_status)
        events, context, record, report = positive_fixture(check_id)
        unrelated = SimpleNamespace(
            validation_id="unrelated.unsupported", trace_id=context.trace_id,
            result=ValidationResult.UNSUPPORTED, invariant_identifier=None,
        )
        report.records.append(unrelated)
        self.assertEqual(QualificationStatus.QUALIFIED,
                         qualify_hard_negative(events, context, record,
                                               report).qualification_status)
        unrelated.result = ValidationResult.NOT_APPLICABLE
        self.assertEqual(QualificationStatus.QUALIFIED,
                         qualify_hard_negative(events, context, record,
                                               report).qualification_status)

    def test_time_nonce_token_and_access_boundary_sweeps(self):
        time_check = "hn.v1.time.positive_causal_margin"
        for observed, expected in (
            (9.9, QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION),
            (10.0, QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED),
            (10.1, QualificationStatus.QUALIFIED),
            (10.5, QualificationStatus.NOT_QUALIFIED_BOUNDARY_NOT_REACHED),
        ):
            events, context, record, report = positive_fixture(time_check)
            events[-1].observed_timestamp = observed
            self.assertEqual(expected,
                             qualify_hard_negative(events, context, record,
                                                   report).qualification_status)
        events, context, record, report = positive_fixture(time_check)
        record = replace(record, boundary_parameters={"epsilon_hn": float("inf")})
        self.assertEqual(QualificationStatus.QUALIFICATION_ERROR,
                         qualify_hard_negative(events, context, record,
                                               report).qualification_status)

        nonce_check = "hn.v1.nonce.fresh_cross_attempt"
        for mutation, expected in (
            ("same_nonce", QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION),
            ("unknown_attempt", QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED),
            ("missing_nonce", QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE),
        ):
            events, context, record, report = positive_fixture(nonce_check)
            if mutation == "same_nonce":
                events[-1].nonce = events[0].nonce
            elif mutation == "unknown_attempt":
                events[-1].auth_attempt_id = None
            else:
                events[-1].nonce = None
            self.assertEqual(expected,
                             qualify_hard_negative(events, context, record,
                                                   report).qualification_status)

        token_check = "hn.v1.token.replacement_token_use"
        for mutation, expected in (
            ("correct", QualificationStatus.QUALIFIED),
            ("superseded", QualificationStatus.NOT_QUALIFIED_INVARIANT_VIOLATION),
            ("failed_replacement", QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED),
            ("ambiguous", QualificationStatus.NOT_QUALIFIED_UNDERDETERMINED),
        ):
            events, context, record, report = positive_fixture(token_check)
            if mutation == "superseded":
                events[-1].token_id = "T1"
            elif mutation == "failed_replacement":
                from simulator.event_model import EventResult
                events[0].result = EventResult.FAILURE
            elif mutation == "ambiguous":
                events[0].supersedes_token_id = "T2"
            self.assertEqual(expected,
                             qualify_hard_negative(events, context, record,
                                                   report).qualification_status)

        access_check = "hn.v1.access.missing_auth_enforced"
        for outcome, expected in (
            ("denied", QualificationStatus.QUALIFIED),
            ("not_executed", QualificationStatus.QUALIFIED),
            (None, QualificationStatus.NOT_QUALIFIED_INSUFFICIENT_EVIDENCE),
        ):
            events, context, record, report = positive_fixture(access_check)
            events[-1].resource_operation_outcome = outcome
            self.assertEqual(expected,
                             qualify_hard_negative(events, context, record,
                                                   report).qualification_status)


if __name__ == "__main__":
    unittest.main()
