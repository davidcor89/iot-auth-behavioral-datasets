"""4D.4B global scheduling, canonical build, and integrity contracts."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from simulator.composition.build import (
    ADAPTER_DISPATCH_REGISTRY_VERSION,
    CompositionBuildError,
    CompositionIntegrityError,
    build_composition,
    enforce_detector_evidence_integrity,
)
from simulator.composition.diagnostics import post_composition_diagnostics
from simulator.composition.eligibility import build_eligibility_edges
from simulator.composition.fingerprint import detector_evidence_hash
from simulator.composition.model import (
    CompositionSpecification,
    ControlMembership,
    ControlReserveRequirement,
    DuplicateEvidencePolicy,
    ORDINARY_NORMAL,
    RoleKind,
    RoleRequest,
    ScientificRole,
)
from simulator.composition.scheduler import build_composition_plan
from simulator.config.integrated_role_stress import IntegratedRoleStressConfiguration
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.integrated_stress import build_stress_population_and_pool
from simulator.role_integration.transformation import TransformationStatus


def hn(variant: str) -> ScientificRole:
    return ScientificRole(RoleKind.HARD_NEGATIVE, variant)


def anomaly(variant: str) -> ScientificRole:
    return ScientificRole(RoleKind.ANOMALY, variant)


class ScientificCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = IntegratedRoleStressConfiguration(
            device_count=16, opportunities_per_device=2,
        )
        cls.population, cls.pool = build_stress_population_and_pool(cls.configuration)
        cls.specification = CompositionSpecification(
            role_requests=(
                RoleRequest(ORDINARY_NORMAL, 4, minimum_distinct_devices=4),
                RoleRequest(hn("fresh_nonce_cross_attempt"), 2, minimum_distinct_devices=2),
                RoleRequest(hn("missing_auth_context_enforced"), 2, minimum_distinct_devices=2),
                RoleRequest(anomaly("timestamp_inconsistency"), 2, minimum_distinct_devices=2),
                RoleRequest(anomaly("token_bound_identity_mismatch_authenticated_context"), 2,
                            minimum_distinct_devices=2),
            ),
            schedule_seed=8801,
            ordinary_control_reserves=(
                ControlReserveRequirement(ControlMembership.ACCESS, 1),
            ),
        )
        cls.plan = build_composition_plan(
            cls.population.behavioral_profiles, cls.pool, cls.specification,
        )
        cls.result = build_composition(
            cls.population.behavioral_profiles, cls.pool, cls.plan,
            transformation_seed=8802,
        )

    def test_feasible_mixed_global_plan_and_one_role_per_base(self):
        self.assertEqual("FEASIBLE", self.plan.feasibility_status)
        self.assertEqual(12, len(self.plan.assignments))
        references = [item.base_trace_reference for item in self.plan.assignments]
        self.assertEqual(len(references), len(set(references)))
        self.assertEqual(
            {item.role.key: item.count for item in self.specification.role_requests},
            {role: data["assigned"] for role, data in
             self.plan.diagnostics["requested_vs_assigned"].items()},
        )

    def test_ordinary_reserve_control_memberships_and_no_duplicate_samples(self):
        ordinary = [item for item in self.plan.assignments if item.role == ORDINARY_NORMAL]
        self.assertEqual(4, len(ordinary))
        self.assertTrue(any(ControlMembership.ACCESS.value in item.control_memberships
                            for item in ordinary))
        multi = [item for item in ordinary if len(item.control_memberships) > 1]
        self.assertTrue(multi)
        self.assertEqual(len(ordinary), len({item.base_trace_reference for item in ordinary}))

    def test_device_coverage_profile_and_structural_descriptors(self):
        for request in self.specification.role_requests:
            selected = [item for item in self.plan.assignments if item.role == request.role]
            self.assertGreaterEqual(len({item.device_key for item in selected}),
                                    request.minimum_distinct_devices)
        edges = build_eligibility_edges(
            self.population.behavioral_profiles, self.pool,
            (ORDINARY_NORMAL,), self.specification.schedule_seed,
        )
        edge = edges[0]
        self.assertEqual(
            {"lifecycle", "recovery", "renewal", "activity_count",
             "authentication_attempt_count", "token_count", "access_count",
             "base_event_count", "base_duration", "opportunity_source"},
            set(edge.structural_descriptors),
        )
        self.assertTrue({"T_AUTH_center", "T_RENEWAL_dispersion", "A", "R", "N",
                         "L_AUTH", "L_SESSION", "L_ACCESS", "lifecycle_cell",
                         "scalar_strata"}.issubset(edge.profile_descriptors))

    def test_same_seed_and_all_input_order_independence(self):
        repeated = build_composition_plan(
            self.population.behavioral_profiles, self.pool, self.specification,
        )
        reversed_specification = replace(
            self.specification,
            role_requests=tuple(reversed(self.specification.role_requests)),
        )
        reversed_roles = build_composition_plan(
            self.population.behavioral_profiles, self.pool, reversed_specification,
        )
        reversed_pool = replace(
            self.pool,
            bases=tuple(reversed(self.pool.bases)),
            applicability_matrix=tuple(reversed(self.pool.applicability_matrix)),
        )
        reversed_bases = build_composition_plan(
            self.population.behavioral_profiles, reversed_pool, self.specification,
        )
        reversed_population = replace(
            self.population.behavioral_profiles,
            assignments=tuple(reversed(self.population.behavioral_profiles.assignments)),
        )
        reversed_devices = build_composition_plan(
            reversed_population, self.pool, self.specification,
        )
        expected = [(item.base_trace_reference, item.role.key) for item in self.plan.assignments]
        self.assertEqual(self.plan.composition_fingerprint, repeated.composition_fingerprint)
        self.assertEqual(expected, [(item.base_trace_reference, item.role.key)
                                    for item in reversed_roles.assignments])
        self.assertEqual(self.plan.composition_fingerprint,
                         reversed_roles.composition_fingerprint)
        self.assertEqual(expected, [(item.base_trace_reference, item.role.key)
                                    for item in reversed_bases.assignments])
        self.assertEqual(expected, [(item.base_trace_reference, item.role.key)
                                    for item in reversed_devices.assignments])

    def test_schedule_seed_is_isolated_from_population_pool_and_global_rng(self):
        state = random.getstate()
        alternative = build_composition_plan(
            self.population.behavioral_profiles, self.pool,
            replace(self.specification, schedule_seed=8803),
        )
        self.assertEqual(state, random.getstate())
        self.assertEqual(self.plan.population_fingerprint, alternative.population_fingerprint)
        self.assertEqual(self.plan.pool_fingerprint, alternative.pool_fingerprint)
        self.assertEqual(self.plan.eligibility_digest, alternative.eligibility_digest)
        self.assertEqual(12, len(alternative.assignments))

    def test_transformation_seed_cannot_change_frozen_plan(self):
        before = self.plan.normalized_record()
        other = build_composition(
            self.population.behavioral_profiles, self.pool, self.plan,
            transformation_seed=9911,
        )
        self.assertEqual(before, self.plan.normalized_record())
        self.assertEqual(self.plan.composition_fingerprint, other.plan_fingerprint)

    def test_replacement_shared_support_is_globally_infeasible_before_transform(self):
        specification = CompositionSpecification((
            RoleRequest(hn("last_valid_use_before_replacement"), 3),
            RoleRequest(hn("correct_replacement_token_use"), 3),
            RoleRequest(anomaly("token_replay_after_replacement"), 3),
        ), schedule_seed=9001)
        plan = build_composition_plan(
            self.population.behavioral_profiles, self.pool, specification,
        )
        self.assertEqual("INFEASIBLE", plan.feasibility_status)
        self.assertIsNotNone(plan.infeasibility)
        self.assertTrue(plan.infeasibility.unsatisfied_constraints)
        with patch("simulator.composition.build._integrate") as adapter:
            with self.assertRaises(CompositionBuildError):
                build_composition(self.population.behavioral_profiles, self.pool, plan,
                                  transformation_seed=2)
            adapter.assert_not_called()

    def test_access_and_binding_overlap_never_duplicate_bases(self):
        specification = CompositionSpecification((
            RoleRequest(hn("missing_auth_context_enforced"), 3),
            RoleRequest(anomaly("access_success_without_auth_context"), 3),
            RoleRequest(hn("token_identity_binding_coherent"), 3),
            RoleRequest(anomaly("token_bound_identity_mismatch_authenticated_context"), 3),
        ), schedule_seed=9002)
        plan = build_composition_plan(
            self.population.behavioral_profiles, self.pool, specification,
        )
        self.assertEqual("FEASIBLE", plan.feasibility_status)
        bases = [item.base_trace_reference for item in plan.assignments]
        self.assertEqual(len(bases), len(set(bases)))

    def test_adversarial_overlapping_roles_enforce_device_minima_globally(self):
        specification = CompositionSpecification((
            RoleRequest(hn("missing_auth_context_enforced"), 4,
                        minimum_distinct_devices=4, maximum_per_device=1),
            RoleRequest(anomaly("access_success_without_auth_context"), 4,
                        minimum_distinct_devices=4, maximum_per_device=1),
        ), schedule_seed=9003)
        plan = build_composition_plan(
            self.population.behavioral_profiles, self.pool, specification,
        )
        self.assertEqual("FEASIBLE", plan.feasibility_status)
        for request in specification.role_requests:
            selected = [item for item in plan.assignments if item.role == request.role]
            self.assertEqual(4, len(selected))
            self.assertEqual(4, len({item.device_key for item in selected}))

    def test_ordinary_reserve_can_make_individually_feasible_roles_jointly_infeasible(self):
        specification = CompositionSpecification(
            (
                RoleRequest(ORDINARY_NORMAL, 1),
                RoleRequest(anomaly("access_success_without_auth_context"), 9),
            ),
            schedule_seed=9004,
            ordinary_control_reserves=(
                ControlReserveRequirement(ControlMembership.ACCESS, 1),
            ),
        )
        plan = build_composition_plan(
            self.population.behavioral_profiles, self.pool, specification,
        )
        self.assertEqual("INFEASIBLE", plan.feasibility_status)
        self.assertEqual(9, plan.infeasibility.maximum_immediately_eligible_support[
            "anomaly:access_success_without_auth_context"
        ])
        self.assertGreater(plan.infeasibility.bases_blocked_by_exclusivity["ordinary_normal"], 0)

    def test_build_postconditions_canonical_reload_provenance_and_immutability(self):
        self.assertEqual(12, len(self.result.samples))
        self.assertTrue(all(item.canonical_reload_valid for item in self.result.samples))
        for item in self.result.samples:
            self.assertTrue(item.provenance_reference.startswith("sha256:"))
            if item.assignment.role.kind is RoleKind.HARD_NEGATIVE:
                self.assertEqual("normal", item.detector_truth)
                self.assertEqual((), item.complete_invariant_ids)
                self.assertIn("qualified", item.semantic_qualifications)
            if item.assignment.role.kind is RoleKind.ANOMALY:
                self.assertEqual("anomaly", item.detector_truth)
                self.assertTrue(set(item.intended_invariant_ids).issubset(
                    item.complete_invariant_ids))
        self.assertEqual(self.pool.pool_fingerprint, self.plan.pool_fingerprint)
        self.assertEqual(self.population.behavioral_profiles.population_fingerprint,
                         self.plan.population_fingerprint)

    def test_failed_transformation_is_not_replaced_or_resampled(self):
        derived = next(item for item in self.plan.assignments
                       if item.role.kind is not RoleKind.ORDINARY_NORMAL)
        small_plan = replace(self.plan, assignments=(derived,))
        failed = SimpleNamespace(
            status=TransformationStatus.NOT_APPLICABLE, reason="test failure",
        )
        with patch("simulator.composition.build._integrate", return_value=failed) as adapter:
            with self.assertRaises(CompositionBuildError):
                build_composition(self.population.behavioral_profiles, self.pool, small_plan,
                                  transformation_seed=3)
            self.assertEqual(1, adapter.call_count)

    def test_detector_hash_uses_only_declared_observations(self):
        record = dict(self.pool.bases[0].event_records[0])
        original = detector_evidence_hash((record,))
        record.update({
            "anomaly_label": "privileged", "behavioral_profile_id": "secret",
            "authenticated_identity": "changed", "authorization_decision": "changed",
        })
        self.assertEqual(original, detector_evidence_hash((record,)))
        record["result"] = "failure" if record["result"] != "failure" else "success"
        self.assertNotEqual(original, detector_evidence_hash((record,)))

    def test_same_truth_duplicates_report_and_conflicting_truth_reject_policy(self):
        assignment = self.plan.assignments[0]
        normal = SimpleNamespace(
            assignment=assignment, event_records=(), event_count_delta=0, duration=0.0,
            event_type_delta=(), outcome_markers=(), intended_invariant_ids=(),
            complete_invariant_ids=(), collision_classification="none",
            aggregate_gate_passed=True, canonical_reload_valid=True,
            detector_evidence_hash="same", detector_truth="normal",
        )
        same = post_composition_diagnostics((normal, normal))
        self.assertEqual(1, same["same_truth_duplicate_groups"])
        report = enforce_detector_evidence_integrity(
            (normal, normal), DuplicateEvidencePolicy.REPORT_ONLY,
        )
        self.assertEqual(1, report["same_truth_duplicate_groups"])
        with self.assertRaises(CompositionIntegrityError):
            enforce_detector_evidence_integrity(
                (normal, normal),
                DuplicateEvidencePolicy.REJECT_DUPLICATE_EVIDENCE,
            )
        conflicting = replace(
            self.result.samples[0], detector_evidence_hash="collision", detector_truth="normal"
        )
        anomalous = replace(
            self.result.samples[-1], detector_evidence_hash="collision", detector_truth="anomaly"
        )
        diagnostic = post_composition_diagnostics((conflicting, anomalous))
        self.assertEqual(1, diagnostic["conflicting_truth_duplicate_groups"])
        for policy in DuplicateEvidencePolicy:
            with self.assertRaises(CompositionIntegrityError):
                enforce_detector_evidence_integrity(
                    (conflicting, anomalous), policy,
                )

        ordinary = next(item for item in self.plan.assignments
                        if item.role is ORDINARY_NORMAL or item.role == ORDINARY_NORMAL)
        one_sample_plan = replace(self.plan, assignments=(ordinary,))
        with patch(
            "simulator.composition.build.post_composition_diagnostics",
            return_value={"conflicting_truth_duplicate_groups": 1},
        ):
            with self.assertRaises(CompositionIntegrityError):
                build_composition(
                    self.population.behavioral_profiles, self.pool, one_sample_plan,
                    transformation_seed=8802,
                )

    def test_versions_detector_separation_and_non_final_policy(self):
        self.assertTrue(ADAPTER_DISPATCH_REGISTRY_VERSION.startswith("4d4b-"))
        self.assertEqual("non_final_engineering_composition", self.specification.purpose)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        detector_payload = str(self.result.diagnostics).lower()
        self.assertNotIn("t_auth_center", detector_payload)
        self.assertNotIn("behavioral_profile_id", detector_payload)


if __name__ == "__main__":
    unittest.main()
