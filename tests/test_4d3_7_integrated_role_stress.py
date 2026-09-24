"""Focused 4D.3.7 integrated role-transformation stress tests."""

from __future__ import annotations

import random
import unittest
from collections import Counter
from dataclasses import replace

from simulator.config.integrated_role_stress import (
    INTEGRATED_STRESS_CONFIGURATION_VERSION,
    IntegratedRoleStressConfiguration,
)
from simulator.data.dataset_contract import field_manifest
from simulator.data.synchronized_views import (
    CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
)
from simulator.device_profiles.scientific_population import (
    build_scientific_profile_population,
)
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.identity_binding_boundary import (
    PERSISTENT_IDENTITY_ANOMALY,
    TOKEN_IDENTITY_ANOMALY,
)
from simulator.role_integration.direct_timing import (
    POSITIVE_MARGIN_HN,
    TIMESTAMP_ANOMALY,
    RoleTransformationRequest,
    integrate_role,
)
from simulator.role_integration.integrated_stress import (
    ANOMALY_VARIANTS,
    COLLISION_CLASSIFICATION_VERSION,
    HANDOFF_SCHEMA_VERSION,
    HN_VARIANTS,
    INTEGRATED_STRESS_ORCHESTRATOR_VERSION,
    INTEGRATED_STRESS_OUTCOME_SCHEMA_VERSION,
    INTEGRATED_STRESS_PROVENANCE_VERSION,
    INTEGRATED_VARIANT_CATALOGUE_VERSION,
    INTEGRATED_VARIANT_SPECIFICATIONS,
    INTEGRATED_VARIANTS,
    INTENDED_INVARIANT_REGISTRY_VERSION,
    STRUCTURAL_DIFFERENCE_NORMALIZATION_VERSION,
    build_stress_population_and_pool,
    exercise_integrated_roles,
    summarize_outcomes,
)
from simulator.role_integration.pool import OPPORTUNITY_POOL_BUILDER_VERSION


class IntegratedRoleStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = IntegratedRoleStressConfiguration(
            device_count=16, opportunities_per_device=1,
        )
        cls.population, cls.pool = build_stress_population_and_pool(
            cls.configuration
        )
        cls.result = exercise_integrated_roles(
            cls.population, cls.pool, cls.configuration,
        )
        cls.summary = summarize_outcomes(cls.result)

    def test_versions_complete_catalogue_and_matrix(self):
        versions = (
            INTEGRATED_STRESS_CONFIGURATION_VERSION,
            INTEGRATED_VARIANT_CATALOGUE_VERSION,
            INTENDED_INVARIANT_REGISTRY_VERSION,
            INTEGRATED_STRESS_ORCHESTRATOR_VERSION,
            INTEGRATED_STRESS_OUTCOME_SCHEMA_VERSION,
            COLLISION_CLASSIFICATION_VERSION,
            STRUCTURAL_DIFFERENCE_NORMALIZATION_VERSION,
            INTEGRATED_STRESS_PROVENANCE_VERSION,
            HANDOFF_SCHEMA_VERSION,
            CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
        )
        self.assertTrue(all(version.startswith("4d3.7-") for version in versions))
        self.assertEqual(15, len(HN_VARIANTS))
        self.assertEqual(11, len(ANOMALY_VARIANTS))
        self.assertEqual(26, len(INTEGRATED_VARIANTS))
        self.assertEqual(26, len(set(INTEGRATED_VARIANTS)))
        self.assertEqual(set(INTEGRATED_VARIANTS), {
            item.variant_id for item in INTEGRATED_VARIANT_SPECIFICATIONS
        })
        self.assertEqual(
            len(self.pool.bases) * 26, len(self.result.outcomes),
        )
        self.assertEqual(set(INTEGRATED_VARIANTS), set(self.summary))

    def test_base_contract_profile_preservation_and_no_mutation(self):
        self.assertFalse(self.pool.rejected)
        self.assertEqual(
            OPPORTUNITY_POOL_BUILDER_VERSION,
            self.pool.diagnostics["canonical_admission_version"],
        )
        self.assertEqual(
            CANONICAL_EVIDENCE_SERIALIZATION_VERSION,
            self.pool.diagnostics["canonical_serialization_version"],
        )
        self.assertTrue(all(base.validation.accepted for base in self.pool.bases))
        before = tuple(base.to_record() for base in self.pool.bases)
        for outcome in self.result.outcomes:
            self.assertTrue(outcome.profile_preserved)
            self.assertTrue(outcome.base_reference_preserved)
            self.assertTrue(outcome.provenance_complete)
            assignment = self.population.behavioral_profiles.assignment_for_device(
                next(base.device_id for base in self.pool.bases
                     if base.base_trace_reference == outcome.base_trace_reference)
            )
            self.assertEqual(assignment.population_device_key,
                             outcome.population_device_key)
            self.assertEqual(assignment.profile.behavioral_profile_id,
                             outcome.behavioral_profile_id)
            base = next(
                base for base in self.pool.bases
                if base.base_trace_reference == outcome.base_trace_reference
            )
            self.assertEqual(base.structural_summary.lifecycle, outcome.lifecycle)
            self.assertEqual(base.structural_summary.recovery, outcome.recovery)
            self.assertEqual(base.structural_summary.renewal, outcome.renewal)
            self.assertEqual(base.structural_summary.activity_count,
                             outcome.activity_count)
        self.assertEqual(before, tuple(base.to_record() for base in self.pool.bases))

    def test_hns_are_normal_qualified_and_have_no_anomaly_truth(self):
        for outcome in self.result.outcomes:
            if outcome.role_type != "hard_negative" or outcome.construction_status != "APPLIED":
                continue
            self.assertTrue(outcome.hn_independently_normal)
            self.assertEqual("qualified", outcome.hn_qualification_status)
            self.assertEqual((), outcome.complete_validated_invariant_ids)
            self.assertFalse(outcome.gt_consolidated)
            self.assertTrue(outcome.aggregate_build_gate_passed)
            self.assertTrue(outcome.canonical_reload_integrity_passed)
            self.assertEqual("validated_normal",
                             outcome.canonical_reload_validation_status)
            self.assertEqual("qualified",
                             outcome.canonical_reload_hn_qualification_status)
            self.assertEqual((), outcome.canonical_reload_validated_invariant_ids)

    def test_anomalies_retain_intended_and_complete_invariant_sets(self):
        for outcome in self.result.outcomes:
            if outcome.role_type != "anomaly" or outcome.construction_status != "APPLIED":
                continue
            self.assertTrue(outcome.intended_invariants_validated)
            self.assertTrue(outcome.gt_consolidated)
            self.assertTrue(set(outcome.intended_invariant_ids).issubset(
                outcome.complete_validated_invariant_ids
            ))
            self.assertTrue(outcome.canonical_reload_integrity_passed)
            self.assertEqual(outcome.complete_validated_invariant_ids,
                             outcome.canonical_reload_validated_invariant_ids)
        for variant in (
            "access_success_despite_explicit_deny",
            "access_success_without_security_prerequisites",
        ):
            rows = [
                item for item in self.result.outcomes
                if item.variant_id == variant
                and item.construction_status == "APPLIED"
            ]
            self.assertTrue(rows)
            self.assertTrue(all(
                item.collision_classification == "expected_frozen_multi_invariant"
                for item in rows
            ))
        identity = [
            item for item in self.result.outcomes
            if item.variant_id == PERSISTENT_IDENTITY_ANOMALY
            and item.construction_status == "APPLIED"
        ]
        for item in identity:
            expected = {"INV-IDENT-AUTH-ACTOR-CONSISTENCY"}
            if item.lifecycle == "ACCESS":
                expected.add("INV-ACCESS-AUTH-CONTEXT")
                self.assertEqual("known_representation_coupling",
                                 item.collision_classification)
                self.assertFalse(item.aggregate_build_gate_passed)
            else:
                self.assertEqual("none", item.collision_classification)
                self.assertTrue(item.aggregate_build_gate_passed)
            self.assertEqual(expected, set(item.complete_validated_invariant_ids))

    def test_token_binding_is_independent_of_persistent_identity(self):
        rows = [
            item for item in self.result.outcomes
            if item.variant_id == TOKEN_IDENTITY_ANOMALY
            and item.construction_status == "APPLIED"
        ]
        self.assertTrue(rows)
        for item in rows:
            self.assertEqual(("INV-BIND-TOKEN-CONTEXT",),
                             item.complete_validated_invariant_ids)
            self.assertNotIn("INV-IDENT-AUTH-ACTOR-CONSISTENCY",
                             item.complete_validated_invariant_ids)
            self.assertNotIn("INV-ACCESS-AUTH-CONTEXT",
                             item.complete_validated_invariant_ids)
            self.assertEqual("none", item.collision_classification)

    def test_reference_and_structural_signatures_remain_distinct(self):
        reference_variants = {
            "legitimate_same_attempt_nonce_recurrence",
            "persistent_authenticated_identity_coherent",
            "fresh_nonce_cross_attempt",
            "repeated_current_token_use",
            "last_valid_use_before_replacement",
            "correct_replacement_token_use",
            "token_identity_binding_coherent",
            "token_protected_context_binding_coherent",
        }
        for item in self.result.outcomes:
            if item.construction_status != "APPLIED":
                continue
            if item.variant_id in reference_variants:
                self.assertTrue(item.structural_difference.exact_event_equality)
            if item.variant_id in {
                "missing_auth_context_enforced", "invalid_credential_enforced",
                "inactive_session_enforced", "no_grant_enforced",
                "explicit_deny_enforced", "valid_context_legitimate_denial",
            }:
                self.assertGreater(item.structural_difference.event_count_delta, 0)
            if item.variant_id in {"small_positive_causal_margin", "timestamp_inconsistency"}:
                self.assertGreater(item.structural_difference.observed_timestamp_change_count, 0)
                self.assertEqual(0, item.structural_difference.semantic_timestamp_change_count)

    def test_timing_boundaries_retain_their_sign_after_canonical_reload(self):
        for variant, positive in (
            (POSITIVE_MARGIN_HN, True),
            (TIMESTAMP_ANOMALY, False),
        ):
            result = next(
                candidate
                for base in self.pool.bases
                for candidate in (
                    integrate_role(base, RoleTransformationRequest(
                        base.base_trace_reference,
                        variant,
                        self.configuration.transformation_seed,
                    )),
                )
                if candidate.status.value == "APPLIED"
            )
            provenance = result.transformation_provenance
            predecessor = result.event_records[
                int(provenance["predecessor_event_reference"].split(":")[1]) - 1
            ]
            target = result.event_records[
                int(provenance["target_event_reference"].split(":")[1]) - 1
            ]
            margin = (
                target["observed_timestamp"]
                - predecessor["observed_timestamp"]
            )
            if positive:
                self.assertGreater(margin, 0.0)
            else:
                self.assertLess(margin, 0.0)

    def test_variant_and_base_order_independence(self):
        reversed_variants = exercise_integrated_roles(
            self.population, self.pool, self.configuration,
            variants=tuple(reversed(INTEGRATED_VARIANTS)),
        )
        reversed_bases = exercise_integrated_roles(
            self.population, self.pool, self.configuration,
            bases=tuple(reversed(self.pool.bases)),
        )
        self.assertEqual(self.result.result_fingerprint,
                         reversed_variants.result_fingerprint)
        self.assertEqual(self.result.result_fingerprint,
                         reversed_bases.result_fingerprint)

    def test_same_seed_reproducibility_and_rng_isolation(self):
        state = random.getstate()
        population, pool = build_stress_population_and_pool(self.configuration)
        repeated = exercise_integrated_roles(population, pool, self.configuration)
        self.assertEqual(state, random.getstate())
        self.assertEqual(self.result.population_fingerprint,
                         repeated.population_fingerprint)
        self.assertEqual(self.result.opportunity_pool_fingerprint,
                         repeated.opportunity_pool_fingerprint)
        self.assertEqual(self.result.result_fingerprint,
                         repeated.result_fingerprint)

    def test_device_order_independence_and_different_seed_sanity(self):
        assignments = self.population.behavioral_profiles.assignments
        bindings = [(item.device_id, item.population_device_key) for item in assignments]
        normal = build_scientific_profile_population(
            bindings, self.configuration.population_configuration(),
        )
        reversed_population = build_scientific_profile_population(
            reversed(bindings), self.configuration.population_configuration(),
        )
        self.assertEqual(normal.population_fingerprint,
                         reversed_population.population_fingerprint)
        alternative_configuration = replace(
            self.configuration, population_seed=4707,
            device_construction_seed=4706, pool_seed=4708,
            planning_root_seed=4709,
        )
        alternative, alternative_pool = build_stress_population_and_pool(
            alternative_configuration
        )
        self.assertNotEqual(self.result.population_fingerprint,
                            alternative.behavioral_profiles.population_fingerprint)
        self.assertNotEqual(self.result.opportunity_pool_fingerprint,
                            alternative_pool.pool_fingerprint)
        self.assertEqual(26, len(INTEGRATED_VARIANTS))

    def test_references_unique_complete_and_detector_separated(self):
        references = [item.integrated_reference for item in self.result.outcomes]
        self.assertEqual(len(references), len(set(references)))
        self.assertTrue(all(item.selected_target_references is not None
                            for item in self.result.outcomes))
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        exported = {item["field_name"] for item in field_manifest()["fields"]}
        forbidden = {
            "base_trace_reference", "integrated_reference", "variant_intent",
            "intended_invariant_ids", "complete_validated_invariant_ids",
            "collision_classification", "canonical_target_references",
            "hn_qualification_status", "aggregate_build_gate_passed",
            "behavioral_profile_id", "population_device_key",
        }
        self.assertTrue(forbidden.isdisjoint(exported))


if __name__ == "__main__":
    unittest.main()
