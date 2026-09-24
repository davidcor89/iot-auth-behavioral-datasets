"""Focused 4D.3.5 token-currentness boundary integration tests."""

from __future__ import annotations

import random
import unittest
from collections import Counter

from simulator.config.token_role_transformation import (
    TOKEN_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
)
from simulator.data.dataset_contract import field_manifest
from simulator.event_model import EventType
from simulator.hard_negative import QualificationStatus
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.models import canonical_json
from simulator.role_integration.token_boundary import (
    CORRECT_REPLACEMENT_T2_HN,
    LAST_VALID_T1_HN,
    POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION,
    PRE_REPLACEMENT_T1_SELECTION_POLICY_VERSION,
    REPEATED_CURRENT_SELECTION_POLICY_VERSION,
    REPEATED_CURRENT_TOKEN_HN,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY,
    TOKEN_BOUNDARY_ADAPTER_VERSION,
    TOKEN_DERIVED_REFERENCE_VERSION,
    TOKEN_EVIDENCE_RESOLVER_VERSION,
    TOKEN_MATCHED_BOUNDARY_REFERENCE_VERSION,
    TOKEN_OPPORTUNITY_CLASSIFIER_VERSION,
    TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION,
    TOKEN_TRANSFORMATION_PROVENANCE_VERSION,
    TokenRoleTransformationRequest,
    integrate_token_role,
    resolve_repeated_current_token_use,
    select_token_replacement,
    token_matched_boundary_reference,
)
from simulator.role_integration.transformation import TransformationStatus
from tests.test_4d3_4_nonce_boundary_integration import _pool


VARIANTS = (
    REPEATED_CURRENT_TOKEN_HN,
    LAST_VALID_T1_HN,
    CORRECT_REPLACEMENT_T2_HN,
    SUPERSEDED_TOKEN_REPLAY_ANOMALY,
)


def _request(base, variant, seed=4304):
    return TokenRoleTransformationRequest(base.base_trace_reference, variant, seed)


class TokenBoundaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()
        cls.repeated = [
            base for base in cls.pool.bases
            if resolve_repeated_current_token_use(base) is not None
        ]
        cls.replacements = [
            base for base in cls.pool.bases if select_token_replacement(base) is not None
        ]
        cls.results = {
            (base.base_trace_reference, variant): integrate_token_role(
                base, _request(base, variant),
            )
            for base in cls.pool.bases for variant in VARIANTS
        }

    def test_versions_exact_scope_and_variant_specific_applicability(self):
        versions = (
            TOKEN_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
            TOKEN_EVIDENCE_RESOLVER_VERSION,
            REPEATED_CURRENT_SELECTION_POLICY_VERSION,
            TOKEN_REPLACEMENT_LINEAGE_RESOLVER_VERSION,
            PRE_REPLACEMENT_T1_SELECTION_POLICY_VERSION,
            POST_REPLACEMENT_T2_SELECTION_POLICY_VERSION,
            TOKEN_BOUNDARY_ADAPTER_VERSION,
            TOKEN_TRANSFORMATION_PROVENANCE_VERSION,
            TOKEN_DERIVED_REFERENCE_VERSION,
            TOKEN_MATCHED_BOUNDARY_REFERENCE_VERSION,
            TOKEN_OPPORTUNITY_CLASSIFIER_VERSION,
        )
        self.assertTrue(all(version.startswith("4d3.5-") for version in versions))
        predicates = {
            result.predicate_id: sum(
                item.applicable
                for vector in self.pool.applicability_matrix
                for item in vector
                if item.predicate_id == result.predicate_id
            )
            for result in self.pool.applicability_matrix[0]
        }
        self.assertEqual(48, len(self.pool.bases))
        self.assertEqual(8, predicates[ApplicabilityPredicate.MULTIPLE_CURRENT_TOKEN_USES.value])
        self.assertEqual(4, predicates[ApplicabilityPredicate.SUCCESSFUL_TOKEN_REPLACEMENT.value])
        self.assertEqual(4, predicates[ApplicabilityPredicate.CURRENT_T1_USE_BEFORE_REPLACEMENT.value])
        self.assertEqual(4, predicates[ApplicabilityPredicate.T2_PRESENTATION_AFTER_REPLACEMENT.value])
        self.assertEqual(8, len(self.repeated))
        self.assertEqual(6, len({base.device_id for base in self.repeated}))
        self.assertEqual(4, len(self.replacements))
        self.assertEqual(4, len({base.device_id for base in self.replacements}))
        expected_applied = {
            REPEATED_CURRENT_TOKEN_HN: 8,
            LAST_VALID_T1_HN: 4,
            CORRECT_REPLACEMENT_T2_HN: 4,
            SUPERSEDED_TOKEN_REPLAY_ANOMALY: 4,
        }
        for variant, applied in expected_applied.items():
            statuses = Counter(
                self.results[(base.base_trace_reference, variant)].status
                for base in self.pool.bases
            )
            self.assertEqual(applied, statuses[TransformationStatus.APPLIED])
            self.assertEqual(48 - applied, statuses[TransformationStatus.NOT_APPLICABLE])
            self.assertEqual(0, statuses[TransformationStatus.UNSUPPORTED])
            self.assertEqual(0, statuses[TransformationStatus.CONSTRUCTION_ERROR])

    def test_resolvers_use_protected_operations_and_explicit_replacement_lineage(self):
        repeated_shapes = Counter(
            (
                base.structural_summary.lifecycle,
                base.structural_summary.recovery,
                base.structural_summary.renewal,
                base.structural_summary.activity_count,
            )
            for base in self.repeated
        )
        self.assertEqual({"ACCESS"}, {shape[0] for shape in repeated_shapes})
        self.assertEqual({"NO_RENEWAL", "RENEWAL"}, {shape[2] for shape in repeated_shapes})
        self.assertEqual({3, 4}, {shape[3] for shape in repeated_shapes})
        for base in self.repeated:
            evidence = resolve_repeated_current_token_use(base)
            self.assertGreaterEqual(len(evidence.uses), 2)
            self.assertEqual({EventType.ACCESS_REQUEST.value}, {
                item.event_type for item in evidence.uses
            })
            self.assertEqual({evidence.token_id}, {
                item.token_id for item in evidence.uses
            })
            self.assertEqual({evidence.protected_session_id}, {
                item.protected_session_id for item in evidence.uses
            })
            self.assertTrue(all(not item.expired_at_use for item in evidence.uses))
        for base in self.replacements:
            evidence = select_token_replacement(base)
            self.assertNotEqual(evidence.t1_token_id, evidence.t2_token_id)
            self.assertEqual(evidence.t1_generation + 1, evidence.t2_generation)
            self.assertEqual("RENEWAL", base.structural_summary.renewal)
            self.assertTrue(evidence.semantic_order_valid)
            self.assertTrue(evidence.observed_order_valid)
            self.assertIsNotNone(evidence.pre_replacement_use)
            self.assertIsNotNone(evidence.post_replacement_use)
            self.assertIsNotNone(evidence.post_replacement_validation)
            self.assertEqual(EventType.ACCESS_REQUEST.value,
                             evidence.pre_replacement_use.event_type)
            self.assertEqual(EventType.TOKEN_PRESENTED.value,
                             evidence.post_replacement_use.event_type)
            self.assertEqual(EventType.TOKEN_VALIDATED.value,
                             evidence.post_replacement_validation.event_type)
            self.assertEqual(evidence.protected_session_id,
                             evidence.pre_replacement_use.protected_session_id)
            self.assertEqual(evidence.protected_session_id,
                             evidence.post_replacement_use.protected_session_id)
            self.assertLess(evidence.pre_replacement_use.timestamp,
                            evidence.replacement_timestamp)
            self.assertLess(evidence.replacement_timestamp,
                            evidence.post_replacement_use.timestamp)

    def test_all_three_hns_are_reference_only_normal_and_qualified(self):
        applicable = {
            REPEATED_CURRENT_TOKEN_HN: self.repeated,
            LAST_VALID_T1_HN: self.replacements,
            CORRECT_REPLACEMENT_T2_HN: self.replacements,
        }
        for variant, bases in applicable.items():
            for base in bases:
                result = self.results[(base.base_trace_reference, variant)]
                self.assertEqual(base.event_records_json, result.event_records_json)
                self.assertEqual(base.session_context_json, result.session_context_json)
                self.assertEqual("reference_only",
                                 result.transformation_provenance["construction_mode"])
                self.assertEqual([], result.transformation_provenance["transformed_event_references"])
                self.assertEqual("validated_normal", result.validation_status)
                self.assertEqual((), result.validated_invariant_ids)
                self.assertTrue(result.validation_build_gate_passed)
                self.assertEqual(
                    QualificationStatus.QUALIFIED,
                    result.hard_negative_qualification.qualification_status,
                )

    def test_replay_substitutes_one_operation_and_preserves_replacement(self):
        allowed_target = {
            "token_id", "current_token_id", "token_generation",
            "token_lineage_status", "supersedes_token_id",
            "superseded_by_token_id", "token_replacement_timestamp",
            "token_expiry", "token_scope", "token_bound_identity",
            "token_bound_protected_session_id", "injection_id",
            "anomaly_variant_name", "invariant_families", "source_context",
            "anomaly_label",
        }
        allowed_enforcement = allowed_target | {
            "event_type", "result", "failure_reason", "new_state",
            "token_validation_result",
        }
        for base in self.replacements:
            evidence = select_token_replacement(base)
            result = self.results[(
                base.base_trace_reference, SUPERSEDED_TOKEN_REPLAY_ANOMALY,
            )]
            self.assertEqual("validated_positive", result.validation_status)
            self.assertEqual(("INV-REPLAY-TOKEN-SUPERSEDED",),
                             result.validated_invariant_ids)
            self.assertTrue(result.validation_build_gate_passed)
            self.assertFalse(result.transformation_provenance["token_validity_collision"])
            self.assertEqual(len(base.event_records), len(result.event_records))
            changed = {
                evidence.post_replacement_use.ordinal - 1: allowed_target,
                evidence.post_replacement_validation.ordinal - 1: allowed_enforcement,
            }
            for index, (original, derived) in enumerate(
                zip(base.event_records, result.event_records)
            ):
                self.assertEqual(original["timestamp"], derived["timestamp"])
                self.assertEqual(original["observed_timestamp"], derived["observed_timestamp"])
                for key, value in original.items():
                    if derived.get(key) != value:
                        self.assertIn(index, changed)
                        self.assertIn(key, changed[index])
            replay = result.event_records[evidence.post_replacement_use.ordinal - 1]
            enforcement = result.event_records[
                evidence.post_replacement_validation.ordinal - 1
            ]
            replacement = result.event_records[evidence.replacement_ordinal - 1]
            self.assertEqual(evidence.t1_token_id, replay["token_id"])
            self.assertEqual(evidence.t2_token_id, replay["current_token_id"])
            self.assertEqual("superseded", replay["token_lineage_status"])
            self.assertEqual(evidence.t2_token_id, replay["superseded_by_token_id"])
            self.assertEqual(evidence.t2_token_id, replacement["token_id"])
            self.assertEqual(evidence.t1_token_id, replacement["supersedes_token_id"])
            self.assertEqual(EventType.TOKEN_REJECTED.value, enforcement["event_type"])
            self.assertEqual("rejected", enforcement["token_validation_result"])
            self.assertEqual(evidence.t1_token_id, enforcement["token_id"])

    def test_matched_boundary_pool_immutability_and_rng_isolation(self):
        pool_before = canonical_json(self.pool.to_record())
        for base in self.replacements:
            base_before = canonical_json(base.to_record())
            evidence = select_token_replacement(base)
            normal = self.results[(base.base_trace_reference, CORRECT_REPLACEMENT_T2_HN)]
            replay = self.results[(base.base_trace_reference, SUPERSEDED_TOKEN_REPLAY_ANOMALY)]
            expected = token_matched_boundary_reference(base.base_trace_reference)
            self.assertEqual(expected,
                             normal.transformation_provenance["matched_boundary_reference"])
            self.assertEqual(expected,
                             replay.transformation_provenance["matched_boundary_reference"])
            self.assertEqual(
                f"event:{evidence.post_replacement_use.ordinal}",
                normal.transformation_provenance["selected_event_references"][
                    "post_replacement_t2_use"
                ],
            )
            self.assertEqual(
                f"event:{evidence.post_replacement_use.ordinal}",
                replay.transformation_provenance["selected_event_references"][
                    "post_replacement_t2_use"
                ],
            )
            self.assertEqual(base_before, canonical_json(base.to_record()))
        self.assertEqual(pool_before, canonical_json(self.pool.to_record()))
        base = self.replacements[0]
        state = random.getstate()
        first = integrate_token_role(base, _request(base, SUPERSEDED_TOKEN_REPLAY_ANOMALY, 77))
        second = integrate_token_role(base, _request(base, SUPERSEDED_TOKEN_REPLAY_ANOMALY, 77))
        self.assertEqual(first, second)
        self.assertEqual(state, random.getstate())

    def test_unsupported_scope_detector_boundary_and_schema_neutrality(self):
        base = self.replacements[0]
        unsupported = integrate_token_role(
            base, _request(base, "token_identity_binding_mismatch"),
        )
        self.assertEqual(TransformationStatus.UNSUPPORTED, unsupported.status)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        detector = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        self.assertTrue({
            "t1_token_id", "t2_token_id", "replacement_reference",
            "selected_token_use", "variant_id", "qualification_status",
            "counterfactual_reference", "behavioral_profile_id",
        }.isdisjoint(detector))


if __name__ == "__main__":
    unittest.main()
