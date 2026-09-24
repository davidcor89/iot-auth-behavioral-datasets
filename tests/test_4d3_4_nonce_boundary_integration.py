"""Focused 4D.3.4 cross-attempt nonce-boundary integration tests."""

from __future__ import annotations

import random
import unittest
from collections import Counter
from dataclasses import replace

from simulator.config.nonce_role_transformation import (
    NONCE_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
)
from simulator.config.opportunity_pool import OpportunityPoolConfiguration
from simulator.config.scientific_population import ScientificPopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import field_manifest
from simulator.hard_negative import QualificationStatus
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.models import canonical_json
from simulator.role_integration.nonce_boundary import (
    FRESH_NONCE_HN,
    NONCE_ATTEMPT_PAIR_POLICY_VERSION,
    NONCE_BOUNDARY_ADAPTER_VERSION,
    NONCE_DERIVED_REFERENCE_VERSION,
    NONCE_EVIDENCE_RESOLVER_VERSION,
    NONCE_MATCHED_BOUNDARY_REFERENCE_VERSION,
    NONCE_OPPORTUNITY_SOURCE_CLASSIFIER_VERSION,
    NONCE_REUSE_ANOMALY,
    NONCE_TRANSFORMATION_PROVENANCE_VERSION,
    NonceRoleTransformationRequest,
    classify_multi_attempt_source,
    integrate_nonce_role,
    nonce_matched_boundary_reference,
    resolve_authentication_attempts,
    select_nonce_attempt_pair,
)
from simulator.role_integration.pool import build_opportunity_pool
from simulator.role_integration.transformation import TransformationStatus
from simulator.runner import build_run_population


def _pool():
    previous = cfg.simulation.num_devices
    try:
        cfg.simulation.num_devices = 16
        random.seed(991)
        population = build_run_population(ScientificPopulationConfiguration(
            device_count=16, population_seed=41,
        ))
        return build_opportunity_pool(
            population,
            OpportunityPoolConfiguration(
                opportunities_per_device=3, pool_seed=4301,
                planning_root_seed=4302, role_assignment_seed=4303,
                transformation_seed=4304,
            ),
        )
    finally:
        cfg.simulation.num_devices = previous


def _request(base, variant, seed=4304):
    return NonceRoleTransformationRequest(base.base_trace_reference, variant, seed)


class NonceBoundaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()
        cls.precise = [base for base in cls.pool.bases if select_nonce_attempt_pair(base)]
        cls.results = {
            (base.base_trace_reference, variant): integrate_nonce_role(
                base, _request(base, variant),
            )
            for base in cls.pool.bases
            for variant in (FRESH_NONCE_HN, NONCE_REUSE_ANOMALY)
        }

    def test_versions_exact_scope_and_broad_precise_denominators(self):
        for version in (
            NONCE_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
            NONCE_EVIDENCE_RESOLVER_VERSION,
            NONCE_ATTEMPT_PAIR_POLICY_VERSION,
            NONCE_BOUNDARY_ADAPTER_VERSION,
            NONCE_TRANSFORMATION_PROVENANCE_VERSION,
            NONCE_DERIVED_REFERENCE_VERSION,
            NONCE_OPPORTUNITY_SOURCE_CLASSIFIER_VERSION,
            NONCE_MATCHED_BOUNDARY_REFERENCE_VERSION,
        ):
            self.assertTrue(version.startswith("4d3.4-"))
        broad = [
            base for base, vector in zip(self.pool.bases, self.pool.applicability_matrix)
            if next(item for item in vector if item.predicate_id ==
                    ApplicabilityPredicate.DISTINCT_AUTHENTICATION_ATTEMPTS.value).applicable
        ]
        self.assertEqual(48, len(self.pool.bases))
        self.assertEqual(31, len(broad))
        self.assertEqual(31, len(self.precise))
        self.assertEqual(14, len({base.device_id for base in self.precise}))
        for variant in (FRESH_NONCE_HN, NONCE_REUSE_ANOMALY):
            statuses = Counter(
                self.results[(base.base_trace_reference, variant)].status
                for base in self.pool.bases
            )
            self.assertEqual(31, statuses[TransformationStatus.APPLIED])
            self.assertEqual(17, statuses[TransformationStatus.NOT_APPLICABLE])
            self.assertEqual(0, statuses[TransformationStatus.UNSUPPORTED])
            self.assertEqual(0, statuses[TransformationStatus.CONSTRUCTION_ERROR])

    def test_resolver_pair_policy_and_source_characterization(self):
        sources = Counter(classify_multi_attempt_source(base) for base in self.precise)
        self.assertEqual({
            "retry_only": 27, "renewal_only": 3, "retry_and_renewal": 1,
        }, dict(sources))
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, {
            base.structural_summary.lifecycle for base in self.precise
        })
        for base in self.precise:
            attempts = resolve_authentication_attempts(base)
            pair = select_nonce_attempt_pair(base)
            self.assertGreaterEqual(len(attempts), 2)
            self.assertEqual(next(
                (candidate for left, earlier in enumerate(attempts)
                 for later in attempts[left + 1:]
                 if (candidate := (earlier, later))
                 and earlier.attempt_id != later.attempt_id
                 and earlier.semantic_order_valid and later.semantic_order_valid
                 and earlier.observed_order_valid and later.observed_order_valid
                 and later.source_kind in {"retry", "renewal"}
                 and later.link_event_id
                 and earlier.last_ordinal < later.nonce_received_ordinal),
                None,
            ), (pair.earlier, pair.later))
            self.assertNotEqual(pair.earlier.attempt_id, pair.later.attempt_id)
            self.assertNotEqual(pair.earlier.nonce, pair.later.nonce)
            self.assertIn(pair.relationship, {"retry", "renewal"})

    def test_fresh_hn_is_reference_only_normal_and_qualified(self):
        for base in self.precise:
            result = self.results[(base.base_trace_reference, FRESH_NONCE_HN)]
            self.assertEqual(base.event_records_json, result.event_records_json)
            self.assertEqual(base.session_context_json, result.session_context_json)
            self.assertEqual("reference_only", result.transformation_provenance["construction_mode"])
            self.assertEqual([], result.transformation_provenance["transformed_nonce_event_ids"])
            self.assertEqual("validated_normal", result.validation_status)
            self.assertEqual((), result.validated_invariant_ids)
            self.assertTrue(result.validation_build_gate_passed)
            self.assertEqual(
                QualificationStatus.QUALIFIED,
                result.hard_negative_qualification.qualification_status,
            )

    def test_nonce_reuse_changes_complete_later_nonce_evidence_only(self):
        allowed_metadata = {
            "nonce", "injection_id", "anomaly_variant_name", "invariant_families",
            "source_context", "anomaly_label",
        }
        for base in self.precise:
            pair = select_nonce_attempt_pair(base)
            result = self.results[(base.base_trace_reference, NONCE_REUSE_ANOMALY)]
            self.assertEqual("validated_positive", result.validation_status)
            self.assertEqual(("INV-REPLAY-NONCE-CROSS-ATTEMPT",),
                             result.validated_invariant_ids)
            self.assertTrue(result.validation_build_gate_passed)
            self.assertEqual(len(base.event_records), len(result.event_records))
            later_indices = {ordinal - 1 for ordinal in pair.later.nonce_bearing_ordinals}
            for index, (original, derived) in enumerate(
                zip(base.event_records, result.event_records)
            ):
                for key, value in original.items():
                    if derived.get(key) != value:
                        self.assertIn(index, later_indices)
                        self.assertIn(key, allowed_metadata)
            later = [result.event_records[index]["nonce"] for index in later_indices]
            self.assertEqual({pair.earlier.nonce}, set(later))
            # Earlier attempt remains byte-identical.
            for ordinal in pair.earlier.nonce_bearing_ordinals:
                self.assertEqual(base.event_records[ordinal - 1],
                                 result.event_records[ordinal - 1])

    def test_structure_timing_activity_pairing_and_immutability(self):
        pool_before = canonical_json(self.pool.to_record())
        for base in self.precise:
            base_before = canonical_json(base.to_record())
            pair = select_nonce_attempt_pair(base)
            hn = self.results[(base.base_trace_reference, FRESH_NONCE_HN)]
            anomaly = self.results[(base.base_trace_reference, NONCE_REUSE_ANOMALY)]
            for result in (hn, anomaly):
                self.assertEqual(
                    [item["event_type"] for item in base.event_records],
                    [item["event_type"] for item in result.event_records],
                )
                self.assertEqual(
                    [item["timestamp"] for item in base.event_records],
                    [item["timestamp"] for item in result.event_records],
                )
                self.assertEqual(
                    [item["observed_timestamp"] for item in base.event_records],
                    [item["observed_timestamp"] for item in result.event_records],
                )
                self.assertEqual(base.structural_summary.activity_count, sum(
                    item["event_type"] == "access_granted"
                    and item["resource_operation_outcome"] == "success"
                    for item in result.event_records
                ))
            self.assertEqual(
                (pair.earlier.attempt_id, pair.later.attempt_id),
                (hn.transformation_provenance["selected_earlier_attempt_id"],
                 hn.transformation_provenance["selected_later_attempt_id"]),
            )
            self.assertEqual(
                (pair.earlier.attempt_id, pair.later.attempt_id),
                (anomaly.transformation_provenance["selected_earlier_attempt_id"],
                 anomaly.transformation_provenance["selected_later_attempt_id"]),
            )
            expected_pair = nonce_matched_boundary_reference(base.base_trace_reference)
            self.assertEqual(expected_pair, hn.transformation_provenance["matched_boundary_reference"])
            self.assertEqual(expected_pair, anomaly.transformation_provenance["matched_boundary_reference"])
            self.assertEqual(base_before, canonical_json(base.to_record()))
        self.assertEqual(pool_before, canonical_json(self.pool.to_record()))

    def test_rng_isolation_failure_statuses_and_detector_boundary(self):
        base = self.precise[-1]
        state = random.getstate()
        first = integrate_nonce_role(base, _request(base, NONCE_REUSE_ANOMALY, 77))
        second = integrate_nonce_role(base, _request(base, NONCE_REUSE_ANOMALY, 77))
        self.assertEqual(first, second)
        self.assertEqual(state, random.getstate())
        unsupported = integrate_nonce_role(base, _request(base, "token_replay_after_replacement"))
        self.assertEqual(TransformationStatus.UNSUPPORTED, unsupported.status)
        non_applicable = next(item for item in self.pool.bases if item not in self.precise)
        self.assertEqual(
            TransformationStatus.NOT_APPLICABLE,
            integrate_nonce_role(non_applicable, _request(non_applicable, FRESH_NONCE_HN)).status,
        )
        records = tuple(
            canonical_json({**record, "nonce": None})
            if record.get("auth_attempt_id") == select_nonce_attempt_pair(base).later.attempt_id
            else canonical_json(record)
            for record in base.event_records
        )
        incomplete = replace(base, event_records_json=records)
        self.assertEqual(
            TransformationStatus.UNSUPPORTED,
            integrate_nonce_role(incomplete, _request(incomplete, FRESH_NONCE_HN)).status,
        )
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        detector = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        self.assertTrue({
            "base_trace_reference", "attempt_pair", "opportunity_source",
            "variant_id", "qualification_status", "counterfactual_reference",
            "behavioral_profile_id",
        }.isdisjoint(detector))


if __name__ == "__main__":
    unittest.main()
