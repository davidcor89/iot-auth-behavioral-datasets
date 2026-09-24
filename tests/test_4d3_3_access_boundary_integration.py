"""Focused 4D.3.3 access-boundary integration tests."""

from __future__ import annotations

import random
import unittest
from collections import Counter

from simulator.config.access_role_transformation import (
    ACCESS_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
)
from simulator.config.opportunity_pool import OpportunityPoolConfiguration
from simulator.config.scientific_population import ScientificPopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import field_manifest
from simulator.hard_negative import QualificationStatus
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.access_boundary import (
    ACCESS_ANOMALY_ADAPTER_VERSION,
    ACCESS_EVIDENCE_RESOLVER_VERSION,
    ACCESS_HN_ADAPTER_VERSION,
    ACCESS_HN_INSERTION_POLICY_VERSION,
    ACCESS_MATCHED_BOUNDARY_REFERENCE_VERSION,
    ACCESS_ROLE_LOCAL_TIMING_VERSION,
    ACCESS_TARGET_SELECTION_POLICY_VERSION,
    ACCESS_TRANSFORMATION_PROVENANCE_VERSION,
    ANOMALY_VARIANTS,
    HN_VARIANTS,
    AccessRoleTransformationRequest,
    integrate_access_role,
    matched_boundary_reference,
    resolve_access_operations,
    select_access_target,
)
from simulator.role_integration.models import canonical_json
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
    return AccessRoleTransformationRequest(base.base_trace_reference, variant, seed)


def _successful(records):
    return [record for record in records if (
        record["event_type"] == "access_granted"
        and record["result"] == "success"
        and record["resource_operation_outcome"] == "success"
    )]


class AccessBoundaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()
        cls.access_bases = [
            base for base in cls.pool.bases if resolve_access_operations(base)
        ]
        cls.results = {
            (base.base_trace_reference, variant): integrate_access_role(
                base, _request(base, variant),
            )
            for base in cls.pool.bases
            for variant in HN_VARIANTS + ANOMALY_VARIANTS
        }

    def test_versions_and_exact_scope(self):
        self.assertEqual(6, len(HN_VARIANTS))
        self.assertEqual(6, len(ANOMALY_VARIANTS))
        for version in (
            ACCESS_ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
            ACCESS_EVIDENCE_RESOLVER_VERSION,
            ACCESS_TARGET_SELECTION_POLICY_VERSION,
            ACCESS_HN_INSERTION_POLICY_VERSION,
            ACCESS_HN_ADAPTER_VERSION,
            ACCESS_ANOMALY_ADAPTER_VERSION,
            ACCESS_ROLE_LOCAL_TIMING_VERSION,
            ACCESS_TRANSFORMATION_PROVENANCE_VERSION,
            ACCESS_MATCHED_BOUNDARY_REFERENCE_VERSION,
        ):
            self.assertTrue(version.startswith("4d3.3-"))

    def test_characterization_applicability_and_statuses(self):
        self.assertEqual(48, len(self.pool.bases))
        self.assertEqual(11, len(self.access_bases))
        self.assertEqual({"DIRECT", "ONE_RETRY"}, {
            base.structural_summary.recovery for base in self.access_bases
        })
        self.assertEqual({"NO_RENEWAL", "RENEWAL"}, {
            base.structural_summary.renewal for base in self.access_bases
        })
        self.assertEqual({1, 3, 4}, {
            base.structural_summary.activity_count for base in self.access_bases
        })
        for variant in HN_VARIANTS + ANOMALY_VARIANTS:
            counts = Counter(
                self.results[(base.base_trace_reference, variant)].status
                for base in self.pool.bases
            )
            self.assertEqual(11, counts[TransformationStatus.APPLIED], variant)
            self.assertEqual(37, counts[TransformationStatus.NOT_APPLICABLE], variant)
            self.assertEqual(0, counts[TransformationStatus.UNSUPPORTED], variant)
            self.assertEqual(0, counts[TransformationStatus.CONSTRUCTION_ERROR], variant)

    def test_hns_preserve_original_successes_validate_and_qualify(self):
        for base in self.access_bases:
            base_snapshot = canonical_json(base.to_record())
            base_success_ids = [item["event_id"] for item in _successful(base.event_records)]
            for variant in HN_VARIANTS:
                derived = self.results[(base.base_trace_reference, variant)]
                derived_by_id = {item["event_id"]: item for item in derived.event_records}
                self.assertEqual(base_success_ids, [
                    item["event_id"] for item in _successful(derived.event_records)
                    if item["event_id"] in set(base_success_ids)
                ])
                for record in base.event_records:
                    self.assertEqual(record, derived_by_id[record["event_id"]])
                self.assertEqual(len(base_success_ids), len(_successful(derived.event_records)))
                self.assertEqual(len(base_success_ids) + 1, sum(
                    item["event_type"] == "access_request"
                    for item in derived.event_records
                ))
                self.assertEqual("validated_normal", derived.validation_status)
                self.assertEqual((), derived.validated_invariant_ids)
                self.assertTrue(derived.validation_build_gate_passed)
                self.assertEqual(
                    QualificationStatus.QUALIFIED,
                    derived.hard_negative_qualification.qualification_status,
                )
                self.assertEqual("qualified", derived.transformation_provenance["qualification_status"])
            self.assertEqual(base_snapshot, canonical_json(base.to_record()))

    def test_anomalies_mutate_only_one_canonical_request_relation(self):
        expected = {
            "access_success_without_auth_context": {"INV-ACCESS-AUTH-CONTEXT"},
            "access_success_without_valid_credential": {"INV-ACCESS-CREDENTIAL"},
            "access_success_without_active_session": {"INV-ACCESS-ACTIVE-SESSION"},
            "access_success_without_applicable_grant": {"INV-ACCESS-AUTHZ-GRANT"},
            "access_success_despite_explicit_deny": {
                "INV-ACCESS-AUTHZ-GRANT", "INV-ACCESS-DENY-ENFORCED",
            },
            "access_success_without_security_prerequisites": {
                "INV-ACCESS-AUTH-CONTEXT", "INV-ACCESS-CREDENTIAL",
                "INV-ACCESS-ACTIVE-SESSION",
            },
        }
        metadata = {
            "injection_id", "anomaly_variant_name", "invariant_families",
            "source_context", "anomaly_label",
        }
        for base in self.access_bases:
            target = select_access_target(base)
            self.assertIsNotNone(target)
            target_indices = {target.request_ordinal - 1, target.operation_ordinal - 1}
            for variant in ANOMALY_VARIANTS:
                derived = self.results[(base.base_trace_reference, variant)]
                self.assertEqual(len(base.event_records), len(derived.event_records))
                self.assertEqual(expected[variant], set(derived.validated_invariant_ids))
                self.assertEqual("validated_positive", derived.validation_status)
                self.assertTrue(derived.validation_build_gate_passed)
                self.assertEqual(len(_successful(base.event_records)),
                                 len(_successful(derived.event_records)))
                for index, original in enumerate(base.event_records):
                    transformed = derived.event_records[index]
                    if index not in target_indices:
                        self.assertEqual(original, transformed)
                    else:
                        for key, value in original.items():
                            if key in metadata:
                                continue
                            # Targeted fields differ by variant; every other
                            # request relation and every event remain exact.
                            if transformed.get(key) != value:
                                self.assertIn(key, {
                                    "authenticated_context_active", "authenticated_identity",
                                    "token_context_active", "token_validation_result",
                                    "protected_session_active", "authorization_decision",
                                    "authorization_evidence_complete",
                                    "authorization_authoritative",
                                })

    def test_target_policy_pairing_rng_isolation_and_base_immutability(self):
        base = self.access_bases[-1]
        first = select_access_target(base)
        self.assertEqual(resolve_access_operations(base)[0], first)
        snapshot = canonical_json(base.to_record())
        state = random.getstate()
        one = integrate_access_role(base, _request(base, ANOMALY_VARIANTS[0], 77))
        two = integrate_access_role(base, _request(base, ANOMALY_VARIANTS[0], 77))
        self.assertEqual(one, two)
        self.assertEqual(state, random.getstate())
        self.assertEqual(snapshot, canonical_json(base.to_record()))
        for hn, anomaly in zip(HN_VARIANTS[:5], ANOMALY_VARIANTS[:5]):
            left = self.results[(base.base_trace_reference, hn)]
            right = self.results[(base.base_trace_reference, anomaly)]
            expected_pair = matched_boundary_reference(base.base_trace_reference, hn)
            self.assertEqual(expected_pair, left.transformation_provenance["matched_boundary_reference"])
            self.assertEqual(expected_pair, right.transformation_provenance["matched_boundary_reference"])

    def test_existing_timestamps_and_profile_structure_are_preserved(self):
        for base in self.access_bases:
            base_times = {item["event_id"]: item["timestamp"] for item in base.event_records}
            for variant in HN_VARIANTS + ANOMALY_VARIANTS:
                derived = self.results[(base.base_trace_reference, variant)]
                derived_times = {item["event_id"]: item["timestamp"] for item in derived.event_records}
                self.assertEqual(base_times, {
                    key: derived_times[key] for key in base_times
                })
                self.assertEqual(base.behavioral_profile_id,
                                 base.structural_summary.behavioral_profile_id)
                self.assertEqual(base.structural_summary.activity_count,
                                 len(_successful(derived.event_records)))

    def test_detector_and_legacy_dispatch_boundaries(self):
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        detector = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        forbidden = {
            "base_trace_reference", "selected_request_reference",
            "added_request_id", "variant_id", "qualification_status",
            "matched_boundary_reference", "behavioral_profile_id",
            "population_device_key",
        }
        self.assertTrue(forbidden.isdisjoint(detector))
        non_access = next(base for base in self.pool.bases if not resolve_access_operations(base))
        unsupported = integrate_access_role(
            self.access_bases[0], _request(self.access_bases[0], "nonce_reuse"),
        )
        self.assertEqual(TransformationStatus.UNSUPPORTED, unsupported.status)
        not_applicable = integrate_access_role(
            non_access, _request(non_access, HN_VARIANTS[0]),
        )
        self.assertEqual(TransformationStatus.NOT_APPLICABLE, not_applicable.status)


if __name__ == "__main__":
    unittest.main()
