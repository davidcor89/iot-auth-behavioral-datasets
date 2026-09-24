"""Focused 4D.3.2 direct-evidence and timing-boundary integration tests."""

from __future__ import annotations

import json
import random
import unittest
from collections import Counter
from dataclasses import replace

from simulator.config.opportunity_pool import OpportunityPoolConfiguration
from simulator.config.role_transformation import (
    ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
    DirectTimingTransformationConfiguration,
)
from simulator.config.scientific_population import ScientificPopulationConfiguration
from simulator.config.settings import cfg
from simulator.data.dataset_contract import field_manifest
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.direct_timing import (
    COUNTERFACTUAL_QA_REFERENCE_VERSION,
    DERIVED_TRACE_REFERENCE_VERSION,
    DIRECT_REFERENCE_ADAPTER_VERSION,
    IDENTITY_COHERENT_HN,
    POSITIVE_MARGIN_HN,
    SAME_ATTEMPT_NONCE_HN,
    SUPPORTED_ROLE_VARIANTS,
    TARGET_SELECTION_POLICY_VERSION,
    TIMESTAMP_ANOMALY,
    TIMING_BOUNDARY_ADAPTER_VERSION,
    TIMING_TRANSFORMATION_PROVENANCE_VERSION,
    RoleTransformationRequest,
    counterfactual_group_reference,
    integrate_role,
    materialize_base,
    resolve_timing_relation,
)
from simulator.hard_negative import QualificationStatus
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
    return RoleTransformationRequest(base.base_trace_reference, variant, seed)


class DirectTimingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()

    def test_version_checkpoint_and_four_variant_scope(self):
        self.assertEqual(4, len(SUPPORTED_ROLE_VARIANTS))
        self.assertEqual({
            SAME_ATTEMPT_NONCE_HN, IDENTITY_COHERENT_HN,
            POSITIVE_MARGIN_HN, TIMESTAMP_ANOMALY,
        }, set(SUPPORTED_ROLE_VARIANTS))
        for value in (
            DIRECT_REFERENCE_ADAPTER_VERSION, TIMING_BOUNDARY_ADAPTER_VERSION,
            TARGET_SELECTION_POLICY_VERSION, DERIVED_TRACE_REFERENCE_VERSION,
            TIMING_TRANSFORMATION_PROVENANCE_VERSION,
            COUNTERFACTUAL_QA_REFERENCE_VERSION,
            ROLE_TRANSFORMATION_CONFIGURATION_VERSION,
        ):
            self.assertTrue(value.startswith("4d3.2-"))

    def test_reference_only_hns_preserve_base_and_qualify(self):
        for base in self.pool.bases:
            before = base.to_record()
            for variant in (SAME_ATTEMPT_NONCE_HN, IDENTITY_COHERENT_HN):
                derived = integrate_role(base, _request(base, variant))
                self.assertEqual(TransformationStatus.APPLIED, derived.status)
                self.assertEqual(base.event_records_json, derived.event_records_json)
                self.assertEqual(base.session_context_json, derived.session_context_json)
                self.assertEqual("validated_normal", derived.validation_status)
                self.assertTrue(derived.validation_build_gate_passed)
                self.assertEqual((), derived.validated_invariant_ids)
                self.assertEqual(
                    QualificationStatus.QUALIFIED,
                    derived.hard_negative_qualification.qualification_status,
                )
                self.assertEqual(base.base_trace_reference, derived.base_trace_reference)
            self.assertEqual(before, base.to_record())

    def test_timing_pair_changes_only_target_observed_evidence(self):
        for base in self.pool.bases:
            relation = resolve_timing_relation(base)
            self.assertIsNotNone(relation)
            hn = integrate_role(base, _request(base, POSITIVE_MARGIN_HN))
            anomaly = integrate_role(base, _request(base, TIMESTAMP_ANOMALY))
            self.assertEqual(TransformationStatus.APPLIED, hn.status)
            self.assertEqual(TransformationStatus.APPLIED, anomaly.status)
            self.assertEqual(
                hn.transformation_provenance["counterfactual_group_reference"],
                anomaly.transformation_provenance["counterfactual_group_reference"],
            )
            ordinal = relation.target_ordinal - 1
            base_records = list(base.event_records)
            hn_records = list(hn.event_records)
            anomaly_records = list(anomaly.event_records)
            for index in range(len(base_records)):
                if index == ordinal:
                    continue
                self.assertEqual(base_records[index], hn_records[index])
                self.assertEqual(base_records[index], anomaly_records[index])
            # Semantic timestamps and all non-observed target evidence persist.
            ignored = {
                "observed_timestamp", "observed_timestamp_source",
                "targeted_temporal_relationship", "injection_id",
                "anomaly_variant_name", "invariant_families",
                "source_context", "anomaly_label",
            }
            for key, value in base_records[ordinal].items():
                if key not in ignored:
                    self.assertEqual(value, hn_records[ordinal][key])
                    self.assertEqual(value, anomaly_records[ordinal][key])
            hn_margin = (
                hn_records[ordinal]["observed_timestamp"]
                - relation.predecessor_observed_timestamp
            )
            anomaly_margin = (
                anomaly_records[ordinal]["observed_timestamp"]
                - relation.predecessor_observed_timestamp
            )
            self.assertGreater(hn_margin, 0)
            self.assertLessEqual(hn_margin, _request(base, POSITIVE_MARGIN_HN).configuration.epsilon_hn_s)
            self.assertLess(anomaly_margin, 0)
            self.assertEqual("validated_normal", hn.validation_status)
            self.assertEqual(QualificationStatus.QUALIFIED,
                             hn.hard_negative_qualification.qualification_status)
            self.assertEqual("validated_positive", anomaly.validation_status)
            self.assertEqual(("INV-TIME-OBSERVED-CAUSAL",), anomaly.validated_invariant_ids)
            self.assertTrue(hn.validation_build_gate_passed)
            self.assertTrue(anomaly.validation_build_gate_passed)

    def test_target_policy_is_first_canonical_pair_and_role_independent(self):
        for base in self.pool.bases:
            relation = resolve_timing_relation(base)
            events = base.event_records
            eligible = []
            for left, predecessor in enumerate(events):
                for right, target in enumerate(events[left + 1:], left + 1):
                    if (
                        predecessor.get("event_type") == "nonce_received"
                        and target.get("event_type") == "response_sent"
                        and predecessor.get("auth_attempt_id") == target.get("auth_attempt_id")
                        and predecessor.get("nonce") == target.get("nonce")
                        and predecessor.get("nonce")
                        and predecessor["timestamp"] < target["timestamp"]
                    ):
                        eligible.append((left + 1, right + 1))
            self.assertEqual(eligible[0], (
                relation.predecessor_ordinal, relation.target_ordinal,
            ))
            hn = integrate_role(base, _request(base, POSITIVE_MARGIN_HN))
            anomaly = integrate_role(base, _request(base, TIMESTAMP_ANOMALY))
            self.assertEqual(
                hn.transformation_provenance["predecessor_event_reference"],
                anomaly.transformation_provenance["predecessor_event_reference"],
            )
            self.assertEqual(
                hn.transformation_provenance["target_event_reference"],
                anomaly.transformation_provenance["target_event_reference"],
            )

    def test_reproducibility_seed_isolation_and_base_immutability(self):
        base = self.pool.bases[-1]
        snapshot = canonical_json(base.to_record())
        first = integrate_role(base, _request(base, POSITIVE_MARGIN_HN, 123))
        second = integrate_role(base, _request(base, POSITIVE_MARGIN_HN, 123))
        changed = integrate_role(base, _request(base, POSITIVE_MARGIN_HN, 124))
        self.assertEqual(first, second)
        self.assertNotEqual(
            first.transformation_provenance["transformed_observed_margin_s"],
            changed.transformation_provenance["transformed_observed_margin_s"],
        )
        state = random.getstate()
        integrate_role(base, _request(base, TIMESTAMP_ANOMALY, 999))
        self.assertEqual(state, random.getstate())
        self.assertEqual(snapshot, canonical_json(base.to_record()))
        self.assertEqual(
            counterfactual_group_reference(base.base_trace_reference, _request(base, POSITIVE_MARGIN_HN).configuration),
            first.transformation_provenance["counterfactual_group_reference"],
        )

    def test_structure_is_preserved_across_available_depths_and_choices(self):
        seen = {
            "lifecycle": Counter(), "recovery": Counter(),
            "renewal": Counter(), "activity": Counter(),
        }
        for base in self.pool.bases:
            before = base.structural_summary
            derived = integrate_role(base, _request(base, TIMESTAMP_ANOMALY))
            self.assertEqual(len(base.event_records), len(derived.event_records))
            seen["lifecycle"][before.lifecycle] += 1
            seen["recovery"][before.recovery] += 1
            seen["renewal"][before.renewal] += 1
            seen["activity"][before.activity_count] += 1
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, set(seen["lifecycle"]))
        self.assertEqual({"DIRECT", "ONE_RETRY"}, set(seen["recovery"]))
        self.assertEqual({"NO_RENEWAL", "RENEWAL"}, set(seen["renewal"]))
        self.assertGreater(len(seen["activity"]), 1)

    def test_failure_statuses_do_not_mutate_or_backfill(self):
        base = self.pool.bases[0]
        before = base.to_record()
        unsupported = integrate_role(base, _request(base, "other_variant"))
        self.assertEqual(TransformationStatus.UNSUPPORTED, unsupported.status)
        self.assertEqual(base.event_records_json, unsupported.event_records_json)

        without_nonce = tuple(
            canonical_json({**record, "nonce": None}) for record in base.event_records
        )
        inapplicable_base = replace(base, event_records_json=without_nonce)
        inapplicable = integrate_role(
            inapplicable_base,
            RoleTransformationRequest(
                inapplicable_base.base_trace_reference, SAME_ATTEMPT_NONCE_HN, 4304,
            ),
        )
        self.assertEqual(TransformationStatus.NOT_APPLICABLE, inapplicable.status)
        self.assertEqual(without_nonce, inapplicable.event_records_json)
        self.assertEqual(before, base.to_record())

    def test_no_role_or_profile_metadata_enters_detector_manifest(self):
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        detector_names = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        forbidden = {
            "base_trace_reference", "derived_trace_reference", "variant_id",
            "applicability", "qualification_status", "counterfactual_group_reference",
            "behavioral_profile_id", "population_device_key",
        }
        self.assertTrue(forbidden.isdisjoint(detector_names))
        events, context = materialize_base(self.pool.bases[0])
        self.assertTrue(events)
        self.assertEqual(self.pool.bases[0].device_id, context.device_id)

    def test_canonical_admission_pool_fingerprint_is_stable(self):
        self.assertEqual(
            "sha256:2ce2a58a1511b0da3df32f1dc5a5f692d4e2f1f876c593ad1302bf456f9c2436",
            self.pool.pool_fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
