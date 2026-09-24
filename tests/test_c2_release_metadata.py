"""C2.6 release metadata, capability, and information-boundary gates."""

import unittest

from simulator.anomaly_contract import (
    SYNCHRONIZED_GENERATION_PROFILE,
    synchronized_capability_report,
)
from simulator.config.settings import cfg
from simulator.data.dataset_contract import (
    DATASET_INTERFACE_VERSION,
    DATASET_MANIFEST_VERSION,
    DATASET_VERSION,
    FIELD_MANIFEST_VERSION,
    dataset_capability_manifest,
    field_manifest,
)
from simulator.invariant_registry import (
    IDENTITY_RELATION_CAPABILITIES,
    INVARIANT_REGISTRY,
    TOKEN_BINDING_RELATION_CAPABILITIES,
)
from simulator.provenance import (
    BEHAVIOR_MODEL_VERSION,
    EVENT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    GENERATOR_SOFTWARE_VERSION,
    build_generation_provenance,
)
from simulator.validation import (
    EVENT_GROUND_TRUTH_SCHEMA_VERSION,
    GROUND_TRUTH_SCHEMA_VERSION,
    TRACE_GROUND_TRUTH_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    VALIDATOR_VERSION,
)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.provenance = build_generation_provenance(
            cfg,
            run_id="sync-v1.1.0-release-test",
            generator_git_commit="release-test-commit",
            generator_git_dirty=False,
        )
        self.manifest = dataset_capability_manifest(
            self.provenance,
            generation_profiles=(SYNCHRONIZED_GENERATION_PROFILE,),
        )

    def test_release_identifiers_are_final_and_feature_schema_is_unchanged(self):
        final_identifiers = (
            GENERATOR_SOFTWARE_VERSION,
            EVENT_SCHEMA_VERSION,
            VALIDATOR_VERSION,
            VALIDATION_SCHEMA_VERSION,
            EVENT_GROUND_TRUTH_SCHEMA_VERSION,
            TRACE_GROUND_TRUTH_SCHEMA_VERSION,
            GROUND_TRUTH_SCHEMA_VERSION,
            FIELD_MANIFEST_VERSION,
            DATASET_INTERFACE_VERSION,
            DATASET_MANIFEST_VERSION,
            DATASET_VERSION,
            SYNCHRONIZED_GENERATION_PROFILE,
        )
        self.assertTrue(all("sync-v1.1.0" in item for item in final_identifiers))
        self.assertTrue(all("development" not in item for item in final_identifiers))
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)

    def test_release_provenance_is_complete_and_content_addressed(self):
        record = self.provenance.to_dict()
        required = {
            "behavior_model_version",
            "generator_software_version",
            "generator_git_commit",
            "generator_git_dirty",
            "generator_git_state_scope",
            "event_schema_version",
            "feature_schema_version",
            "event_ground_truth_schema_version",
            "trace_ground_truth_schema_version",
            "validation_schema_version",
            "field_manifest_version",
            "dataset_interface_version",
            "dataset_manifest_version",
            "dataset_version",
            "invariant_registry_version",
            "run_id",
            "configuration_snapshot",
            "configuration_snapshot_reference",
            "randomness_contract",
            "anomaly_capability_contract",
            "temporal_contract",
        }
        self.assertTrue(required.issubset(record))
        self.assertEqual(BEHAVIOR_MODEL_VERSION, record["behavior_model_version"])
        self.assertTrue(record["configuration_snapshot_reference"].startswith("sha256:"))
        self.assertIsNotNone(record["configuration_snapshot"]["simulation"]["random_seed"])
        self.assertFalse(record["randomness_contract"]["byte_identical_reproducibility"])

    def test_manifest_versions_match_run_provenance(self):
        pairs = {
            "dataset_interface_version": "dataset_interface_version",
            "dataset_version": "dataset_version",
            "invariant_registry_version": "invariant_registry_version",
            "event_schema_version": "event_schema_version",
            "feature_schema_version": "feature_schema_version",
            "generator_software_version": "generator_software_version",
            "generator_git_commit": "generator_git_commit",
            "configuration_version": "configuration_version",
            "configuration_snapshot_reference": "configuration_snapshot_reference",
        }
        for manifest_name, provenance_name in pairs.items():
            self.assertEqual(
                getattr(self.provenance, provenance_name),
                self.manifest[manifest_name],
            )
        self.assertEqual(DATASET_MANIFEST_VERSION, self.manifest["manifest_version"])
        self.assertEqual(FIELD_MANIFEST_VERSION, self.manifest["field_manifest_version"])

    def test_registry_release_boundary_is_nine_exact_eleven_partial(self):
        exact = self.manifest["exact_supported_invariant_ids"]
        partial = self.manifest["partial_invariant_capabilities"]
        self.assertEqual(20, len(INVARIANT_REGISTRY))
        self.assertEqual(9, len(exact))
        self.assertEqual(11, len(partial))
        self.assertEqual(set(INVARIANT_REGISTRY), set(exact) | set(partial))

    def test_relation_level_capabilities_remain_unflattened(self):
        self.assertEqual(
            "exact",
            IDENTITY_RELATION_CAPABILITIES[
                "persistent_identity_to_authenticated_identity"
            ]["status"],
        )
        self.assertEqual(
            "unsupported_by_current_evidence",
            IDENTITY_RELATION_CAPABILITIES[
                "authenticated_identity_to_request_acting_identity"
            ]["status"],
        )
        self.assertEqual(
            "exact",
            TOKEN_BINDING_RELATION_CAPABILITIES[
                "token_bound_identity_to_authenticated_identity"
            ]["status"],
        )
        self.assertEqual(
            "exact",
            TOKEN_BINDING_RELATION_CAPABILITIES[
                "token_bound_protected_context_to_active_context"
            ]["status"],
        )
        self.assertEqual(
            "unsupported_by_current_evidence",
            TOKEN_BINDING_RELATION_CAPABILITIES["issuer_applicability"]["status"],
        )
        self.assertEqual(
            "partial",
            TOKEN_BINDING_RELATION_CAPABILITIES[
                "resource_action_scope_applicability"
            ]["status"],
        )

    def test_each_supported_variant_has_scope_evidence_checks_and_provenance(self):
        report = synchronized_capability_report()
        self.assertEqual(
            "independent_atomic_validator_evidence_only",
            report["ground_truth_derivation"],
        )
        self.assertIn("canonical_invariant_ids", report["injection_provenance_contract"])
        for variant in report["supported_variants"]:
            item = report["variants"][variant]
            self.assertTrue(item["invariant_ids"], variant)
            self.assertTrue(item["history_scope"], variant)
            self.assertEqual(set(item["invariant_ids"]), set(item["required_evidence_by_invariant"]))
            self.assertEqual(set(item["invariant_ids"]), set(item["validator_checks_by_invariant"]))
            self.assertTrue(all(item["validator_checks_by_invariant"].values()))

    def test_unsupported_release_capabilities_are_explicit(self):
        unsupported = set(
            self.manifest["explicitly_unsupported_or_out_of_scope_capabilities"]
        )
        required = {
            "independent_request_level_acting_identity",
            "token_issuer_applicability",
            "universal_resource_action_token_binding",
            "cross_session_token_replay",
            "sequence_replay_or_duplication",
            "statistical_failure_rate_anomalies",
            "statistical_renewal_anomalies",
            "global_discrete_event_scheduling_or_concurrency",
            "compromised_gateway_or_device_semantics",
            "operational_monitoring_point_mapping",
            "final_detector_feature_or_prediction_horizon_decisions",
        }
        self.assertTrue(required.issubset(unsupported))

    def test_detector_manifest_excludes_all_privileged_release_fields(self):
        detector = {
            item["field_name"] for item in field_manifest()["fields"]
            if item["export_view"] == "detector_observations"
        }
        forbidden = {
            "validated_invariant_ids",
            "validated_invariant_evidence",
            "scenario_intent",
            "injection_records",
            "authenticated_identity",
            "device_id",
            "token_bound_identity",
            "token_bound_protected_session_id",
            "semantic_timestamp",
            "validation_status",
        }
        self.assertTrue(detector.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
