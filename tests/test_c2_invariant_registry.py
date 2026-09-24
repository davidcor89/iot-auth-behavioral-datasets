"""C2.0 tests for the canonical Behavioral-model v1.1 invariant registry."""

import json
import random
import re
import unittest
from unittest.mock import patch

from simulator.anomaly_contract import (
    SYNCHRONIZED_GENERATION_PROFILE,
    InjectionRecord,
)
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.invariant_registry import (
    EXACT_VALIDATOR_CHECK_TO_INVARIANT,
    EXACT_VARIANT_TO_INVARIANTS,
    IDENTITY_RELATION_CAPABILITIES,
    INVARIANT_REGISTRY,
    INVARIANT_REGISTRY_VERSION,
    ImplementationStatus,
    TOKEN_BINDING_RELATION_CAPABILITIES,
    invariant_registry_report,
)
from simulator.provenance import BEHAVIOR_MODEL_VERSION, build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import validate_synchronized_dataset


EXPECTED_INVARIANT_IDS = {
    "INV-AUTH-FAIL-NO-CONTEXT",
    "INV-AUTH-CONTEXT-REQUIRES-SUCCESS",
    "INV-LIFE-TOKEN-AFTER-AUTH",
    "INV-LIFE-TOKEN-USE-AFTER-ESTABLISH",
    "INV-LIFE-SESSION-ESTABLISH-PRECONDITIONS",
    "INV-TIME-CAUSAL-ORDER",
    "INV-TIME-TOKEN-VALIDITY",
    "INV-TIME-OBSERVED-CAUSAL",
    "INV-SESS-ENROLLMENT-PERSISTS",
    "INV-SESS-ACTIVE-COHERENCE",
    "INV-SESS-RENEWAL-CONTINUITY",
    "INV-REPLAY-NONCE-CROSS-ATTEMPT",
    "INV-REPLAY-TOKEN-SUPERSEDED",
    "INV-ACCESS-AUTH-CONTEXT",
    "INV-ACCESS-CREDENTIAL",
    "INV-ACCESS-ACTIVE-SESSION",
    "INV-ACCESS-AUTHZ-GRANT",
    "INV-ACCESS-DENY-ENFORCED",
    "INV-IDENT-AUTH-ACTOR-CONSISTENCY",
    "INV-BIND-TOKEN-CONTEXT",
}


def execute_variant(name, *, seed):
    random.seed(seed)
    spec = ScenarioEngine(seed).synchronized_anomaly(name)
    persistent = PersistentDeviceContext(device_id=f"device-{name}")
    return EventEngine(
        device_id=persistent.device_id,
        gateway_id="gateway-c2",
        auth_server_id="auth-c2",
        start_time=40_000.0,
        persistent_context=persistent,
    ).execute(spec)


def validate(pair):
    provenance = build_generation_provenance(
        cfg, run_id="c2-registry-test", generator_git_commit="c2-test-commit"
    )
    return validate_synchronized_dataset(
        [pair],
        provenance=provenance,
        field_manifest=field_manifest(),
        dataset_manifest=dataset_capability_manifest(
            provenance,
            generation_profiles=(pair[1].generation_profile,),
        ),
    )


class InvariantRegistryTests(unittest.TestCase):
    def test_registry_contains_exactly_the_adopted_stable_ids(self):
        self.assertEqual(EXPECTED_INVARIANT_IDS, set(INVARIANT_REGISTRY))
        self.assertEqual(len(INVARIANT_REGISTRY), len(set(INVARIANT_REGISTRY)))
        self.assertTrue(
            all(re.fullmatch(r"INV-[A-Z]+-[A-Z0-9-]+", item) for item in INVARIANT_REGISTRY)
        )

    def test_every_registry_entry_has_complete_machine_readable_metadata(self):
        report = invariant_registry_report()
        self.assertEqual(INVARIANT_REGISTRY_VERSION, report["registry_version"])
        self.assertEqual("behavioral-model-v1.1", report["behavior_model_version"])
        self.assertEqual(BEHAVIOR_MODEL_VERSION, report["behavior_model_version"])
        for item in report["invariants"]:
            for field in (
                "invariant_id", "short_name", "semantic_family", "scope",
                "decidability", "implementation_status", "version_introduced",
                "minimum_evidence",
            ):
                self.assertTrue(item[field], (item["invariant_id"], field))
        json.dumps(report)

    def test_statuses_are_conservative_and_token_supersession_is_exact(self):
        exact = {
            key for key, item in INVARIANT_REGISTRY.items()
            if item.implementation_status is ImplementationStatus.EXACT
        }
        self.assertEqual(
            {
                "INV-TIME-OBSERVED-CAUSAL",
                "INV-SESS-ENROLLMENT-PERSISTS",
                "INV-REPLAY-NONCE-CROSS-ATTEMPT",
                "INV-REPLAY-TOKEN-SUPERSEDED",
                "INV-ACCESS-AUTH-CONTEXT",
                "INV-ACCESS-CREDENTIAL",
                "INV-ACCESS-ACTIVE-SESSION",
                "INV-ACCESS-AUTHZ-GRANT",
                "INV-ACCESS-DENY-ENFORCED",
            },
            exact,
        )
        self.assertEqual(
            ImplementationStatus.EXACT,
            INVARIANT_REGISTRY["INV-REPLAY-TOKEN-SUPERSEDED"].implementation_status,
        )
        self.assertEqual(
            ImplementationStatus.PARTIAL,
            INVARIANT_REGISTRY[
                "INV-IDENT-AUTH-ACTOR-CONSISTENCY"
            ].implementation_status,
        )

    def test_scenario_names_are_not_invariant_ids(self):
        historical_or_composite = {
            "access_without_auth", "impersonation", "identity_token_mismatch",
            "replay_token", "nonce_reuse", "timestamp_inconsistency",
        }
        self.assertTrue(historical_or_composite.isdisjoint(INVARIANT_REGISTRY))
        self.assertFalse(invariant_registry_report()["legacy_scenario_names_are_invariant_ids"])
        self.assertEqual(
            {
                "nonce_reuse",
                "timestamp_inconsistency",
                "token_replay_after_replacement",
                "access_success_without_auth_context",
                "access_success_without_valid_credential",
                "access_success_without_active_session",
                "access_success_without_applicable_grant",
                "access_success_despite_explicit_deny",
                "access_success_without_security_prerequisites",
                "authenticated_identity_mismatch_persistent_device",
                "token_bound_identity_mismatch_authenticated_context",
            },
            set(EXACT_VARIANT_TO_INVARIANTS),
        )

    def test_identity_relation_capability_is_exact_only_for_supported_form(self):
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

    def test_token_binding_capabilities_preserve_relation_level_limits(self):
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

    def test_exact_variant_records_and_validator_results_carry_registry_ids(self):
        for index, (variant, expected_ids) in enumerate(EXACT_VARIANT_TO_INVARIANTS.items()):
            with self.subTest(variant=variant), patch.dict(
                STEALTH_FRACTION, {variant: 0.0}
            ):
                pair = execute_variant(variant, seed=2200 + index)
                record = pair[1].injection_records[0]
                self.assertEqual(expected_ids, tuple(record["canonical_invariant_ids"]))
                report = validate(pair)
                mapped_check = next(
                    check for check, invariant_id in EXACT_VALIDATOR_CHECK_TO_INVARIANT.items()
                    if invariant_id in expected_ids
                )
                validation_record = next(
                    item for item in report.records if item.validation_id == mapped_check
                )
                self.assertEqual(expected_ids[0], validation_record.invariant_identifier)
                if variant == "timestamp_inconsistency":
                    temporal = next(
                        item for item in report.records
                        if item.validation_id == "c1.7.temporal.supported_subset"
                    )
                    self.assertIsNone(temporal.invariant_identifier)

    def test_injection_provenance_can_represent_multiple_invariant_ids(self):
        record = InjectionRecord(
            injection_id="injection",
            historical_scenario_name="scenario-provenance-only",
            anomaly_variant_name="variant",
            capability_status="test",
            mapped_semantic_concept="test concept",
            invariant_families=("access",),
            canonical_invariant_ids=(
                "INV-ACCESS-AUTH-CONTEXT",
                "INV-ACCESS-CREDENTIAL",
            ),
        )
        self.assertEqual(2, len(record.to_dict()["canonical_invariant_ids"]))

    def test_dataset_manifest_embeds_registry_version_and_c2_profile(self):
        provenance = build_generation_provenance(
            cfg, run_id="manifest-test", generator_git_commit="c2-test"
        )
        manifest = dataset_capability_manifest(
            provenance,
            generation_profiles=(SYNCHRONIZED_GENERATION_PROFILE,),
        )
        self.assertEqual(INVARIANT_REGISTRY_VERSION, manifest["invariant_registry_version"])
        self.assertEqual(SYNCHRONIZED_GENERATION_PROFILE, manifest["generation_profile"])
        self.assertEqual("sync-v1.1.0", manifest["generator_software_version"])


if __name__ == "__main__":
    unittest.main()
