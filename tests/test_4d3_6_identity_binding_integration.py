"""Focused 4D.3.6 identity and token-binding integration tests."""

from __future__ import annotations

import json
import random
import unittest
from collections import Counter

from simulator.config.identity_binding_role_transformation import (
    IDENTITY_BINDING_TRANSFORMATION_CONFIGURATION_VERSION,
)
from simulator.data.dataset_contract import field_manifest
from simulator.hard_negative import QualificationStatus
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.role_integration.applicability import ApplicabilityPredicate
from simulator.role_integration.direct_timing import (
    DIRECT_REFERENCE_ADAPTER_VERSION,
    IDENTITY_COHERENT_HN,
    RoleTransformationRequest,
    integrate_role,
)
from simulator.role_integration.identity_binding_boundary import (
    AUTHENTICATED_IDENTITY_RESOLVER_VERSION,
    AUTHENTICATION_TARGET_POLICY_VERSION,
    BINDING_HN_ADAPTER_VERSION,
    IDENTITY_BINDING_DERIVED_REFERENCE_VERSION,
    IDENTITY_BINDING_MATCHED_REFERENCE_VERSION,
    IDENTITY_BINDING_PROVENANCE_VERSION,
    PERSISTENT_IDENTITY_ANOMALY,
    PERSISTENT_IDENTITY_ANOMALY_ADAPTER_VERSION,
    SUPPORTED_IDENTITY_BINDING_VARIANTS,
    TOKEN_BINDING_ANOMALY_ADAPTER_VERSION,
    TOKEN_BINDING_RESOLVER_VERSION,
    TOKEN_BINDING_TARGET_POLICY_VERSION,
    TOKEN_CONTEXT_COHERENT_HN,
    TOKEN_IDENTITY_ANOMALY,
    TOKEN_IDENTITY_COHERENT_HN,
    IdentityBindingRoleTransformationRequest,
    binding_matched_boundary_reference,
    identity_matched_boundary_reference,
    integrate_identity_binding_role,
    resolve_authenticated_identity,
    resolve_token_identity_binding,
    resolve_token_protected_context_binding,
)
from simulator.role_integration.transformation import TransformationStatus
from tests.test_4d3_4_nonce_boundary_integration import _pool


POOL_FINGERPRINT = (
    "sha256:2ce2a58a1511b0da3df32f1dc5a5f692d4e2f1f876c593ad1302bf456f9c2436"
)


def _request(base, variant, seed=4304):
    return IdentityBindingRoleTransformationRequest(
        base.base_trace_reference, variant, seed,
    )


class IdentityBindingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _pool()
        cls.identity = [
            base for base in cls.pool.bases
            if resolve_authenticated_identity(base) is not None
        ]
        cls.identity_binding = [
            base for base in cls.pool.bases
            if resolve_token_identity_binding(base) is not None
        ]
        cls.context_binding = [
            base for base in cls.pool.bases
            if resolve_token_protected_context_binding(base) is not None
        ]
        cls.results = {
            (base.base_trace_reference, variant): integrate_identity_binding_role(
                base, _request(base, variant),
            )
            for base in cls.pool.bases
            for variant in SUPPORTED_IDENTITY_BINDING_VARIANTS
        }

    def test_versions_and_broad_precise_applicability(self):
        versions = (
            IDENTITY_BINDING_TRANSFORMATION_CONFIGURATION_VERSION,
            AUTHENTICATED_IDENTITY_RESOLVER_VERSION,
            AUTHENTICATION_TARGET_POLICY_VERSION,
            TOKEN_BINDING_RESOLVER_VERSION,
            TOKEN_BINDING_TARGET_POLICY_VERSION,
            PERSISTENT_IDENTITY_ANOMALY_ADAPTER_VERSION,
            BINDING_HN_ADAPTER_VERSION,
            TOKEN_BINDING_ANOMALY_ADAPTER_VERSION,
            IDENTITY_BINDING_PROVENANCE_VERSION,
            IDENTITY_BINDING_DERIVED_REFERENCE_VERSION,
            IDENTITY_BINDING_MATCHED_REFERENCE_VERSION,
        )
        self.assertTrue(all(version.startswith("4d3.6-") for version in versions))
        self.assertEqual(POOL_FINGERPRINT, self.pool.pool_fingerprint)
        broad = Counter()
        for vector in self.pool.applicability_matrix:
            for item in vector:
                if item.applicable:
                    broad[item.predicate_id] += 1
        self.assertEqual(48, broad[
            ApplicabilityPredicate.PERSISTENT_IDENTITY_ASSOCIATION.value
        ])
        self.assertEqual(33, broad[
            ApplicabilityPredicate.TOKEN_AUTHENTICATED_IDENTITY_BINDING.value
        ])
        self.assertEqual(33, broad[
            ApplicabilityPredicate.TOKEN_PROTECTED_CONTEXT_BINDING.value
        ])
        self.assertEqual(48, len(self.identity))
        self.assertEqual(33, len(self.identity_binding))
        self.assertEqual(33, len(self.context_binding))
        self.assertEqual(16, len({base.device_id for base in self.identity}))
        self.assertEqual(14, len({base.device_id for base in self.identity_binding}))

    def test_canonical_selection_is_first_complete_and_role_independent(self):
        for base in self.identity:
            evidence = resolve_authenticated_identity(base)
            successes = [
                index + 1 for index, event in enumerate(base.event_records)
                if event["event_type"] == "authentication_success"
                and event["result"] == "success"
                and event.get("authenticated_identity") == base.device_id
            ]
            self.assertEqual(successes[0], evidence.authentication_ordinal)
        for base in self.identity_binding:
            identity = resolve_token_identity_binding(base)
            context = resolve_token_protected_context_binding(base)
            self.assertEqual("token_presented", identity.use_event_type)
            self.assertEqual("session_opened", context.use_event_type)
            self.assertEqual(identity.token_id, context.token_id)
            self.assertLess(identity.use_ordinal, context.use_ordinal)

    def test_binding_hns_are_reference_only_normal_and_qualified(self):
        applicable = {
            TOKEN_IDENTITY_COHERENT_HN: self.identity_binding,
            TOKEN_CONTEXT_COHERENT_HN: self.context_binding,
        }
        for variant, bases in applicable.items():
            for base in bases:
                result = self.results[(base.base_trace_reference, variant)]
                self.assertEqual(TransformationStatus.APPLIED, result.status)
                self.assertEqual(base.event_records_json, result.event_records_json)
                self.assertEqual(base.session_context_json, result.session_context_json)
                self.assertEqual("reference_only",
                                 result.transformation_provenance["construction_mode"])
                self.assertEqual([], result.transformation_provenance[
                    "transformed_event_references"
                ])
                self.assertEqual("validated_normal", result.validation_status)
                self.assertTrue(result.validation_build_gate_passed)
                self.assertEqual((), result.validated_invariant_ids)
                self.assertEqual(
                    QualificationStatus.QUALIFIED,
                    result.hard_negative_qualification.qualification_status,
                )

    def test_persistent_identity_anomaly_propagates_coherently(self):
        allowed = {
            "authenticated_identity", "token_bound_identity", "injection_id",
            "anomaly_variant_name", "invariant_families", "source_context",
            "anomaly_label",
        }
        lifecycle = Counter()
        recovery = Counter()
        renewal = Counter()
        collision = Counter()
        for base in self.identity:
            result = self.results[(
                base.base_trace_reference, PERSISTENT_IDENTITY_ANOMALY,
            )]
            self.assertEqual(TransformationStatus.APPLIED, result.status)
            expected = {"INV-IDENT-AUTH-ACTOR-CONSISTENCY"}
            if base.structural_summary.lifecycle == "ACCESS":
                expected.add("INV-ACCESS-AUTH-CONTEXT")
                self.assertEqual("validation_failed", result.validation_status)
                self.assertFalse(result.validation_build_gate_passed)
            else:
                self.assertEqual("validated_positive", result.validation_status)
                self.assertTrue(result.validation_build_gate_passed)
            self.assertEqual(expected, set(result.validated_invariant_ids))
            context = json.loads(result.session_context_json)
            self.assertEqual(base.device_id, context["device_id"])
            self.assertNotEqual(base.device_id, context["authenticated_identity"])
            for original, derived in zip(base.event_records, result.event_records):
                for key, value in original.items():
                    if derived.get(key) != value:
                        self.assertIn(key, allowed)
                if derived.get("token_bound_identity") is not None:
                    self.assertEqual(
                        derived.get("authenticated_identity"),
                        derived.get("token_bound_identity"),
                    )
            self.assertTrue(result.transformation_provenance[
                "protected_context_binding_preserved"
            ])
            self.assertEqual(
                result.validation_build_gate_passed,
                result.transformation_provenance["aggregate_build_gate_passed"],
            )
            self.assertEqual(
                ["INV-IDENT-AUTH-ACTOR-CONSISTENCY"],
                result.transformation_provenance[
                    "intended_validated_invariant_ids"
                ],
            )
            lifecycle[base.structural_summary.lifecycle] += 1
            recovery[base.structural_summary.recovery] += 1
            renewal[base.structural_summary.renewal] += 1
            collision.update(result.transformation_provenance[
                "additional_invariant_collisions"
            ])
        self.assertEqual({"AUTH", "SESSION", "ACCESS"}, set(lifecycle))
        self.assertEqual({"DIRECT", "ONE_RETRY"}, set(recovery))
        self.assertEqual({"NO_RENEWAL", "RENEWAL"}, set(renewal))
        self.assertEqual({"INV-ACCESS-AUTH-CONTEXT": 11}, dict(collision))

    def test_token_identity_anomaly_changes_only_selected_lineage_binding(self):
        allowed = {
            "token_bound_identity", "injection_id", "anomaly_variant_name",
            "invariant_families", "source_context", "anomaly_label",
        }
        for base in self.identity_binding:
            evidence = resolve_token_identity_binding(base)
            result = self.results[(base.base_trace_reference, TOKEN_IDENTITY_ANOMALY)]
            self.assertEqual(TransformationStatus.APPLIED, result.status)
            self.assertEqual("validated_positive", result.validation_status)
            self.assertTrue(result.validation_build_gate_passed)
            self.assertEqual(("INV-BIND-TOKEN-CONTEXT",),
                             result.validated_invariant_ids)
            for original, derived in zip(base.event_records, result.event_records):
                for key, value in original.items():
                    if derived.get(key) != value:
                        self.assertIn(key, allowed)
                self.assertEqual(original.get("authenticated_identity"),
                                 derived.get("authenticated_identity"))
                self.assertEqual(original.get("token_bound_protected_session_id"),
                                 derived.get("token_bound_protected_session_id"))
                self.assertEqual(original.get("token_id"), derived.get("token_id"))
                if (
                    original.get("token_id") != evidence.token_id
                    and original.get("token_bound_identity") is not None
                ):
                    self.assertEqual(original["token_bound_identity"],
                                     derived["token_bound_identity"])
            self.assertEqual([], result.transformation_provenance[
                "additional_invariant_collisions"
            ])

    def test_same_base_qa_references_and_existing_identity_hn_are_unchanged(self):
        for base in self.identity:
            existing = integrate_role(base, RoleTransformationRequest(
                base.base_trace_reference, IDENTITY_COHERENT_HN, 4304,
            ))
            anomaly = self.results[(
                base.base_trace_reference, PERSISTENT_IDENTITY_ANOMALY,
            )]
            self.assertEqual(DIRECT_REFERENCE_ADAPTER_VERSION,
                             existing.transformation_provenance["adapter_version"])
            self.assertEqual(base.event_records_json, existing.event_records_json)
            self.assertEqual(
                identity_matched_boundary_reference(base.base_trace_reference),
                anomaly.transformation_provenance["matched_boundary_reference"],
            )
        for base in self.identity_binding:
            hn = self.results[(base.base_trace_reference, TOKEN_IDENTITY_COHERENT_HN)]
            anomaly = self.results[(base.base_trace_reference, TOKEN_IDENTITY_ANOMALY)]
            expected = binding_matched_boundary_reference(base.base_trace_reference)
            self.assertEqual(expected, hn.transformation_provenance[
                "matched_boundary_reference"
            ])
            self.assertEqual(expected, anomaly.transformation_provenance[
                "matched_boundary_reference"
            ])

    def test_base_pool_and_global_rng_are_immutable(self):
        before_pool = tuple(base.to_record() for base in self.pool.bases)
        state = random.getstate()
        for base in self.pool.bases:
            for variant in SUPPORTED_IDENTITY_BINDING_VARIANTS:
                integrate_identity_binding_role(base, _request(base, variant))
        self.assertEqual(state, random.getstate())
        self.assertEqual(before_pool, tuple(base.to_record() for base in self.pool.bases))
        self.assertEqual(POOL_FINGERPRINT, self.pool.pool_fingerprint)

    def test_privileged_metadata_is_absent_from_detector_fields(self):
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        exported = {item["field_name"] for item in field_manifest()["fields"]}
        forbidden = {
            "selected_event_references", "matched_boundary_reference",
            "persistent_identity_relation", "token_binding_relation",
            "binding_applicability", "identity_binding_transformation",
            "behavioral_profile_id", "population_device_key",
        }
        self.assertTrue(forbidden.isdisjoint(exported))

    def test_unsupported_variant_is_explicit(self):
        base = self.pool.bases[0]
        result = integrate_identity_binding_role(
            base, _request(base, "not-a-4d3.6-variant"),
        )
        self.assertEqual(TransformationStatus.UNSUPPORTED, result.status)


if __name__ == "__main__":
    unittest.main()
