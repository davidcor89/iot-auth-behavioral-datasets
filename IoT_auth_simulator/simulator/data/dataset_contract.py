"""Machine-readable C1.6 dataset information-boundary contract.

This module classifies implementation fields; it does not decide which fields
04 will use in an experiment.  The synchronized detector view is deliberately
small.  Fields whose observability depends on a yet-undefined monitoring point
are kept outside it and described conservatively as debug or provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from simulator.anomaly_contract import synchronized_capability_report
from simulator.provenance import GenerationProvenance
from simulator.validation import (
    EVENT_GROUND_TRUTH_SCHEMA_VERSION,
    GROUND_TRUTH_SCHEMA_VERSION,
    TRACE_GROUND_TRUTH_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    VALIDATOR_VERSION,
)
from simulator.invariant_registry import INVARIANT_REGISTRY_VERSION, invariant_registry_report
from simulator.hard_negative import HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION


FIELD_MANIFEST_VERSION = "sync-v1.1.0+4c4.activity-field-manifest-v5"
DATASET_MANIFEST_VERSION = "sync-v1.1.0-hn.4-dataset-manifest"
DATASET_INTERFACE_VERSION = "sync-v1.1.0-dataset-interface"
DATASET_VERSION = "sync-v1.1.0"
SYNCHRONIZED_OBSERVATION_VIEW = "detector_observations"
HISTORICAL_FEATURE_VIEW = "historical_109_column_feature_representation"


class FieldRole(str, Enum):
    OBSERVABLE = "observable"
    DERIVED_OBSERVABLE = "derived_observable"
    GROUND_TRUTH = "ground_truth"
    DEBUG = "debug"
    PROVENANCE = "provenance"
    IDENTIFIER = "identifier"


class Availability(str, Enum):
    EVENT_TIME = "event_time"
    SESSION_FINAL = "session_final"
    RETROSPECTIVE = "retrospective"
    FUTURE_INFORMATION_DEPENDENT = "future_information_dependent"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class FieldDefinition:
    field_name: str
    export_view: str
    data_type: str
    role: FieldRole
    availability: Availability
    source: str
    derivation_description: Optional[str] = None
    source_fields: tuple[str, ...] = ()
    temporal_scope: Optional[str] = None
    uses_future_information: bool = False
    requires_session_completion: bool = False
    grouping_identifier: bool = False
    privileged: bool = False
    detector_feature_default: bool = False
    semantic_event_relationship: Optional[str] = None
    anomaly_invariant_relationship: Optional[str] = None
    implementation_notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["role"] = self.role.value
        result["availability"] = self.availability.value
        result["type"] = result.pop("data_type")
        return result


def _field(
    name: str,
    view: str,
    data_type: str,
    role: FieldRole,
    availability: Availability,
    source: str,
    **kwargs: Any,
) -> FieldDefinition:
    return FieldDefinition(
        field_name=name,
        export_view=view,
        data_type=data_type,
        role=role,
        availability=availability,
        source=source,
        **kwargs,
    )


_OBS = SYNCHRONIZED_OBSERVATION_VIEW
_GT_EVENT = "ground_truth_events"
_GT_TRACE = "ground_truth_traces"
_PROV_DATASET = "provenance_dataset"
_PROV_TRACE = "provenance_traces"
_PROV_EVENT = "provenance_events"
_DEBUG_EVENT = "debug_events"
_DEBUG_TRACE = "debug_traces"
_HN_QUALIFICATION = "hard_negative_qualification"


# Inclusion here means role/horizon legitimacy only, never predictive utility.
SYNCHRONIZED_FIELD_DEFINITIONS: tuple[FieldDefinition, ...] = (
    _field("event_type", _OBS, "string", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.event_type", detector_feature_default=True,
           semantic_event_relationship="implemented event name; mapping remains noncanonical"),
    _field("observed_timestamp", _OBS, "number", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.observed_timestamp", detector_feature_default=True,
           implementation_notes="Detector receives observed time, never hidden semantic time."),
    _field("observed_delay_since_previous_event", _OBS, "number", FieldRole.DERIVED_OBSERVABLE,
           Availability.EVENT_TIME, "synchronized export transformation",
           derivation_description="Current observed timestamp minus the preceding observed timestamp in the trace.",
           source_fields=("observed_timestamp",), temporal_scope="trace_prefix",
           detector_feature_default=True),
    _field("result", _OBS, "string", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.result", detector_feature_default=True),
    _field("failure_reason", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.failure_reason", detector_feature_default=True),
    _field("retry_count", _OBS, "integer", FieldRole.DERIVED_OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.retry_count", derivation_description="Count available through the current trace prefix.",
           temporal_scope="trace_prefix", detector_feature_default=True),
    _field("token_id", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.token_id", detector_feature_default=True,
           anomaly_invariant_relationship="raw material only; no replay convenience flag"),
    _field("nonce", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.nonce", detector_feature_default=True,
           anomaly_invariant_relationship="raw challenge evidence only; no nonce_reused flag"),
    _field("topic", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.topic", detector_feature_default=True),
    _field("resource_id", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.resource_id", detector_feature_default=True),
    _field("requested_action", _OBS, "string|null", FieldRole.OBSERVABLE, Availability.EVENT_TIME,
           "AuthEvent.requested_action", detector_feature_default=True),

    _field("event_id", _GT_EVENT, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "AuthEvent.event_id", privileged=True),
    _field("event_ordinal", _GT_EVENT, "integer", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "synchronized export ordering", privileged=True),
    _field("anomaly_label", _GT_EVENT, "string|null", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "AuthEvent.anomaly_label", privileged=True,
           implementation_notes="Candidate label reconciled with independent C1.7 validation status."),
    _field("anomaly_variant_name", _GT_EVENT, "string|null", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "AuthEvent.anomaly_variant_name", privileged=True),
    _field("invariant_families", _GT_EVENT, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "AuthEvent.invariant_families", privileged=True),
    _field("validated_invariant_ids", _GT_EVENT, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "C2.5 confirmed atomic validation evidence", privileged=True,
           anomaly_invariant_relationship="stable IDs derived from confirmed validator records only"),
    _field("validated_invariant_evidence", _GT_EVENT, "array[object]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "C2.5 event/operation validation association", privileged=True,
           implementation_notes="Retains target and supporting event references; never detector-visible."),
    _field("observable_violation_candidate", _GT_EVENT, "boolean", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "presence of C1.5 event candidate evidence", privileged=True),
    _field("validation_status", _GT_EVENT, "string", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "C1.7 independent validator", privileged=True),

    _field("trace_id", _GT_TRACE, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "SessionContext.trace_id", grouping_identifier=True, privileged=True),
    _field("observable_anomaly", _GT_TRACE, "boolean", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "SessionContext.observable_anomaly", privileged=True,
           implementation_notes="C1.5 evidence outcome accompanied by independent C1.7 status."),
    _field("observable_violation_candidate", _GT_TRACE, "boolean", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "SessionContext.observable_violation_candidate", privileged=True),
    _field("anomaly_variants", _GT_TRACE, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "SessionContext.injection_records", privileged=True),
    _field("invariant_families", _GT_TRACE, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "SessionContext.injection_records", privileged=True),
    _field("validated_invariant_ids", _GT_TRACE, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "C2.5 aggregate of confirmed event/operation evidence", privileged=True),
    _field("validated_invariant_evidence", _GT_TRACE, "array[object]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "C2.5 aggregate event/operation associations", privileged=True),
    _field("validation_statuses", _GT_TRACE, "array[string]", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "SessionContext.injection_records", privileged=True),
    _field("hard_negative_status", _GT_TRACE, "string", FieldRole.GROUND_TRUTH,
           Availability.NOT_APPLICABLE, "dataset capability contract", privileged=True,
           implementation_notes="Not implemented; reserved without assigning a hard-negative label."),

    # Dataset-level provenance is serialized as one declared object.
    _field("generation", _PROV_DATASET, "object", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "GenerationProvenance", privileged=True),
    _field("capabilities", _PROV_DATASET, "object", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "dataset_capability_manifest", privileged=True),
    _field("field_manifest_version", _PROV_DATASET, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "C1.7 integrated dataset contract", privileged=True),
    _field("row_alignment", _PROV_DATASET, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "C1.6 information-boundary contract", privileged=True),
    _field("randomness_metadata", _PROV_DATASET, "object", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "generation configuration", privileged=True),

    _field("run_id", _PROV_TRACE, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "GenerationProvenance.run_id", grouping_identifier=True, privileged=True),
    _field("trace_id", _PROV_TRACE, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "SessionContext.trace_id", grouping_identifier=True, privileged=True),
    _field("scenario_id", _PROV_TRACE, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "SessionContext.scenario_id", grouping_identifier=True, privileged=True),
    _field("device_id", _PROV_TRACE, "string", FieldRole.IDENTIFIER, Availability.NOT_APPLICABLE,
           "SessionContext.device_id", grouping_identifier=True, privileged=True),
    _field("legacy_session_id", _PROV_TRACE, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "SessionContext.legacy_session_id",
           grouping_identifier=True, privileged=True),
    _field("protected_session_id", _PROV_TRACE, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "SessionContext.protected_session_id",
           grouping_identifier=True, privileged=True),
    _field("auth_attempt_ids", _PROV_TRACE, "array[string]", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "SessionContext.auth_attempt_ids",
           grouping_identifier=True, privileged=True),
    _field("generation_profile", _PROV_TRACE, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.generation_profile", privileged=True),
    _field("scenario_family", _PROV_TRACE, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.attack_type", privileged=True),
    _field("scenario_intent", _PROV_TRACE, "boolean", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.attack_scenario_intent", privileged=True),
    _field("attack_phase", _PROV_TRACE, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.attack_phase", privileged=True),
    _field("generator_severity", _PROV_TRACE, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.severity", privileged=True),
    _field("injection_records", _PROV_TRACE, "array[object]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "SessionContext.injection_records", privileged=True,
           anomaly_invariant_relationship="complete C1.5 transformation and source-material lineage"),
    _field("trace_planning", _PROV_TRACE, "object|null", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "4C TracePlan provenance record", privileged=True,
           implementation_notes=(
               "Present only for opt-in ordinary planning; 4C.5 records "
               "applicability and causal L/R/N/activity choices plus separated "
               "privileged timing-realization provenance, without profile IDs, "
               "latent values, or raw planning variates."
           )),

    _field("observation_row", _PROV_EVENT, "integer", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "synchronized export ordering", privileged=True),
    _field("event_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.event_id", privileged=True),
    _field("trace_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.trace_id", grouping_identifier=True, privileged=True),
    _field("scenario_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.scenario_id", grouping_identifier=True, privileged=True),
    _field("device_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.device_id", grouping_identifier=True, privileged=True),
    _field("gateway_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.gateway_id", grouping_identifier=True, privileged=True),
    _field("auth_server_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.auth_server_id", grouping_identifier=True, privileged=True),
    _field("broker_id", _PROV_EVENT, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.broker_id", grouping_identifier=True, privileged=True),
    _field("legacy_session_id", _PROV_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.session_id", grouping_identifier=True, privileged=True),
    _field("protected_session_id", _PROV_EVENT, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.protected_session_id", grouping_identifier=True, privileged=True),
    _field("auth_attempt_id", _PROV_EVENT, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.auth_attempt_id", grouping_identifier=True, privileged=True),
    _field("access_request_id", _PROV_EVENT, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.access_request_id", grouping_identifier=True, privileged=True),
    _field("injection_id", _PROV_EVENT, "string|null", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "AuthEvent.injection_id", privileged=True),

    _field("event_id", _DEBUG_EVENT, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "AuthEvent.event_id", privileged=True),
    _field("semantic_timestamp", _DEBUG_EVENT, "number", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.timestamp", privileged=True,
           implementation_notes="Authoritative generator/validation time; hidden from detector."),
    _field("observed_timestamp_source", _DEBUG_EVENT, "string", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.observed_timestamp_source", privileged=True),
    _field("targeted_temporal_relationship", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.targeted_temporal_relationship", privileged=True),
    _field("previous_state", _DEBUG_EVENT, "string", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.previous_state", privileged=True),
    _field("new_state", _DEBUG_EVENT, "string", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.new_state", privileged=True),
    _field("authenticated_identity", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authenticated_identity", privileged=True,
           implementation_notes="Conservative: monitoring point unresolved."),
    _field("token_validation_result", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_validation_result", privileged=True,
           implementation_notes="Conservative: internal semantic decision evidence."),
    _field("authorization_decision", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authorization_decision", privileged=True,
           implementation_notes="Conservative: monitoring point unresolved."),
    _field("resource_operation_outcome", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.resource_operation_outcome", privileged=True),
    _field("authentication_required", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authentication_required", privileged=True),
    _field("credential_required", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.credential_required", privileged=True),
    _field("active_session_required", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.active_session_required", privileged=True,
           implementation_notes="Reference-profile applicability only."),
    _field("authorization_applicable", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authorization_applicable", privileged=True),
    _field("authorization_evidence_complete", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authorization_evidence_complete", privileged=True),
    _field("authorization_authoritative", _DEBUG_EVENT, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authorization_authoritative", privileged=True),
    _field("authenticated_context_active", _DEBUG_EVENT, "boolean", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.authenticated_context_active", privileged=True),
    _field("token_context_active", _DEBUG_EVENT, "boolean", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_context_active", privileged=True),
    _field("protected_session_active", _DEBUG_EVENT, "boolean", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.protected_session_active", privileged=True),
    _field("source_context", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.source_context", privileged=True),
    _field("token_expiry", _DEBUG_EVENT, "number|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_expiry", privileged=True,
           implementation_notes="Conservative until credential visibility is defined."),
    _field("token_scope", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_scope", privileged=True),
    _field("current_token_id", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.current_token_id", privileged=True,
           implementation_notes="C2.1 validator evidence; detector eligibility remains unresolved."),
    _field("token_generation", _DEBUG_EVENT, "integer|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_generation", privileged=True),
    _field("token_lineage_status", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_lineage_status", privileged=True),
    _field("supersedes_token_id", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.supersedes_token_id", privileged=True),
    _field("superseded_by_token_id", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.superseded_by_token_id", privileged=True),
    _field("token_replacement_timestamp", _DEBUG_EVENT, "number|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_replacement_timestamp", privileged=True),
    _field("token_bound_identity", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_bound_identity", privileged=True,
           implementation_notes="C2.4 identity-binding validation evidence; detector eligibility remains unresolved."),
    _field("token_bound_protected_session_id", _DEBUG_EVENT, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "AuthEvent.token_bound_protected_session_id", privileged=True,
           implementation_notes="Existing reference-profile token-lineage context association; not a universal session-binding claim."),

    _field("trace_id", _DEBUG_TRACE, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "SessionContext.trace_id", privileged=True),
    _field("semantic_time_context", _DEBUG_TRACE, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "SessionContext C1.4 fields", privileged=True),
    _field("security_decision_context", _DEBUG_TRACE, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "SessionContext C1.3 fields", privileged=True),
    _field("experimental_scores", _DEBUG_TRACE, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "historical SessionContext score helpers", privileged=True,
           implementation_notes="trust/deviation/gateway decisions are noncanonical."),
    _field("privileged_generator_flags", _DEBUG_TRACE, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "historical SessionContext anomaly helpers", privileged=True),
    _field("full_session_aggregates", _DEBUG_TRACE, "object", FieldRole.DEBUG,
           Availability.RETROSPECTIVE, "legacy feature builder", privileged=True,
           derivation_description="Aggregates require the complete event sequence.",
           source_fields=("events",), temporal_scope="complete_trace",
           uses_future_information=True, requires_session_completion=True),

    # HN.1 declares a future privileged methodological surface. Ordinary
    # generation emits no records and anomaly GT remains unchanged.
    _field("hard_negative_candidate_id", _HN_QUALIFICATION, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "HardNegativeQualificationRecord", privileged=True),
    _field("trace_id", _HN_QUALIFICATION, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "HardNegativeQualificationRecord", grouping_identifier=True, privileged=True),
    _field("scenario_id", _HN_QUALIFICATION, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "HardNegativeQualificationRecord", grouping_identifier=True, privileged=True),
    _field("device_id", _HN_QUALIFICATION, "string", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "HardNegativeQualificationRecord", grouping_identifier=True, privileged=True),
    _field("candidate_status", _HN_QUALIFICATION, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN construction intent", privileged=True),
    _field("primary_intended_family_id", _HN_QUALIFICATION, "string|null", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN family registry", privileged=True),
    _field("intended_variant_id", _HN_QUALIFICATION, "string|null", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN variant registry", privileged=True),
    _field("intended_boundary_invariant_ids", _HN_QUALIFICATION, "array[string]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN variant mapping", privileged=True),
    _field("construction_status", _HN_QUALIFICATION, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN construction record", privileged=True),
    _field("construction_parameters", _HN_QUALIFICATION, "object", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future HN constructor", privileged=True),
    _field("boundary_parameters", _HN_QUALIFICATION, "object", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future HN boundary constructor", privileged=True),
    _field("qualification_status", _HN_QUALIFICATION, "string", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future independent HN qualification", privileged=True),
    _field("qualified_family_ids", _HN_QUALIFICATION, "array[string]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future independent HN qualification", privileged=True),
    _field("secondary_satisfied_family_ids", _HN_QUALIFICATION, "array[string]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future incidental-family discovery", privileged=True),
    _field("qualified_boundary_invariant_ids", _HN_QUALIFICATION, "array[string]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "future independent HN qualification", privileged=True),
    _field("qualification_check_ids", _HN_QUALIFICATION, "array[string]", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN qualification-check registry", privileged=True),
    _field("target_event_ids", _HN_QUALIFICATION, "array[string]", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "future independent HN qualification", privileged=True),
    _field("evidence_event_ids", _HN_QUALIFICATION, "array[string]", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "future independent HN qualification", privileged=True),
    _field("normal_side_predicates", _HN_QUALIFICATION, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("boundary_proximity_evidence", _HN_QUALIFICATION, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("relevant_validation_check_ids", _HN_QUALIFICATION, "array[string]", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("relevant_validation_outcomes", _HN_QUALIFICATION, "object", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("disqualifying_invariant_ids", _HN_QUALIFICATION, "array[string]", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("qualification_reason", _HN_QUALIFICATION, "string", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "future HN.2 evidence", privileged=True),
    _field("counterfactual_pair_id", _HN_QUALIFICATION, "string|null", FieldRole.IDENTIFIER,
           Availability.NOT_APPLICABLE, "reserved future pairing provenance", privileged=True),
    _field("qualifier_implementation_version", _HN_QUALIFICATION, "string|null", FieldRole.PROVENANCE,
           Availability.NOT_APPLICABLE, "HN qualifier implementation", privileged=True),
    _field("trace_ground_truth_status", _HN_QUALIFICATION, "string|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "independent validator trace result", privileged=True),
    _field("trace_observable_anomaly", _HN_QUALIFICATION, "boolean|null", FieldRole.DEBUG,
           Availability.NOT_APPLICABLE, "independent semantic anomaly GT safeguard", privileged=True),
)


def field_manifest() -> dict[str, Any]:
    return {
        "manifest_version": FIELD_MANIFEST_VERSION,
        "authority_boundary": {
            "classification_owner": "03",
            "experiment_selection_owner": "04",
            "canonical_behavior_owner": "02",
        },
        "role_vocabulary": [role.value for role in FieldRole],
        "availability_vocabulary": [item.value for item in Availability],
        "default_detector_view": _OBS,
        "default_detector_view_policy": (
            "Only observable and derived_observable fields marked "
            "detector_feature_default; no identifiers, GT, debug, or provenance."
        ),
        "fields": [item.to_dict() for item in SYNCHRONIZED_FIELD_DEFINITIONS],
        "legacy_representation": historical_feature_contract(),
    }


def fields_for_view(view: str) -> tuple[FieldDefinition, ...]:
    return tuple(item for item in SYNCHRONIZED_FIELD_DEFINITIONS if item.export_view == view)


def field_names_for_view(view: str) -> tuple[str, ...]:
    return tuple(item.field_name for item in fields_for_view(view))


def historical_feature_contract(columns: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Describe—not legitimize—the mixed historical session-feature table."""
    privileged_names = {
        "is_anomaly", "attack_type", "attack_phase", "severity",
        "attack_scenario_intent", "observable_anomaly",
        "observable_violation_candidate", "generation_profile", "injection_count",
        "token_age_at_replay", "nonce_age_at_reuse", "timestamp_delta_s",
        "duplicate_session_count", "identity_claim_mismatch",
        "token_device_mismatch", "unauthorized_access_attempt",
        "identity_mismatch", "n_token_reuses", "n_nonce_reuses", "n_state_jumps",
        "trust_score", "behavior_deviation_score", "gateway_decision",
        "replay_window_violation", "topic_scope_violation", "source_ip_change",
    }
    identifier_names = {
        "scenario_id", "trace_id", "session_id", "protected_session_id",
        "device_id", "gateway_id",
    }
    retrospective_prefixes = ("n_", "reached_", "visited_")
    migration = []
    for name in (() if columns is None else columns):
        if name in identifier_names:
            classification = FieldRole.IDENTIFIER.value
        elif name in privileged_names:
            classification = "legacy_privileged_or_shortcut"
        elif name.startswith(retrospective_prefixes) or name in {
            "session_duration_s", "mean_delay_s", "max_delay_s", "min_delay_s",
            "failure_rate", "auth_attempt_count", "access_request_count",
        }:
            classification = "legacy_retrospective"
        else:
            classification = "ambiguous_requires_03_observability_review"
        migration.append({"field_name": name, "provisional_disposition": classification})
    return {
        "view_name": HISTORICAL_FEATURE_VIEW,
        "status": "legacy_mixed_role_not_detector_ready",
        "canonical_schema": False,
        "detector_ready": False,
        "historical_column_count_is_not_a_requirement": True,
        "migration_metadata": migration,
    }


def dataset_capability_manifest(
    provenance: GenerationProvenance,
    *,
    generation_profiles: Iterable[str],
) -> dict[str, Any]:
    anomaly = synchronized_capability_report()
    profiles = sorted(set(generation_profiles))
    supported = sorted(anomaly["supported_variants"])
    all_non_supported = sorted(
        name for name, item in anomaly["variants"].items()
        if name not in supported
    )
    quarantined = sorted(
        name for name, item in anomaly["variants"].items()
        if item["status"] == "quarantined_historical_variant"
    )
    composite = sorted(
        name for name, item in anomaly["variants"].items()
        if item["status"] == "composite_scenario_provenance_only"
    )
    history_dependent = sorted(
        name for name, item in anomaly["variants"].items()
        if item["status"] == "unsupported_history_dependent_behavior"
    )
    registry = invariant_registry_report()
    exact_invariants = sorted(
        item["invariant_id"] for item in registry["invariants"]
        if item["implementation_status"] == "exact"
    )
    partial_invariants = {
        item["invariant_id"]: {
            "implemented_checks": item["current_checks"],
            "relation_capabilities": item["relation_capabilities"],
            "remaining_delta": item["implementation_delta"],
        }
        for item in registry["invariants"]
        if item["implementation_status"] == "partial"
    }
    explicitly_unsupported = [
        "independent_request_level_acting_identity",
        "token_issuer_applicability",
        "universal_resource_action_token_binding",
        "cross_session_token_replay",
        "nonce_uniqueness_beyond_declared_within_trace_attempt_domain",
        "sequence_replay_or_duplication",
        "statistical_failure_rate_anomalies",
        "statistical_renewal_anomalies",
        "global_discrete_event_scheduling_or_concurrency",
        "compromised_gateway_or_device_semantics",
        "operational_monitoring_point_mapping",
        "final_detector_feature_or_prediction_horizon_decisions",
    ]
    return {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "field_manifest_version": FIELD_MANIFEST_VERSION,
        "dataset_interface_version": DATASET_INTERFACE_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_version_compatibility": {
            "release_family": "sync-v1.1.0",
            "feature_schema": (
                "retained unchanged from C1.6 because detector observations "
                "and the legacy derived-feature structure did not change"
            ),
        },
        "invariant_registry_version": INVARIANT_REGISTRY_VERSION,
        "invariant_registry": registry,
        "behavior_model_version": provenance.behavior_model_version,
        "event_schema_version": provenance.event_schema_version,
        "feature_schema_version": provenance.feature_schema_version,
        "generator_software_version": provenance.generator_software_version,
        "generator_git_commit": provenance.generator_git_commit,
        "generator_git_dirty": provenance.generator_git_dirty,
        "generator_git_state_scope": provenance.generator_git_state_scope,
        "run_id": provenance.run_id,
        "configuration_version": provenance.configuration_version,
        "configuration_snapshot_reference": provenance.configuration_snapshot_reference,
        "randomness_contract": provenance.randomness_contract,
        "generation_profile": profiles[0] if len(profiles) == 1 else "mixed",
        "generation_profiles": profiles,
        "supported_synchronized_anomaly_variants": supported,
        "quarantined_historical_variants": quarantined,
        "composite_scenario_provenance_only": composite,
        "unsupported_history_dependent_variants": history_dependent,
        "all_non_supported_historical_variants": all_non_supported,
        "exact_supported_invariant_ids": exact_invariants,
        "partial_invariant_capabilities": partial_invariants,
        "explicitly_unsupported_or_out_of_scope_capabilities": explicitly_unsupported,
        "persistent_device_context_support": "implemented",
        "cross_session_history_support": "device_timeline_only_no_replay_material_history",
        "hard_negative_capability": "opt_in_privileged_qualification_not_anomaly_truth",
        "global_discrete_event_simulation": False,
        "concurrency_support": "none_sequential_generation_only",
        "operational_trace_support": False,
        "calibration_status": "not_calibrated_against_operational_or_testbed_data",
        "validator_version": VALIDATOR_VERSION,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "ground_truth_schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "event_ground_truth_schema_version": EVENT_GROUND_TRUTH_SCHEMA_VERSION,
        "trace_ground_truth_schema_version": TRACE_GROUND_TRUTH_SCHEMA_VERSION,
        "full_executable_invariant_validation": "implemented_supported_v1_subset",
        "default_detector_view": _OBS,
        "release_claim": (
            "Behavioral-model v1.1-aligned scenario-based synthetic generator "
            "for the explicitly declared exact and partial capability subset; "
            "not empirical or operational validation."
        ),
        "hard_negative_capability_contract": provenance.hard_negative_capability_contract,
        "hard_negative_qualification_surface": {
            "view_name": _HN_QUALIFICATION,
            "schema_version": HARD_NEGATIVE_QUALIFICATION_SCHEMA_VERSION,
            "role": "privileged_methodological_provenance_and_debug",
            "ordinary_generation_records": 0,
            "separate_from_anomaly_ground_truth": True,
        },
        "known_limitations": [
            "Only the deterministic subset, three scoped replay/temporal checks, five reference-profile access invariants, the persistent-to-authenticated identity relation, and represented token identity/protected-context binding relations are validated.",
            "Token issuer applicability is unsupported; scope applicability remains separate/partial and no universal token/session binding is claimed.",
            "Authenticated-to-request-actor identity comparison is unsupported because no independent request acting identity is represented.",
            "Validation may reject traces produced by historical transient-progression behavior; failures are retained rather than relabeled.",
            "Monitoring/observation point is unresolved; internal security decisions are excluded.",
            "No cross-session token or nonce material history is claimed.",
            "No final ML prediction horizon or feature selection is encoded.",
            "Historical session feature representation remains mixed-role legacy evidence.",
            "HN.3R rejected protected-request attempts are opt-in and do not advance the ordinary success-path FSM.",
            "HN-v1.0.0 freezes explicit single-family construction and qualification only; sampling, prevalence, empirical difficulty, device-profile variability, and operational calibration remain unfrozen.",
        ],
    }
