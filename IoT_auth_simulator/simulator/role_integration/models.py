"""Immutable privileged values describing validated ordinary base traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


VALID_BASE_ENVELOPE_VERSION = "4d3.1-valid-base-envelope-v1"
BASE_TRACE_REFERENCE_VERSION = "4d3.1-normalized-base-reference-sha256-v1"
TRACE_SECURITY_SNAPSHOT_VERSION = "4d3.1-trace-security-snapshot-v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_reference(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def ordinalize(values: Iterable[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if value is not None and str(value) not in result:
            result[str(value)] = f"ref-{len(result) + 1}"
    return result


def normalized_event_records(records: tuple[Mapping[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Normalize incidental UUID/entity material while retaining relationships."""
    reference_fields = (
        "auth_attempt_id", "protected_session_id", "access_request_id",
        "token_id", "current_token_id", "supersedes_token_id",
        "superseded_by_token_id", "nonce",
    )
    maps = {
        field: ordinalize(record.get(field) for record in records)
        for field in reference_fields
    }
    first_time = float(records[0]["timestamp"]) if records else 0.0
    normalized: list[dict[str, Any]] = []
    retained = (
        "event_type", "previous_state", "new_state", "result", "failure_reason",
        "retry_count", "token_generation", "token_lineage_status",
        "token_validation_result", "authorization_decision",
        "resource_operation_outcome", "authorization_evidence_complete",
        "authorization_authoritative", "authenticated_context_active",
        "token_context_active", "protected_session_active", "requested_action",
    )
    for ordinal, record in enumerate(records, 1):
        item: dict[str, Any] = {"event_ordinal": ordinal}
        item.update({name: record.get(name) for name in retained})
        item["semantic_offset_s"] = round(float(record["timestamp"]) - first_time, 6)
        item["observed_offset_s"] = round(
            float(record.get("observed_timestamp", record["timestamp"])) - first_time, 6,
        )
        item["delay_since_previous_event"] = record.get("delay_since_previous_event")
        for field in reference_fields:
            value = record.get(field)
            item[field] = maps[field].get(str(value)) if value is not None else None
        authenticated = record.get("authenticated_identity")
        claimed = record.get("identity_claim")
        device = record.get("device_id")
        item["authenticated_identity_matches_device"] = (
            authenticated == device if authenticated is not None else None
        )
        item["identity_claim_matches_device"] = claimed == device if claimed is not None else None
        bound = record.get("token_bound_identity")
        item["token_binding_matches_authenticated_identity"] = (
            bound == authenticated if bound is not None and authenticated is not None else None
        )
        bound_session = record.get("token_bound_protected_session_id")
        session = record.get("protected_session_id")
        item["token_binding_matches_protected_session"] = (
            bound_session == session if bound_session is not None and session is not None else None
        )
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class TokenLineageSnapshot:
    token_reference: str
    generation: int | None
    lineage_status: str | None
    supersedes_token_reference: str | None
    superseded_by_token_reference: str | None
    bound_identity_matches_authenticated_identity: bool | None
    bound_context_matches_protected_session: bool | None


@dataclass(frozen=True)
class TraceSecuritySnapshot:
    version: str
    authentication_attempt_references: tuple[str, ...]
    protected_session_references: tuple[str, ...]
    token_references: tuple[str, ...]
    nonce_references: tuple[str, ...]
    access_request_references: tuple[str, ...]
    token_lineage: tuple[TokenLineageSnapshot, ...]
    authentication_result: str
    authenticated_identity_matches_device: bool | None
    token_validation_result: str
    authorization_decisions: tuple[str, ...]
    resource_operation_outcomes: tuple[str, ...]
    authenticated_context_active: bool
    token_context_active: bool
    protected_session_active: bool


@dataclass(frozen=True)
class StructuralRealizationSummary:
    lifecycle: str
    recovery: str
    renewal: str
    activity_count: int
    timing_realization_present: bool
    population_device_key: str
    behavioral_profile_id: str


@dataclass(frozen=True)
class BaseValidationResult:
    validation_reference: str
    validator_version: str
    validation_schema_version: str
    validation_status: str
    build_gate_passed: bool
    validated_invariant_ids: tuple[str, ...]
    anomaly_ground_truth_present: bool
    hard_negative_role_present: bool
    transformation_provenance_present: bool

    @property
    def accepted(self) -> bool:
        return (
            self.validation_status == "validated_normal"
            and self.build_gate_passed
            and not self.validated_invariant_ids
            and not self.anomaly_ground_truth_present
            and not self.hard_negative_role_present
            and not self.transformation_provenance_present
        )


@dataclass(frozen=True)
class ValidBaseTrace:
    envelope_version: str
    base_trace_reference: str
    base_construction_key: str
    opportunity_randomness_reference: str
    population_reference: str
    device_id: str
    population_device_key: str
    behavioral_profile_id: str
    persistent_context_reference: str
    trace_plan_json: str
    planning_provenance_json: str
    event_records_json: tuple[str, ...]
    session_context_json: str
    security_snapshot: TraceSecuritySnapshot
    timing_provenance_json: str
    structural_summary: StructuralRealizationSummary
    validation: BaseValidationResult

    def __post_init__(self) -> None:
        if self.envelope_version != VALID_BASE_ENVELOPE_VERSION:
            raise ValueError("unsupported valid-base envelope version")
        if not self.validation.accepted:
            raise ValueError("ValidBaseTrace requires independently validated normal evidence")

    @property
    def event_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(item) for item in self.event_records_json)

    @property
    def planning_provenance(self) -> dict[str, Any]:
        return json.loads(self.planning_provenance_json)

    @property
    def timing_provenance(self) -> dict[str, Any]:
        return json.loads(self.timing_provenance_json)

    def normalized_identity_record(self) -> dict[str, Any]:
        return {
            "reference_version": BASE_TRACE_REFERENCE_VERSION,
            "envelope_version": self.envelope_version,
            "base_construction_key": self.base_construction_key,
            "opportunity_randomness_reference": self.opportunity_randomness_reference,
            "population_reference": self.population_reference,
            "population_device_key": self.population_device_key,
            "behavioral_profile_id": self.behavioral_profile_id,
            "trace_plan": json.loads(self.trace_plan_json),
            "planning_provenance": self.planning_provenance,
            "timing_provenance": self.timing_provenance,
            "structural_summary": asdict(self.structural_summary),
            "normalized_events": normalized_event_records(self.event_records),
            "validation": asdict(self.validation),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "event_records": [json.loads(item) for item in self.event_records_json],
            "normalized_event_evidence": list(normalized_event_records(self.event_records)),
            "session_context": json.loads(self.session_context_json),
            "trace_plan": json.loads(self.trace_plan_json),
            "planning_provenance": self.planning_provenance,
            "timing_provenance": self.timing_provenance,
        }
