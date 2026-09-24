"""Separated synchronized dataset views with C1.7 validation integration."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

import pandas as pd

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.settings import cfg
from simulator.data.dataset_contract import (
    FIELD_MANIFEST_VERSION,
    dataset_capability_manifest,
    field_manifest,
    field_names_for_view,
    historical_feature_contract,
)
from simulator.engines.event_engine import SessionContext
from simulator.event_model import AuthEvent, AuthState, EventResult, EventType
from simulator.provenance import GenerationProvenance, build_generation_provenance
from simulator.device_profiles.provenance import (
    to_device_profile_provenance,
    validate_device_profile_provenance,
)
from simulator.data.publication import (
    profile_artifact_bytes,
    profile_association_status,
    profile_companion_name,
    profile_publication_declaration,
    publish_staged_bundle,
)
from simulator.hard_negative import (
    HardNegativeQualificationRecord,
    validate_hard_negative_structure,
)
from simulator.validation import (
    EVENT_GROUND_TRUTH_SCHEMA_VERSION,
    TRACE_GROUND_TRUTH_SCHEMA_VERSION,
    ValidationReport,
    validate_synchronized_dataset,
)


SequencePairs = list[tuple[list[AuthEvent], SessionContext]]
CANONICAL_EVIDENCE_SERIALIZATION_VERSION = (
    "4d3.7-six-decimal-event-roundtrip-v1"
)

if TYPE_CHECKING:
    from simulator.runner import RunPopulation


def _event_from_canonical_record(record: Mapping[str, Any]) -> AuthEvent:
    allowed = {item.name for item in fields(AuthEvent)}
    values = {key: value for key, value in record.items() if key in allowed}
    values["event_type"] = EventType(values["event_type"])
    values["previous_state"] = AuthState(values["previous_state"])
    values["new_state"] = AuthState(values["new_state"])
    values["result"] = EventResult(values["result"])
    values["invariant_families"] = tuple(values.get("invariant_families", ()))
    return AuthEvent(**values)


def _context_from_canonical_record(record: Mapping[str, Any]) -> SessionContext:
    values = dict(record)
    for name in (
        "auth_attempt_ids", "renewal_auth_attempt_ids", "access_request_ids",
        "authorization_decisions", "resource_operation_outcomes",
        "injection_records",
    ):
        values[name] = tuple(values.get(name, ()))
    return SessionContext(**values)


def canonicalize_sequence_pairs(sequences: SequencePairs) -> SequencePairs:
    """Round-trip the exact event representation published to consumers.

    Validation and all synchronized views must operate on these canonical
    values.  This function does not repair precision collisions: a trace made
    invalid by serialization remains invalid and must be rejected by its
    admission boundary.
    """
    canonical: SequencePairs = []
    for events, context in sequences:
        event_records = json.loads(json.dumps(
            [event.to_dict() for event in events], allow_nan=False,
        ))
        context_record = json.loads(json.dumps(context.to_dict(), allow_nan=False))
        canonical.append((
            [_event_from_canonical_record(record) for record in event_records],
            _context_from_canonical_record(context_record),
        ))
    return canonical


def _require_synchronized_profiles(sequences: SequencePairs) -> None:
    incompatible = sorted({
        context.generation_profile for _, context in sequences
        if context.generation_profile != SYNCHRONIZED_GENERATION_PROFILE
    })
    if incompatible:
        raise ValueError(
            "synchronized exports reject historical generation profiles: "
            + ", ".join(incompatible)
        )


def _observed_time(event: AuthEvent) -> float:
    return float(event.observed_timestamp if event.observed_timestamp is not None else event.timestamp)


def to_detector_observation_df(sequences: SequencePairs) -> pd.DataFrame:
    """Return only legitimate observed evidence; no IDs, GT, debug, or provenance."""
    _require_synchronized_profiles(sequences)
    rows: list[dict[str, Any]] = []
    for events, _context in sequences:
        previous_observed: Optional[float] = None
        for event in events:
            observed = _observed_time(event)
            delay = 0.0 if previous_observed is None else observed - previous_observed
            rows.append({
                "event_type": event.event_type.value,
                "observed_timestamp": round(observed, 6),
                "observed_delay_since_previous_event": round(delay, 6),
                "result": event.result.value,
                "failure_reason": event.failure_reason,
                "retry_count": event.retry_count,
                "token_id": event.token_id,
                "nonce": event.nonce,
                "topic": event.topic,
                "resource_id": event.resource_id,
                "requested_action": event.requested_action,
            })
            previous_observed = observed
    return pd.DataFrame(rows, columns=field_names_for_view("detector_observations"))


def to_ground_truth_views(
    sequences: SequencePairs,
    validation: Optional[ValidationReport] = None,
) -> dict[str, list[dict[str, Any]]]:
    _require_synchronized_profiles(sequences)
    event_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    ordinal = 0
    for events, context in sequences:
        records = list(context.injection_records)
        record_by_id = {record.get("injection_id"): record for record in records}
        for event in events:
            record = record_by_id.get(event.injection_id)
            event_rows.append({
                "event_id": event.event_id,
                "event_ordinal": ordinal,
                "anomaly_label": event.anomaly_label,
                "anomaly_variant_name": event.anomaly_variant_name,
                "invariant_families": list(event.invariant_families),
                "validated_invariant_ids": list(
                    validation.event_validated_invariant_ids.get(event.event_id, ())
                ) if validation else [],
                "validated_invariant_evidence": list(
                    validation.event_validated_invariant_evidence.get(
                        event.event_id, ()
                    )
                ) if validation else [],
                "observable_violation_candidate": bool(
                    record and record.get("observable_violation_candidate", False)
                ),
                "validation_status": (
                    validation.event_ground_truth_statuses.get(event.event_id)
                    if validation else (
                        record.get("validation_status") if record
                        else "validation_not_run"
                    )
                ),
            })
            ordinal += 1
        trace_rows.append({
            "trace_id": context.trace_id,
            "observable_anomaly": bool(context.observable_anomaly),
            "observable_violation_candidate": bool(context.observable_violation_candidate),
            "anomaly_variants": sorted({
                record.get("anomaly_variant_name") for record in records
                if record.get("anomaly_variant_name")
            }),
            "invariant_families": sorted({
                family for record in records
                for family in record.get("invariant_families", ())
            }),
            "validated_invariant_ids": list(
                validation.trace_validated_invariant_ids.get(context.trace_id, ())
            ) if validation else [],
            "validated_invariant_evidence": [
                evidence
                for event in events
                for evidence in validation.event_validated_invariant_evidence.get(
                    event.event_id, ()
                )
            ] if validation else [],
            "validation_statuses": [
                validation.trace_ground_truth_statuses.get(
                    context.trace_id, "validation_failed"
                )
            ] if validation else ["validation_not_run"],
            "hard_negative_status": "not_implemented",
        })
    return {
        "event_schema_version": EVENT_GROUND_TRUTH_SCHEMA_VERSION,
        "trace_schema_version": TRACE_GROUND_TRUTH_SCHEMA_VERSION,
        "events": event_rows,
        "traces": trace_rows,
    }


def to_hard_negative_qualification_view(
    records: list[HardNegativeQualificationRecord],
    *,
    require_gate: bool = True,
) -> list[dict[str, Any]]:
    """Serialize the privileged HN surface; ordinary generation supplies none."""
    if require_gate:
        gate = validate_hard_negative_structure(records, field_manifest=field_manifest())
        if not gate.passed:
            raise ValueError("invalid hard-negative qualification records: " + "; ".join(gate.errors))
    return [record.to_dict() for record in records]


def to_provenance_view(
    sequences: SequencePairs,
    provenance: GenerationProvenance,
    trace_planning_records: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict[str, Any]:
    _require_synchronized_profiles(sequences)
    sequence_trace_ids = {context.trace_id for _, context in sequences}
    unknown_planning_trace_ids = set(trace_planning_records or {}) - sequence_trace_ids
    if unknown_planning_trace_ids:
        raise ValueError(
            "trace planning provenance references unknown traces: "
            + ", ".join(sorted(unknown_planning_trace_ids))
        )
    traces: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    observation_row = 0
    for events, context in sequences:
        traces.append({
            "run_id": provenance.run_id,
            "trace_id": context.trace_id,
            "scenario_id": context.scenario_id,
            "device_id": context.device_id,
            "legacy_session_id": context.legacy_session_id,
            "protected_session_id": context.protected_session_id,
            "auth_attempt_ids": list(context.auth_attempt_ids),
            "generation_profile": context.generation_profile,
            "scenario_family": context.attack_type,
            "scenario_intent": bool(context.attack_scenario_intent),
            "attack_phase": context.attack_phase,
            "generator_severity": context.severity,
            "injection_records": list(context.injection_records),
            "trace_planning": (
                dict(trace_planning_records[context.trace_id])
                if trace_planning_records is not None
                and context.trace_id in trace_planning_records
                else None
            ),
        })
        for event in events:
            events_out.append({
                "observation_row": observation_row,
                "event_id": event.event_id,
                "trace_id": event.trace_id,
                "scenario_id": event.scenario_id,
                "device_id": event.device_id,
                "gateway_id": event.gateway_id,
                "auth_server_id": event.auth_server_id,
                "broker_id": event.broker_id,
                "legacy_session_id": event.session_id,
                "protected_session_id": event.protected_session_id,
                "auth_attempt_id": event.auth_attempt_id,
                "access_request_id": event.access_request_id,
                "injection_id": event.injection_id,
            })
            observation_row += 1
    capabilities = dataset_capability_manifest(
        provenance,
        generation_profiles=(context.generation_profile for _, context in sequences),
    )
    simulation_config = provenance.configuration_snapshot.get("simulation", {})
    planning_records = list((trace_planning_records or {}).values())
    planning_stream_declared = bool(planning_records)
    randomness_metadata = {
        "master_seed": simulation_config.get("random_seed"),
        "stream_model": (
            "legacy_global_event_stream_plus_isolated_trace_planning_stream"
            if planning_stream_declared else "single_python_random_stream"
        ),
        "derived_stream_metadata_available": planning_stream_declared,
    }
    if planning_stream_declared:
        randomness_metadata.update({
            "planning_rng_derivation_versions": sorted({
                str(record["planning_rng_derivation_version"])
                for record in planning_records
            }),
            "planning_rng_namespaces": sorted({
                str(record["planning_rng_namespace"])
                for record in planning_records
            }),
            "planning_decision_slot_order_versions": sorted({
                str(record["planning_decision_slot_order_version"])
                for record in planning_records
            }),
        })
        timing_records = [
            record["timing_realization"] for record in planning_records
            if record.get("timing_realization") is not None
        ]
        if timing_records:
            randomness_metadata["stream_model"] = (
                "legacy_global_event_stream_plus_isolated_trace_planning_"
                "and_relation_timing_derivation"
            )
            randomness_metadata["timing_realization"] = {
                "derived_stream_metadata_available": True,
                "decision_slots": [5],
                "rng_derivation_versions": sorted({
                    str(record["timing_rng_derivation_version"])
                    for record in timing_records
                }),
                "realization_algorithm_versions": sorted({
                    str(record["timing_realization_algorithm_version"])
                    for record in timing_records
                }),
                "configuration_fingerprints": sorted({
                    str(record["timing_configuration_fingerprint"])
                    for record in timing_records
                }),
            }
    return {
        "dataset": {
            "generation": provenance.to_dict(),
            "capabilities": capabilities,
            "field_manifest_version": FIELD_MANIFEST_VERSION,
            "row_alignment": (
                "provenance.events.observation_row maps to zero-based row order in detector_observations"
            ),
            "randomness_metadata": randomness_metadata,
        },
        "traces": traces,
        "events": events_out,
    }


def to_debug_view(sequences: SequencePairs) -> dict[str, list[dict[str, Any]]]:
    _require_synchronized_profiles(sequences)
    event_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for events, context in sequences:
        for event in events:
            event_rows.append({
                "event_id": event.event_id,
                "semantic_timestamp": round(event.timestamp, 6),
                "observed_timestamp_source": event.observed_timestamp_source,
                "targeted_temporal_relationship": event.targeted_temporal_relationship,
                "previous_state": event.previous_state.value,
                "new_state": event.new_state.value,
                "authenticated_identity": event.authenticated_identity,
                "token_validation_result": event.token_validation_result,
                "authorization_decision": event.authorization_decision,
                "resource_operation_outcome": event.resource_operation_outcome,
                "authentication_required": event.authentication_required,
                "credential_required": event.credential_required,
                "active_session_required": event.active_session_required,
                "authorization_applicable": event.authorization_applicable,
                "authorization_evidence_complete": event.authorization_evidence_complete,
                "authorization_authoritative": event.authorization_authoritative,
                "authenticated_context_active": event.authenticated_context_active,
                "token_context_active": event.token_context_active,
                "protected_session_active": event.protected_session_active,
                "source_context": event.source_context,
                "token_expiry": event.token_expiry,
                "token_scope": event.token_scope,
                "current_token_id": event.current_token_id,
                "token_generation": event.token_generation,
                "token_lineage_status": event.token_lineage_status,
                "supersedes_token_id": event.supersedes_token_id,
                "superseded_by_token_id": event.superseded_by_token_id,
                "token_replacement_timestamp": event.token_replacement_timestamp,
                "token_bound_identity": event.token_bound_identity,
                "token_bound_protected_session_id": event.token_bound_protected_session_id,
            })
        trace_rows.append({
            "trace_id": context.trace_id,
            "semantic_time_context": {
                "semantic_time_domain": context.semantic_time_domain,
                "trace_started_at": context.trace_started_at,
                "trace_ended_at": context.trace_ended_at,
                "token_issued_at": context.token_issued_at,
                "token_presented_at": context.token_presented_at,
                "token_validation_at": context.token_validation_at,
                "token_validity_duration_s": context.token_validity_duration_s,
                "challenge_issued_at": context.challenge_issued_at,
                "challenge_response_at": context.challenge_response_at,
                "protected_session_started_at": context.protected_session_started_at,
                "protected_session_ended_at": context.protected_session_ended_at,
                "renewal_requested_at": context.renewal_requested_at,
                "renewed_token_issued_at": context.renewed_token_issued_at,
            },
            "security_decision_context": {
                "authenticated_identity": context.authenticated_identity,
                "authenticated_context_active": context.authenticated_context_active,
                "token_validation_result": context.token_validation_result,
                "token_context_active": context.token_context_active,
                "protected_session_active": context.protected_session_active,
                "authorization_decision": context.authorization_decision,
                "resource_operation_outcome": context.resource_operation_outcome,
            },
            "experimental_scores": {
                "trust_score": context.trust_score,
                "behavior_deviation_score": context.behavior_deviation_score,
                "gateway_decision": context.gateway_decision,
            },
            "privileged_generator_flags": {
                "replay_window_violation": context.replay_window_violation,
                "topic_scope_violation": context.topic_scope_violation,
                "source_ip_change": context.source_ip_change,
                "identity_claim_mismatch": context.identity_claim_mismatch,
                "token_device_mismatch": context.token_device_mismatch,
                "unauthorized_access_attempt": context.unauthorized_access_attempt,
                "token_age_at_replay": context.token_age_at_replay,
                "nonce_age_at_reuse": context.nonce_age_at_reuse,
                "timestamp_delta_s": context.timestamp_delta_s,
                "duplicate_session_count": context.duplicate_session_count,
            },
            "full_session_aggregates": {
                "event_count": len(events),
                "session_duration_s": round(max(event.timestamp for event in events) - events[0].timestamp, 6),
                "failure_count": sum(event.result.value == "failure" for event in events),
            },
        })
    return {"events": event_rows, "traces": trace_rows}


def build_synchronized_bundle(
    sequences: SequencePairs,
    *,
    provenance: Optional[GenerationProvenance] = None,
    hard_negative_qualification_records: Optional[list[HardNegativeQualificationRecord]] = None,
    run_population: Optional["RunPopulation"] = None,
) -> dict[str, Any]:
    sequences = canonicalize_sequence_pairs(sequences)
    provenance = provenance or build_generation_provenance(cfg)
    observations = to_detector_observation_df(sequences)
    manifest = field_manifest()
    manifest["legacy_representation"] = legacy_contract_for_sequences(sequences)
    provenance_view = to_provenance_view(
        sequences,
        provenance,
        trace_planning_records=(
            run_population.trace_planning_records
            if run_population is not None else None
        ),
    )
    profile_artifact = None
    if run_population is not None:
        profile_artifact = to_device_profile_provenance(
            run_population, run_id=provenance.run_id, sequences=sequences,
        )
        validate_device_profile_provenance(profile_artifact)
    provenance_view["dataset"]["capabilities"]["device_profile_provenance"] = (
        profile_association_status(profile_artifact)
    )
    validation = validate_synchronized_dataset(
        sequences,
        provenance=provenance,
        field_manifest=manifest,
        dataset_manifest=provenance_view["dataset"]["capabilities"],
        hard_negative_qualification_records=hard_negative_qualification_records,
    )
    bundle = {
        "detector_observations": observations.to_dict(orient="records"),
        "ground_truth": to_ground_truth_views(sequences, validation),
        "provenance": provenance_view,
        "debug": to_debug_view(sequences),
        "field_manifest": manifest,
        "validation": validation.to_dict(),
    }
    if hard_negative_qualification_records is not None:
        bundle["hard_negative_qualification"] = to_hard_negative_qualification_view(
            hard_negative_qualification_records, require_gate=False,
        )
    if profile_artifact is not None:
        bundle["device_profile_provenance"] = profile_artifact
    return bundle


class SynchronizedOutputViews:
    """Save separated synchronized views; never write the legacy feature table."""

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        sequences: SequencePairs,
        *,
        stem: str = "iot_auth",
        provenance: Optional[GenerationProvenance] = None,
        hard_negative_qualification_records: Optional[list[HardNegativeQualificationRecord]] = None,
        run_population: Optional["RunPopulation"] = None,
    ) -> dict[str, Path]:
        bundle = build_synchronized_bundle(
            sequences, provenance=provenance,
            hard_negative_qualification_records=hard_negative_qualification_records,
            run_population=run_population,
        )
        paths = {
            "detector_observations": self.output_dir / f"{stem}_observations.csv",
            "ground_truth_events": self.output_dir / f"{stem}_ground_truth_events.csv",
            "ground_truth_traces": self.output_dir / f"{stem}_ground_truth_traces.csv",
            "provenance": self.output_dir / f"{stem}_provenance.json",
            "debug": self.output_dir / f"{stem}_debug.json",
            "field_manifest": self.output_dir / f"{stem}_field_manifest.json",
            "dataset_manifest": self.output_dir / f"{stem}_dataset_manifest.json",
            "validation_report": self.output_dir / f"{stem}_validation_report.json",
        }
        profile_artifact = bundle.get("device_profile_provenance")
        profile_bytes = (
            profile_artifact_bytes(profile_artifact)
            if profile_artifact is not None else None
        )
        bundle["provenance"]["dataset"]["capabilities"]["device_profile_provenance"] = (
            profile_publication_declaration(
                stem=stem, artifact=profile_artifact, artifact_bytes=profile_bytes,
            )
        )
        outputs = {
            paths["detector_observations"].name: pd.DataFrame(
                bundle["detector_observations"]
            ).to_csv(index=False).encode("utf-8"),
            paths["ground_truth_events"].name: pd.DataFrame(
                bundle["ground_truth"]["events"]
            ).to_csv(index=False).encode("utf-8"),
            paths["ground_truth_traces"].name: pd.DataFrame(
                bundle["ground_truth"]["traces"]
            ).to_csv(index=False).encode("utf-8"),
            paths["provenance"].name: json.dumps(
                bundle["provenance"], indent=2, default=str,
            ).encode("utf-8"),
            paths["debug"].name: json.dumps(
                bundle["debug"], indent=2, default=str,
            ).encode("utf-8"),
            paths["field_manifest"].name: json.dumps(
                bundle["field_manifest"], indent=2,
            ).encode("utf-8"),
            paths["validation_report"].name: json.dumps(
                bundle["validation"], indent=2,
            ).encode("utf-8"),
            paths["dataset_manifest"].name: json.dumps(
                bundle["provenance"]["dataset"]["capabilities"], indent=2,
            ).encode("utf-8"),
        }
        if hard_negative_qualification_records is not None:
            paths["hard_negative_qualification"] = self.output_dir / f"{stem}_hard_negative_qualification.json"
            outputs[paths["hard_negative_qualification"].name] = json.dumps(
                bundle["hard_negative_qualification"], indent=2, default=str,
            ).encode("utf-8")
        if profile_bytes is not None:
            paths["device_profile_provenance"] = (
                self.output_dir / profile_companion_name(stem)
            )
            outputs[paths["device_profile_provenance"].name] = profile_bytes
        publish_staged_bundle(
            self.output_dir, stem=stem, outputs=outputs,
            profile_expected=profile_artifact is not None,
        )
        return paths


def legacy_contract_for_sequences(sequences: SequencePairs) -> dict[str, Any]:
    """Provide migration metadata without making legacy columns synchronized."""
    from simulator.data.output_views import to_feature_df

    return historical_feature_contract(to_feature_df(sequences).columns)
