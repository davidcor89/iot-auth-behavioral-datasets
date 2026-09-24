"""Deterministic, engineering-only mixed HN stress audit; not dataset composition.

Run with ``PYTHONPATH=IoT_auth_simulator .venv/bin/python tests/hn4_stress_audit.py``.
No dataset files are written. The emitted JSON is an audit summary, not an ML result.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE, supported_synchronized_variants
from simulator.config.settings import cfg
from simulator.data.dataset_contract import dataset_capability_manifest, field_manifest, field_names_for_view
from simulator.data.synchronized_views import to_detector_observation_df
from simulator.engines.event_engine import EventEngine, STEALTH_FRACTION
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.event_model import EventType
from simulator.hard_negative import (
    CandidateStatus, ConstructionStatus, HN_FAMILIES, HN_VARIANTS,
    HardNegativeQualificationRecord, HardNegativeRunSummary, QualificationStatus,
)
from simulator.hard_negative_generation import HardNegativeRequest, generate_hard_negative
from simulator.hard_negative_qualification import qualify_hard_negative
from simulator.provenance import build_generation_provenance
from simulator.security_context import PersistentDeviceContext
from simulator.validation import validate_synchronized_dataset


def _structure(events: list, context: Any) -> dict[str, Any]:
    """Normalize unstable UUIDs while retaining semantic relationships."""
    tokens: dict[str, int] = {}
    nonces: dict[str, int] = {}
    attempts: dict[str, int] = {}
    sessions: dict[str, int] = {}

    def ordinal(value: Any, mapping: dict[str, int]) -> int | None:
        if value is None:
            return None
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        return mapping[value]

    start = events[0].timestamp
    return {
        "events": [
            (
                event.event_type.value, event.previous_state.value,
                event.new_state.value, event.result.value,
                round(event.timestamp - start, 6),
                round((event.observed_timestamp or event.timestamp) - event.timestamp, 6),
                ordinal(event.token_id, tokens), ordinal(event.nonce, nonces),
                ordinal(event.auth_attempt_id, attempts),
                ordinal(event.protected_session_id, sessions),
                event.token_validation_result, event.authorization_decision,
                event.resource_operation_outcome,
            ) for event in events
        ],
        "observable_anomaly": context.observable_anomaly,
        "persistent_enrolled": context.persistent_enrolled,
    }


def _statistics(items: list[tuple[list, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    def summary(values: list[float]) -> dict[str, float]:
        return {
            "min": round(min(values), 6),
            "median": round(statistics.median(values), 6),
            "max": round(max(values), 6),
        }
    return {
        "trace_length": summary([len(events) for events, _ in items]),
        "trace_span_s": summary([events[-1].timestamp - events[0].timestamp for events, _ in items]),
        "retry_events": summary([sum(event.event_type is EventType.RETRY for event in events) for events, _ in items]),
        "auth_attempts": summary([len(context.auth_attempt_ids) for _, context in items]),
        "token_presentations": summary([sum(event.event_type is EventType.TOKEN_PRESENTED for event in events) for events, _ in items]),
        "requests": summary([sum(event.event_type is EventType.ACCESS_REQUEST for event in events) for events, _ in items]),
        "denials": summary([sum(event.event_type is EventType.ACCESS_DENIED for event in events) for events, _ in items]),
        "protected_sessions": summary([int(context.protected_session_id is not None) for _, context in items]),
        "event_type_counts": dict(Counter(event.event_type.value for events, _ in items for event in events)),
    }


def run_campaign(
    *, seed: int = 7000, per_variant: int = 150, ordinary_count: int = 500,
    per_anomaly: int = 30, intent_only_count: int = 70,
    malformed_count: int = 50, device_count: int = 16,
) -> dict[str, Any]:
    """Cycle explicit requests over reused devices; fail on semantic corruption."""
    persistent = [PersistentDeviceContext(device_id=f"d{index:03d}-audit") for index in range(device_count)]
    provenance = build_generation_provenance(cfg, run_id=f"hn4-stress-{seed}")
    manifest = field_manifest()
    dataset_manifest = dataset_capability_manifest(
        provenance, generation_profiles=(SYNCHRONIZED_GENERATION_PROFILE,),
    )
    sequences: list[tuple[list, Any]] = []
    records: list[HardNegativeQualificationRecord] = []
    by_kind: dict[str, list[tuple[list, Any]]] = defaultdict(list)
    by_family: dict[str, list[tuple[list, Any]]] = defaultdict(list)
    family_devices: dict[str, Counter[str]] = defaultdict(Counter)
    anomaly_statuses: Counter[str] = Counter()
    malformed_statuses: Counter[str] = Counter()
    malformed_gate_failures = 0
    candidate_ids: set[str] = set()
    semantic_rows: list[dict[str, Any]] = []
    qualified_positive_gt = 0
    qualified_invariant_violations = 0
    generator_defects = 0
    time_margins: list[float] = []
    trace_index = 0

    def next_device() -> PersistentDeviceContext:
        nonlocal trace_index
        device = persistent[trace_index % device_count]
        trace_index += 1
        return device

    def next_start_time() -> float:
        # A reused device's later trace must follow its earlier trace on the
        # same synthetic timeline. The 16,000 s per-device gap exceeds the
        # bounded trace spans in this engineering campaign.
        return 10_000.0 + trace_index * 1_000.0

    def accept(events: list, context: Any, kind: str, *, family: str | None = None) -> None:
        sequences.append((events, context))
        by_kind[kind].append((events, context))
        if family:
            by_family[family].append((events, context))
            family_devices[family][context.device_id] += 1
        semantic_rows.append({"kind": kind, "family": family, **_structure(events, context)})

    for repetition in range(per_variant):
        for variant_index, variant in enumerate(HN_VARIANTS):
            device = next_device()
            candidate_id = f"audit-{seed}-{repetition}-{variant_index}"
            if candidate_id in candidate_ids:
                raise AssertionError("duplicate HN candidate ID")
            candidate_ids.add(candidate_id)
            boundary = {"epsilon_hn": 0.01} if variant.variant_id == "small_positive_causal_margin" else {}
            result = generate_hard_negative(
                HardNegativeRequest(
                    variant.family_id, variant.variant_id, candidate_id,
                    {"engineering_repetition": repetition}, boundary,
                ),
                device_id=device.device_id, seed=seed + repetition * 101 + variant_index,
                start_time=next_start_time(), persistent_context=device, provenance=provenance,
            )
            record = result.candidate
            records.append(record)
            accept(list(result.events), result.context, "qualified_hn", family=variant.family_id)
            generator_defects += result.generator_defect is not None
            if variant.variant_id == "small_positive_causal_margin":
                time_margins.append(float(record.boundary_parameters["sampled_margin_s"]))
            qualified_positive_gt += record.qualification_status is QualificationStatus.QUALIFIED and result.context.observable_anomaly
            qualified_invariant_violations += (
                record.qualification_status is QualificationStatus.QUALIFIED
                and bool(result.validation.trace_validated_invariant_ids[result.context.trace_id])
            )
            if (record.construction_status is not ConstructionStatus.APPLIED
                    or record.qualification_status is not QualificationStatus.QUALIFIED
                    or not result.gated_validation.build_gate_passed):
                raise AssertionError(f"HN construction/qualification failed: {variant.variant_id}: {record.qualification_status}")

    for index in range(ordinary_count):
        device = next_device()
        local_seed = seed + 100_000 + index
        random.seed(local_seed)
        scenario_engine = ScenarioEngine(local_seed)
        mode = index % 5
        if mode == 0:
            spec = scenario_engine.normal_with_retry()
        elif mode == 1:
            spec = scenario_engine.normal_with_renewal()
        elif mode == 2:
            spec = scenario_engine.normal_partial()
        else:
            spec = scenario_engine.normal()
        spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
        events, context = EventEngine(
            device_id=device.device_id, gateway_id="gateway", auth_server_id="auth_server",
            broker_id="broker", start_time=next_start_time(), persistent_context=device,
        ).execute(spec)
        accept(events, context, "ordinary_normal")

    anomalies = supported_synchronized_variants()
    with patch.dict(STEALTH_FRACTION, {name: 0.0 for name in anomalies}):
        for repetition in range(per_anomaly):
            for variant_index, anomaly_name in enumerate(anomalies):
                device = next_device()
                local_seed = seed + 200_000 + repetition * 101 + variant_index
                random.seed(local_seed)
                spec = ScenarioEngine(local_seed).synchronized_anomaly(anomaly_name)
                events, context = EventEngine(
                    device_id=device.device_id, gateway_id="gateway", auth_server_id="auth_server",
                    broker_id="broker", start_time=next_start_time(), persistent_context=device,
                ).execute(spec)
                accept(events, context, "anomaly")

    for index in range(intent_only_count + malformed_count):
        device = next_device()
        local_seed = seed + 300_000 + index
        random.seed(local_seed)
        spec = ScenarioEngine(local_seed).normal()
        spec.generation_profile = SYNCHRONIZED_GENERATION_PROFILE
        events, context = EventEngine(
            device_id=device.device_id, gateway_id="gateway", auth_server_id="auth_server",
            broker_id="broker", start_time=next_start_time(), persistent_context=device,
        ).execute(spec)
        variant = HN_VARIANTS[index % len(HN_VARIANTS)]
        candidate_id = f"audit-{seed}-adversarial-{index}"
        if candidate_id in candidate_ids:
            raise AssertionError("duplicate HN candidate ID")
        candidate_ids.add(candidate_id)
        candidate = HardNegativeQualificationRecord(
            hard_negative_candidate_id=candidate_id, trace_id=context.trace_id,
            scenario_id=context.scenario_id, device_id=context.device_id,
            candidate_status=CandidateStatus.REQUESTED,
            primary_intended_family_id=variant.family_id,
            intended_variant_id=variant.variant_id,
            intended_boundary_invariant_ids=variant.invariant_ids,
            construction_status=ConstructionStatus.REQUESTED_NOT_APPLIED,
            construction_parameters={"audit": "intent_only"},
        )
        report = validate_synchronized_dataset(
            [(events, context)], provenance=provenance,
            field_manifest=manifest, dataset_manifest=dataset_manifest,
        )
        if index < intent_only_count:
            qualified = qualify_hard_negative(events, context, candidate, report)
            records.append(qualified)
            accept(events, context, "intent_only", family=variant.family_id)
            if qualified.qualification_status is QualificationStatus.QUALIFIED:
                raise AssertionError("intent-only candidate qualified")
        else:
            malformed = replace(candidate, primary_intended_family_id="unknown-family")
            qualified = qualify_hard_negative(events, context, malformed, report)
            malformed_statuses[qualified.qualification_status.value] += 1
            gated = validate_synchronized_dataset(
                [(events, context)], provenance=provenance,
                field_manifest=manifest, dataset_manifest=dataset_manifest,
                hard_negative_qualification_records=(qualified,),
            )
            malformed_gate_failures += not gated.build_gate_passed
            by_kind["malformed"].append((events, context))
            semantic_rows.append({"kind": "malformed", "family": variant.family_id, **_structure(events, context)})

    aggregate = validate_synchronized_dataset(
        sequences, provenance=provenance, field_manifest=manifest,
        dataset_manifest=dataset_manifest,
        hard_negative_qualification_records=records,
    )
    if not aggregate.build_gate_passed:
        raise AssertionError(f"mixed build gate failed: {aggregate.mandatory_failure_reasons}")
    for _, context in by_kind["anomaly"]:
        anomaly_statuses[aggregate.trace_ground_truth_statuses[context.trace_id]] += 1
    if anomaly_statuses != {"validated_positive": per_anomaly * len(anomalies)}:
        raise AssertionError(f"anomaly status mismatch: {anomaly_statuses}")
    if malformed_gate_failures != malformed_count:
        raise AssertionError("malformed candidate escaped build gate")
    if qualified_positive_gt or qualified_invariant_violations or generator_defects:
        raise AssertionError("qualified HN integrity violated")
    if any(not device.enrolled or device.completed_trace_count < 2 for device in persistent):
        raise AssertionError("persistent device context lost enrollment or reuse")
    last_end_by_device: dict[str, float] = {}
    for events, context in sequences:
        previous_end = last_end_by_device.get(context.device_id)
        if previous_end is not None and events[0].timestamp <= previous_end:
            raise AssertionError("reused device's trace did not follow its previous trace")
        last_end_by_device[context.device_id] = events[-1].timestamp

    detector_fields = field_names_for_view("detector_observations")
    privileged = ("hard_negative", "candidate_id", "family_id", "variant_id", "qualification", "epsilon_hn")
    if any(any(word in field for word in privileged) for field in detector_fields):
        raise AssertionError("HN metadata leaked into detector field names")
    observed = to_detector_observation_df(sequences)
    if any(
        str(value) in candidate_ids or "hn3" in str(value).lower()
        or "hn4" in str(value).lower()
        for value in observed.to_numpy().flat
    ):
        raise AssertionError("HN-specific detector value marker")

    summary = HardNegativeRunSummary.from_records(
        total_traces=trace_index, ordinary_normal_traces=ordinary_count,
        records=records, build_gate_result=aggregate.build_gate_passed,
    ).to_dict()
    ordinary_lengths = {len(events) for events, _ in by_kind["ordinary_normal"]}
    family_fingerprints = {
        family.family_id: {
            "lengths": sorted({len(events) for events, _ in by_family[family.family_id]}),
            "length_exclusive_to_family_vs_ordinary": sorted({len(events) for events, _ in by_family[family.family_id]} - ordinary_lengths),
            "resource_ids": len({event.resource_id for events, _ in by_family[family.family_id] for event in events if event.resource_id}),
            "topics": len({event.topic for events, _ in by_family[family.family_id] for event in events if event.topic}),
        }
        for family in HN_FAMILIES
    }
    semantic_payload = json.dumps(semantic_rows, sort_keys=True, separators=(",", ":"))
    return {
        "seed": seed,
        "total_traces": trace_index,
        "device_count": device_count,
        "device_trace_counts": {item.device_id: item.completed_trace_count for item in persistent},
        "family_devices": {key: dict(value) for key, value in family_devices.items()},
        "ordinary_normal_traces": ordinary_count,
        "qualified_hn_traces": per_variant * len(HN_VARIANTS),
        "anomaly_traces": per_anomaly * len(anomalies),
        "intent_only_traces": intent_only_count,
        "malformed_traces": malformed_count,
        "anomaly_statuses": dict(anomaly_statuses),
        "malformed_statuses": dict(malformed_statuses),
        "malformed_gate_failures": malformed_gate_failures,
        "qualified_positive_gt": qualified_positive_gt,
        "qualified_invariant_violations": qualified_invariant_violations,
        "generator_defects": generator_defects,
        "time_margin_unique_rounded_6": len({round(value, 6) for value in time_margins}),
        "time_margin_range_s": [min(time_margins), max(time_margins)] if time_margins else [],
        "candidate_id_count": len(candidate_ids),
        "build_gate_passed": aggregate.build_gate_passed,
        "config_hash": provenance.configuration_snapshot_reference,
        "semantic_sha256": hashlib.sha256(semantic_payload.encode()).hexdigest(),
        "summary": summary,
        "structure_by_kind": {key: _statistics(value) for key, value in by_kind.items()},
        "family_fingerprints": family_fingerprints,
    }


if __name__ == "__main__":
    print(json.dumps(run_campaign(), sort_keys=True))
