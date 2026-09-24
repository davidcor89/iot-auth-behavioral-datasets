"""Normalized 4D.4B identities, including detector-evidence hashing."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from simulator.role_integration.models import stable_reference


COMPOSITION_FINGERPRINT_VERSION = "4d4b-composition-fingerprint-sha256-v1"
DETECTOR_EVIDENCE_HASH_VERSION = "4d4b-detector-evidence-hash-v1"
DETECTOR_OBSERVATION_CONTRACT = "c1.6-development-derived-schema"


def detector_evidence_record(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    previous = None
    for item in records:
        observed = float(item.get("observed_timestamp", item["timestamp"]))
        delay = 0.0 if previous is None else observed - previous
        rows.append({
            "event_type": item["event_type"],
            "observed_timestamp": round(observed, 6),
            "observed_delay_since_previous_event": round(delay, 6),
            "result": item["result"],
            "failure_reason": item.get("failure_reason"),
            "retry_count": item.get("retry_count"),
            "token_id": item.get("token_id"),
            "nonce": item.get("nonce"),
            "topic": item.get("topic"),
            "resource_id": item.get("resource_id"),
            "requested_action": item.get("requested_action"),
        })
        previous = observed
    return {
        "hash_version": DETECTOR_EVIDENCE_HASH_VERSION,
        "observation_contract": DETECTOR_OBSERVATION_CONTRACT,
        "rows": rows,
    }


def detector_evidence_hash(records: Iterable[Mapping[str, Any]]) -> str:
    return stable_reference(detector_evidence_record(records))


def composition_fingerprint(payload: Mapping[str, Any]) -> str:
    return stable_reference({
        "fingerprint_version": COMPOSITION_FINGERPRINT_VERSION,
        **dict(payload),
    })
