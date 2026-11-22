"""Deterministic post-trace features from the frozen O03 observation view.

This module accepts only the eleven detector-observable O03 fields.  It has no
GT, provenance, Device Profile, validator, or scheduler input and performs no
dataset-level fitting.  Semantic-relation features are kept explicit so the
later C2 representation can omit direct rule proxies without calling legitimate
observable evidence privileged leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from simulator.event_model import EventResult, EventType


FEATURE_SPEC_VERSION = "icds-o03-trace-feature-contract-v1"
FEATURE_EXTRACTOR_VERSION = "icds-o03-trace-feature-extractor-v1"
STRUCTURAL_AUDIT_VERSION = "icds-o03-phase1-structural-audit-v1"
EXPECTED_CANDIDATE_SCHEMA = "4d4c3b-timing-nonce-candidate-v1"

O03_FIELDS = (
    "event_type",
    "observed_timestamp",
    "observed_delay_since_previous_event",
    "result",
    "failure_reason",
    "retry_count",
    "token_id",
    "nonce",
    "topic",
    "resource_id",
    "requested_action",
)
EVENT_TYPE_VOCABULARY = tuple(item.value for item in EventType)
RESULT_VOCABULARY = tuple(item.value for item in EventResult)


class TraceFeatureContractError(ValueError):
    """An input cannot be represented faithfully under the Phase-1 contract."""


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    description: str
    source_o03_fields: tuple[str, ...]
    transformation: str
    output_type: str
    temporal_scope: str
    complete_trace_required: bool
    uses_future_information_relative_to_event_time: bool
    transformation_kind: str
    feature_class: str
    feature_group: str
    predicate_equivalent_direct_rule_proxy: bool
    c1_included: bool
    c2_included: bool
    privileged_information_dependency: bool
    assumptions: tuple[str, ...]
    missing_non_applicable_handling: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for field in ("source_o03_fields", "assumptions"):
            record[field] = list(record[field])
        return record


_COMMON = {
    "temporal_scope": "complete_trace",
    "complete_trace_required": True,
    "uses_future_information_relative_to_event_time": True,
    "transformation_kind": "deterministic_per_trace",
    "privileged_information_dependency": False,
}


def _definition(
    name: str,
    description: str,
    sources: tuple[str, ...],
    transformation: str,
    output_type: str,
    group: str,
    *,
    semantic: bool = False,
    predicate_proxy: bool = False,
    c2: bool = True,
    assumptions: tuple[str, ...] = (),
    missing: str = "not applicable; deterministic numeric zero",
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        description=description,
        source_o03_fields=sources,
        transformation=transformation,
        output_type=output_type,
        feature_class="semantic_relation_feature" if semantic else "generic_observable_summary",
        feature_group=group,
        predicate_equivalent_direct_rule_proxy=predicate_proxy,
        c1_included=True,
        c2_included=c2,
        assumptions=assumptions,
        missing_non_applicable_handling=missing,
        **_COMMON,
    )


def _feature_definitions() -> tuple[FeatureDefinition, ...]:
    definitions: list[FeatureDefinition] = [
        _definition(
            "trace_event_count", "Number of O03 observations in the trace.",
            ("event_type",), "length of the ordered O03 trace", "integer",
            "generic_event_structure",
        )
    ]
    for value in EVENT_TYPE_VOCABULARY:
        definitions.extend((
            _definition(
                f"event_type_{value}_count",
                f"Count of schema-defined `{value}` observations.",
                ("event_type",), f"count(event_type == {value!r})", "integer",
                "generic_event_structure",
                assumptions=("EventType is the fixed implementation-schema vocabulary, not a canonical Behavioral-model vocabulary.",),
            ),
            _definition(
                f"event_type_{value}_frequency",
                f"Frequency of schema-defined `{value}` observations.",
                ("event_type",), f"count(event_type == {value!r}) / trace_event_count",
                "number", "generic_event_structure",
                assumptions=("EventType is the fixed implementation-schema vocabulary, not a canonical Behavioral-model vocabulary.",),
            ),
        ))
    for value in RESULT_VOCABULARY:
        definitions.extend((
            _definition(
                f"result_{value}_count",
                f"Count of schema-defined `{value}` results.",
                ("result",), f"count(result == {value!r})", "integer",
                "generic_result",
            ),
            _definition(
                f"result_{value}_frequency",
                f"Frequency of schema-defined `{value}` results.",
                ("result",), f"count(result == {value!r}) / trace_event_count",
                "number", "generic_result",
            ),
        ))
    definitions.extend((
        _definition(
            "failure_reason_present_count", "Observations carrying any failure reason.",
            ("failure_reason",), "count(non-null failure_reason)", "integer",
            "generic_failure_reason",
            assumptions=("Failure-reason strings have no declared closed vocabulary; literal values are not encoded.",),
        ),
        _definition(
            "failure_reason_present_frequency", "Fraction carrying any failure reason.",
            ("failure_reason",), "count(non-null failure_reason) / trace_event_count",
            "number", "generic_failure_reason",
            assumptions=("Failure-reason strings have no declared closed vocabulary; literal values are not encoded.",),
        ),
        _definition(
            "failure_reason_distinct_count", "Distinct non-null failure reasons.",
            ("failure_reason",), "cardinality(non-null failure_reason)", "integer",
            "generic_failure_reason",
            assumptions=("This is a cardinality summary, not a learned categorical vocabulary.",),
        ),
        _definition(
            "retry_count_sum", "Sum of observable cumulative retry counts.",
            ("retry_count",), "sum(retry_count)", "integer", "generic_retry",
            assumptions=("retry_count is required integer trace-prefix evidence on every O03 row.",),
        ),
        _definition(
            "retry_count_mean", "Mean observable cumulative retry count.",
            ("retry_count",), "mean(retry_count)", "number", "generic_retry",
            assumptions=("retry_count is required integer trace-prefix evidence on every O03 row.",),
        ),
        _definition(
            "retry_count_max", "Maximum observable cumulative retry count.",
            ("retry_count",), "max(retry_count)", "integer", "generic_retry",
            assumptions=("retry_count is required integer trace-prefix evidence on every O03 row.",),
        ),
        _definition(
            "topic_distinct_count", "Distinct non-null observed topic strings.",
            ("topic",), "cardinality(non-null topic)", "integer", "generic_resource",
            assumptions=("Literal topic identities are not encoded.",),
        ),
        _definition(
            "resource_distinct_count", "Distinct non-null observed resource strings.",
            ("resource_id",), "cardinality(non-null resource_id)", "integer",
            "generic_resource", assumptions=("Literal resource identities are not encoded.",),
        ),
        _definition(
            "requested_action_present_count", "Observations carrying a requested action.",
            ("requested_action",), "count(non-null requested_action)", "integer",
            "generic_resource",
            assumptions=("Requested actions have no declared closed dataset-contract vocabulary; literal values are not encoded.",),
        ),
        _definition(
            "requested_action_present_frequency", "Fraction carrying a requested action.",
            ("requested_action",), "count(non-null requested_action) / trace_event_count",
            "number", "generic_resource",
            assumptions=("Requested actions have no declared closed dataset-contract vocabulary; literal values are not encoded.",),
        ),
        _definition(
            "requested_action_distinct_count", "Distinct non-null requested actions.",
            ("requested_action",), "cardinality(non-null requested_action)", "integer",
            "generic_resource",
            assumptions=("This is a cardinality summary, not a learned categorical vocabulary.",),
        ),
        _definition(
            "observed_trace_duration", "Maximum minus minimum observed timestamp.",
            ("observed_timestamp",), "max(observed_timestamp) - min(observed_timestamp)",
            "number", "generic_timing",
        ),
        _definition(
            "observed_inter_event_delay_count", "Number of actual inter-event gaps.",
            ("observed_delay_since_previous_event",), "trace_event_count - 1", "integer",
            "generic_timing",
            assumptions=("The first row's exporter-defined zero is excluded because it is not an inter-event gap.",),
        ),
    ))
    for statistic in ("min", "max", "mean", "median", "population_stddev"):
        definitions.append(_definition(
            f"observed_inter_event_delay_{statistic}",
            f"{statistic.replace('_', ' ').title()} of actual observed inter-event delays.",
            ("observed_delay_since_previous_event",),
            f"{statistic}(observed_delay_since_previous_event[rows 2..N])",
            "number|null", "generic_timing",
            assumptions=("The first row's exporter-defined zero is excluded because it is not an inter-event gap.",),
            missing="null when fewer than two observations; no dataset-level imputation is performed",
        ))
    definitions.extend((
        _definition(
            "nonce_bearing_observation_count", "Observations carrying a nonce value.",
            ("nonce",), "count(non-null nonce)", "integer", "generic_nonce_structure",
        ),
        _definition(
            "nonce_distinct_value_count", "Distinct observed nonce values.",
            ("nonce",), "cardinality(non-null nonce)", "integer", "generic_nonce_structure",
        ),
        _definition(
            "nonce_repeated_value_count", "Nonce values occurring on more than one observation.",
            ("nonce",), "count(values with multiplicity > 1)", "integer",
            "generic_nonce_structure",
            assumptions=("Repetition includes normal propagation across challenge/receipt/response events and does not infer attempt boundaries.",),
        ),
        _definition(
            "nonce_max_multiplicity", "Maximum observation multiplicity of one nonce.",
            ("nonce",), "max(value multiplicity), or 0 with no nonce", "integer",
            "generic_nonce_structure",
        ),
        _definition(
            "nonce_repeated_observation_count", "Nonce observations beyond each value's first occurrence.",
            ("nonce",), "sum(max(multiplicity - 1, 0))", "integer",
            "generic_nonce_structure",
            assumptions=("This equality summary does not infer authentication attempts.",),
        ),
        _definition(
            "semantic_timing_relation_count",
            "Reconstructed nonce_received to response_sent relations within observable attempt segments.",
            ("event_type", "nonce", "observed_timestamp"),
            "count(first unmatched later response_sent with equal nonce per nonce_received inside each challenge segment)",
            "integer", "semantic_timing_relation", semantic=True,
            assumptions=("For the frozen controlled serial grammar, each challenge_sent begins one observable authentication-attempt segment.",),
        ),
        _definition(
            "semantic_timing_min_signed_causal_margin",
            "Minimum signed response_sent minus nonce_received observed timestamp.",
            ("event_type", "nonce", "observed_timestamp"),
            "min(response_sent.observed_timestamp - nonce_received.observed_timestamp) over reconstructed relations",
            "number|null", "semantic_timing_relation", semantic=True,
            predicate_proxy=True, c2=False,
            assumptions=("For the frozen controlled serial grammar, each challenge_sent begins one observable authentication-attempt segment.",),
            missing="null when no complete observable relation exists; no dataset-level imputation is performed",
        ),
        _definition(
            "semantic_nonce_cross_attempt_value_count",
            "Nonce values present in more than one reconstructed attempt segment.",
            ("event_type", "nonce"),
            "count(nonce values whose reconstructed-attempt cardinality > 1)",
            "integer", "semantic_nonce_relation", semantic=True,
            predicate_proxy=True, c2=False,
            assumptions=("For the frozen controlled serial grammar, each challenge_sent begins one observable authentication-attempt segment.",),
        ),
        _definition(
            "semantic_nonce_max_attempt_sharing_count",
            "Maximum reconstructed-attempt cardinality for any observed nonce.",
            ("event_type", "nonce"),
            "max(number of reconstructed attempts containing each nonce), or 0 with no nonce",
            "integer", "semantic_nonce_relation", semantic=True,
            predicate_proxy=True, c2=False,
            assumptions=("For the frozen controlled serial grammar, each challenge_sent begins one observable authentication-attempt segment.",),
        ),
    ))
    return tuple(definitions)


FEATURE_DEFINITIONS = _feature_definitions()
C1_FEATURE_NAMES = tuple(item.name for item in FEATURE_DEFINITIONS if item.c1_included)
C2_FEATURE_NAMES = tuple(item.name for item in FEATURE_DEFINITIONS if item.c2_included)


def feature_specification() -> dict[str, Any]:
    """Return the complete machine-readable feature contract."""
    return {
        "schema_version": FEATURE_SPEC_VERSION,
        "extractor_version": FEATURE_EXTRACTOR_VERSION,
        "prediction_unit": "one complete canonical scientific trace",
        "prediction_horizon": "post_trace_offline",
        "observation_contract": "c1.6-development-derived-schema",
        "source_o03_fields": list(O03_FIELDS),
        "fixed_vocabularies": {
            "event_type": {
                "source": "simulator.event_model.EventType implementation schema",
                "values": list(EVENT_TYPE_VOCABULARY),
                "data_learned": False,
                "canonical_behavioral_vocabulary_claim": False,
            },
            "result": {
                "source": "simulator.event_model.EventResult implementation schema",
                "values": list(RESULT_VOCABULARY),
                "data_learned": False,
            },
            "failure_reason": {
                "closed_vocabulary": False,
                "policy": "structural presence/cardinality summaries only",
            },
            "requested_action": {
                "closed_vocabulary": False,
                "policy": "structural presence/cardinality summaries only",
            },
        },
        "attempt_reconstruction": {
            "scope": "frozen controlled serial candidate grammar",
            "boundary_event": "challenge_sent",
            "rule": "each challenge_sent begins a new ordered attempt segment",
            "privileged_auth_attempt_id_used": False,
            "generalization_status": "researcher_review_required_before_other datasets",
        },
        "profiles": {
            "C1": {
                "description": "evidence-rich O03 trace representation",
                "features": list(C1_FEATURE_NAMES),
            },
            "C2": {
                "description": "relation-proxy-controlled O03 trace representation",
                "features": list(C2_FEATURE_NAMES),
                "excluded_direct_predicate_features": [
                    item.name for item in FEATURE_DEFINITIONS
                    if item.c1_included and not item.c2_included
                ],
            },
        },
        "learned_transformations": {
            "implemented_in_phase_1": False,
            "later_train_only_operations": [
                "imputation fitted from samples",
                "scaling",
                "data-inferred categorical vocabularies",
                "constant or near-constant filtering",
                "feature selection",
                "learned encoding",
            ],
        },
        "features": [item.to_record() for item in FEATURE_DEFINITIONS],
    }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_observations(observations: Sequence[Mapping[str, Any]]) -> None:
    if not observations:
        raise TraceFeatureContractError("a complete trace must contain at least one O03 observation")
    expected = set(O03_FIELDS)
    for index, row in enumerate(observations):
        actual = set(row)
        if actual != expected:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            raise TraceFeatureContractError(
                f"O03 row {index} field mismatch; extra={extra}; missing={missing}"
            )
        if row["event_type"] not in EVENT_TYPE_VOCABULARY:
            raise TraceFeatureContractError(f"unknown event_type at row {index}: {row['event_type']!r}")
        if row["result"] not in RESULT_VOCABULARY:
            raise TraceFeatureContractError(f"unknown result at row {index}: {row['result']!r}")
        for field in ("observed_timestamp", "observed_delay_since_previous_event"):
            if not _is_number(row[field]):
                raise TraceFeatureContractError(f"{field} at row {index} must be finite numeric evidence")
        retry = row["retry_count"]
        if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
            raise TraceFeatureContractError(f"retry_count at row {index} must be a nonnegative integer")
        for field in ("failure_reason", "token_id", "nonce", "topic", "resource_id", "requested_action"):
            if row[field] is not None and not isinstance(row[field], str):
                raise TraceFeatureContractError(f"{field} at row {index} must be string or null")


def _attempt_segments(observations: Sequence[Mapping[str, Any]]) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """Reconstruct candidate attempts solely from observable challenge markers."""
    attempts: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] | None = None
    for index, row in enumerate(observations):
        if row["event_type"] == EventType.CHALLENGE_SENT.value:
            current = []
            attempts.append(current)
        if current is not None:
            current.append(row)
        elif row["nonce"] is not None:
            raise TraceFeatureContractError(
                f"nonce-bearing row {index} occurs before any observable challenge_sent boundary"
            )
    return tuple(tuple(attempt) for attempt in attempts)


def _timing_relations(attempts: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[float, ...]:
    margins: list[float] = []
    for attempt in attempts:
        used_responses: set[int] = set()
        for source_index, source in enumerate(attempt):
            if source["event_type"] != EventType.NONCE_RECEIVED.value or source["nonce"] is None:
                continue
            for target_index in range(source_index + 1, len(attempt)):
                target = attempt[target_index]
                if (
                    target_index not in used_responses
                    and target["event_type"] == EventType.RESPONSE_SENT.value
                    and target["nonce"] == source["nonce"]
                ):
                    used_responses.add(target_index)
                    margins.append(
                        float(target["observed_timestamp"])
                        - float(source["observed_timestamp"])
                    )
                    break
    return tuple(margins)


def extract_trace_features(
    observations: Sequence[Mapping[str, Any]], *, profile: str = "C1",
) -> dict[str, int | float | None]:
    """Extract one deterministic feature vector from one ordered O03 trace."""
    if profile not in {"C1", "C2"}:
        raise ValueError("profile must be 'C1' or 'C2'")
    _validate_observations(observations)

    event_count = len(observations)
    event_types = Counter(str(row["event_type"]) for row in observations)
    results = Counter(str(row["result"]) for row in observations)
    failure_reasons = [str(row["failure_reason"]) for row in observations if row["failure_reason"] is not None]
    retries = [int(row["retry_count"]) for row in observations]
    timestamps = [float(row["observed_timestamp"]) for row in observations]
    delays = [float(row["observed_delay_since_previous_event"]) for row in observations[1:]]
    nonces = [str(row["nonce"]) for row in observations if row["nonce"] is not None]
    nonce_counts = Counter(nonces)
    attempts = _attempt_segments(observations)
    margins = _timing_relations(attempts)
    nonce_attempts: dict[str, set[int]] = defaultdict(set)
    for attempt_index, attempt in enumerate(attempts):
        for nonce in {str(row["nonce"]) for row in attempt if row["nonce"] is not None}:
            nonce_attempts[nonce].add(attempt_index)

    features: dict[str, int | float | None] = {"trace_event_count": event_count}
    for value in EVENT_TYPE_VOCABULARY:
        count = event_types[value]
        features[f"event_type_{value}_count"] = count
        features[f"event_type_{value}_frequency"] = count / event_count
    for value in RESULT_VOCABULARY:
        count = results[value]
        features[f"result_{value}_count"] = count
        features[f"result_{value}_frequency"] = count / event_count
    features.update({
        "failure_reason_present_count": len(failure_reasons),
        "failure_reason_present_frequency": len(failure_reasons) / event_count,
        "failure_reason_distinct_count": len(set(failure_reasons)),
        "retry_count_sum": sum(retries),
        "retry_count_mean": statistics.fmean(retries),
        "retry_count_max": max(retries),
        "topic_distinct_count": len({row["topic"] for row in observations if row["topic"] is not None}),
        "resource_distinct_count": len({row["resource_id"] for row in observations if row["resource_id"] is not None}),
        "requested_action_present_count": sum(row["requested_action"] is not None for row in observations),
        "requested_action_present_frequency": sum(row["requested_action"] is not None for row in observations) / event_count,
        "requested_action_distinct_count": len({row["requested_action"] for row in observations if row["requested_action"] is not None}),
        "observed_trace_duration": max(timestamps) - min(timestamps),
        "observed_inter_event_delay_count": len(delays),
        "observed_inter_event_delay_min": min(delays) if delays else None,
        "observed_inter_event_delay_max": max(delays) if delays else None,
        "observed_inter_event_delay_mean": statistics.fmean(delays) if delays else None,
        "observed_inter_event_delay_median": statistics.median(delays) if delays else None,
        "observed_inter_event_delay_population_stddev": statistics.pstdev(delays) if delays else None,
        "nonce_bearing_observation_count": len(nonces),
        "nonce_distinct_value_count": len(nonce_counts),
        "nonce_repeated_value_count": sum(value > 1 for value in nonce_counts.values()),
        "nonce_max_multiplicity": max(nonce_counts.values(), default=0),
        "nonce_repeated_observation_count": sum(max(value - 1, 0) for value in nonce_counts.values()),
        "semantic_timing_relation_count": len(margins),
        "semantic_timing_min_signed_causal_margin": min(margins) if margins else None,
        "semantic_nonce_cross_attempt_value_count": sum(
            len(attempt_ids) > 1 for attempt_ids in nonce_attempts.values()
        ),
        "semantic_nonce_max_attempt_sharing_count": max(
            (len(attempt_ids) for attempt_ids in nonce_attempts.values()), default=0,
        ),
    })
    names = C1_FEATURE_NAMES if profile == "C1" else C2_FEATURE_NAMES
    if set(features) != set(C1_FEATURE_NAMES):
        raise AssertionError("extractor and feature specification are out of sync")
    return {name: features[name] for name in names}


def extract_candidate_feature_rows(
    candidate: Mapping[str, Any], *, profile: str = "C1",
) -> list[dict[str, Any]]:
    """Extract features while keeping the privileged join key outside the vector."""
    if candidate.get("schema_version") != EXPECTED_CANDIDATE_SCHEMA:
        raise TraceFeatureContractError("unexpected scientific candidate schema")
    samples = candidate.get("samples")
    if not isinstance(samples, list) or candidate.get("sample_count") != len(samples):
        raise TraceFeatureContractError("candidate sample count is inconsistent")
    return [
        {
            "sample_reference": sample["sample_id"],
            "features": extract_trace_features(sample["detector_observations"], profile=profile),
        }
        for sample in samples
    ]


def audit_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Run structural, label-independent extraction checks without ML analysis."""
    first_c1 = extract_candidate_feature_rows(candidate, profile="C1")
    second_c1 = extract_candidate_feature_rows(candidate, profile="C1")
    c2 = extract_candidate_feature_rows(candidate, profile="C2")
    if len(first_c1) != 336:
        raise TraceFeatureContractError("frozen candidate must contain exactly 336 traces")
    by_reference = {row["sample_reference"]: row["features"] for row in first_c1}
    reversed_rows = extract_candidate_feature_rows(
        {**candidate, "samples": list(reversed(candidate["samples"]))}, profile="C1",
    )
    if by_reference != {row["sample_reference"]: row["features"] for row in reversed_rows}:
        raise AssertionError("dataset trace order changed per-trace features")
    missing = Counter(
        name for row in first_c1 for name, value in row["features"].items() if value is None
    )
    timing_support = sum(
        row["features"]["semantic_timing_relation_count"] > 0 for row in first_c1
    )
    attempt_support = sum(
        row["features"]["semantic_nonce_max_attempt_sharing_count"] > 0 for row in first_c1
    )
    audit = {
        "schema_version": STRUCTURAL_AUDIT_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "extractor_version": FEATURE_EXTRACTOR_VERSION,
        "candidate_schema_version": candidate["schema_version"],
        "candidate_fingerprint": candidate.get("candidate_fingerprint"),
        "traces_processed": len(first_c1),
        "extraction_failures": 0,
        "feature_counts": {"C1": len(C1_FEATURE_NAMES), "C2": len(C2_FEATURE_NAMES)},
        "c2_is_subset_of_c1": set(C2_FEATURE_NAMES) <= set(C1_FEATURE_NAMES),
        "c2_excluded_features": sorted(set(C1_FEATURE_NAMES) - set(C2_FEATURE_NAMES)),
        "missing_or_non_applicable_counts_c1": dict(sorted(missing.items())),
        "observable_grammar_support": {
            "traces_with_reconstructed_timing_relation": timing_support,
            "traces_without_reconstructed_timing_relation": len(first_c1) - timing_support,
            "traces_with_observable_challenge_attempt": attempt_support,
            "traces_without_observable_challenge_attempt": len(first_c1) - attempt_support,
        },
        "determinism": {
            "repeat_extraction_identical": first_c1 == second_c1,
            "trace_order_independent": True,
            "c1_feature_rows_fingerprint": _canonical_hash(first_c1),
            "c2_feature_rows_fingerprint": _canonical_hash(c2),
        },
        "dataset_level_fitting_performed": False,
        "labels_or_privileged_metadata_read_by_extractor": False,
        "predictive_analysis_performed": False,
    }
    audit["audit_fingerprint"] = _canonical_hash(audit)
    return audit


def write_feature_specification(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(feature_specification(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--spec-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    write_feature_specification(args.spec_out)
    audit = audit_candidate(candidate)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "traces": audit["traces_processed"],
        "c1_features": audit["feature_counts"]["C1"],
        "c2_features": audit["feature_counts"]["C2"],
        "audit_fingerprint": audit["audit_fingerprint"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
