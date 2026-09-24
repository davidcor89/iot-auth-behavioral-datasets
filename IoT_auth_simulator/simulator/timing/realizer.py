"""Deterministic timing realization subordinate to existing semantic bounds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from simulator.config.population_timing import PopulationTimingConfiguration
from simulator.device_profiles.profile import RelativeTimingTendency, TimingTendencies
from simulator.event_model import EventType


TIMING_PROVENANCE_SCHEMA_VERSION = "4c5-trace-timing-provenance-v1"
TIMING_ROOT_SLOT = 5
TIMING_ROOT_NAMESPACE = "device-profile-timing-realization"


class TimingFamily(str, Enum):
    AUTH = "AUTH"
    TRANSITION = "TRANSITION"
    ACCESS = "ACCESS"
    RENEWAL = "RENEWAL"


@dataclass(frozen=True)
class TimingRelation:
    """Stable structural identity for one already-planned target event."""

    step_index: int
    segment: str
    event_type: EventType
    family: TimingFamily
    occurrence: int
    relation_identity: str


class EmptyTimingEnvelopeError(ValueError):
    """The selected structure cannot fit an existing temporal constraint."""

    def __init__(self, relation_identity: str, detail: str):
        self.relation_identity = relation_identity
        self.detail = detail
        super().__init__(f"empty_timing_envelope:{relation_identity}:{detail}")


def _stable_reference(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def derive_timing_root_reference(
    planning_rng_reference: str,
    configuration: PopulationTimingConfiguration,
) -> str:
    """Reserve conceptual slot 5 without consuming the planning RNG stream."""
    if not planning_rng_reference:
        raise ValueError("planning RNG reference is required for timing")
    return _stable_reference({
        "namespace": TIMING_ROOT_NAMESPACE,
        "planning_rng_reference": planning_rng_reference,
        "planning_decision_slot": TIMING_ROOT_SLOT,
        "rng_derivation_version": configuration.rng_derivation_version,
        "timing_configuration_fingerprint": configuration.fingerprint(),
    })


def derive_relation_randomness_reference(
    timing_root_reference: str,
    relation: TimingRelation,
    configuration: PopulationTimingConfiguration,
) -> str:
    return _stable_reference({
        "timing_root_reference": timing_root_reference,
        "relation_identity": relation.relation_identity,
        "family": relation.family.value,
        "rng_derivation_version": configuration.rng_derivation_version,
    })


def _open_unit(reference: str, component: str) -> float:
    digest = hashlib.sha256(f"{reference}|{component}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / (2**64)


def normal_variate_from_reference(reference: str) -> float:
    """Deterministic standard normal using a versioned Box-Muller transform."""
    u1 = _open_unit(reference, "normal-u1")
    u2 = _open_unit(reference, "normal-u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def signed_uniform_from_reference(reference: str) -> float:
    """Deterministic uniform variate on the open interval (-1, 1)."""
    return 2.0 * _open_unit(reference, "retry-jitter") - 1.0


def map_timing_relations(event_steps: Iterable[EventType]) -> tuple[TimingRelation, ...]:
    """Map structure, not EventType alone, onto the four frozen T families."""
    relations: list[TimingRelation] = []
    occurrences: dict[tuple[str, EventType], int] = {}
    initial_auth_started = False
    initial_auth_succeeded = False
    renewal_started = False

    for index, event_type in enumerate(event_steps):
        if event_type is EventType.RENEWAL_REQUEST:
            renewal_started = True

        if renewal_started:
            segment = "renewal"
            family = TimingFamily.RENEWAL
        elif event_type in {
            EventType.ACCESS_REQUEST,
            EventType.ACCESS_GRANTED,
            EventType.ACCESS_DENIED,
        }:
            segment = "access"
            family = TimingFamily.ACCESS
        elif (
            not initial_auth_succeeded
            and (initial_auth_started or event_type is EventType.AUTHENTICATION_REQUEST)
        ):
            initial_auth_started = True
            segment = "initial-auth"
            family = TimingFamily.AUTH
        elif initial_auth_succeeded:
            segment = "token-session-transition"
            family = TimingFamily.TRANSITION
        else:
            segment = "onboarding"
            family = TimingFamily.TRANSITION

        key = (segment, event_type)
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        relation_identity = (
            f"{segment}/{event_type.value}/occurrence-{occurrence}"
        )
        relations.append(TimingRelation(
            step_index=index,
            segment=segment,
            event_type=event_type,
            family=family,
            occurrence=occurrence,
            relation_identity=relation_identity,
        ))

        if (
            family is TimingFamily.AUTH
            and event_type is EventType.AUTHENTICATION_SUCCESS
        ):
            initial_auth_succeeded = True

    return tuple(relations)


_TOKEN_REQUIRED_TARGETS = frozenset({
    EventType.TOKEN_PRESENTED,
    EventType.TOKEN_VALIDATED,
    EventType.ACCESS_REQUEST,
    EventType.ACCESS_GRANTED,
})


class TimingRealizer:
    """Realize one profile-conditioned trace without changing its structure."""

    def __init__(
        self,
        *,
        configuration: PopulationTimingConfiguration,
        timing_profile: TimingTendencies,
        planning_rng_reference: str,
        event_steps: Iterable[EventType],
    ) -> None:
        self.configuration = configuration
        self.timing_profile = timing_profile
        self.relations = map_timing_relations(tuple(event_steps))
        self.timing_root_reference = derive_timing_root_reference(
            planning_rng_reference, configuration,
        )
        self._next_index = 0
        self._records: list[dict[str, object]] = []

    def prepare_for_execution(self, event_steps: Iterable[EventType]) -> None:
        """Bind to EventEngine's persistence-adjusted steps before realization."""
        if self._next_index or self._records:
            raise ValueError("timing relations cannot be rebound after realization starts")
        self.relations = map_timing_relations(tuple(event_steps))

    def _tendency(self, family: TimingFamily) -> RelativeTimingTendency:
        return {
            TimingFamily.AUTH: self.timing_profile.auth,
            TimingFamily.TRANSITION: self.timing_profile.transition,
            TimingFamily.ACCESS: self.timing_profile.access,
            TimingFamily.RENEWAL: self.timing_profile.renewal,
        }[family]

    def _last_required_before_replacement(self, start_index: int) -> int | None:
        last_required: int | None = None
        for index in range(start_index, len(self.relations)):
            event_type = self.relations[index].event_type
            if index > start_index and event_type is EventType.TOKEN_ISSUED:
                break
            if event_type in _TOKEN_REQUIRED_TARGETS:
                last_required = index
        return last_required

    def _candidate(
        self,
        relation: TimingRelation,
        randomness_reference: str,
        retry_count: int,
    ) -> tuple[float, dict[str, object]]:
        tendency = self._tendency(relation.family)
        if relation.event_type is EventType.RETRY:
            base_center = (
                self.configuration.retry_base_backoff_s
                * self.configuration.retry_backoff_factor ** retry_count
            )
            conditioned_center = tendency.center_modifier * base_center
            conditioned_half_width = (
                tendency.dispersion_modifier
                * self.configuration.retry_jitter_fraction
                * base_center
            )
            noise = signed_uniform_from_reference(randomness_reference)
            candidate = max(
                self.configuration.retry_positive_floor_s,
                conditioned_center + noise * conditioned_half_width,
            )
            return candidate, {
                "distribution": "bounded_uniform_retry_jitter",
                "baseline_center_s": base_center,
                "baseline_dispersion_s": (
                    self.configuration.retry_jitter_fraction * base_center
                ),
                "normalized_noise": noise,
            }

        baseline = self.configuration.baseline_for(relation.event_type)
        conditioned_center = tendency.center_modifier * baseline.center_s
        conditioned_dispersion = tendency.dispersion_modifier * baseline.dispersion_s
        noise = normal_variate_from_reference(randomness_reference)
        candidate = max(0.0, conditioned_center + noise * conditioned_dispersion)
        return candidate, {
            "distribution": "clamped_gaussian",
            "baseline_center_s": baseline.center_s,
            "baseline_dispersion_s": baseline.dispersion_s,
            "normalized_noise": noise,
        }

    def realize_delay(
        self,
        event_type: EventType,
        *,
        current_time: float,
        token_expiry: float | None,
        retry_count: int,
    ) -> float:
        if self._next_index >= len(self.relations):
            raise ValueError("timing realizer received more events than the structural plan")
        relation = self.relations[self._next_index]
        if relation.event_type is not event_type:
            raise ValueError(
                "timing relation/event mismatch: expected "
                f"{relation.event_type.value}, received {event_type.value}"
            )
        randomness_reference = derive_relation_randomness_reference(
            self.timing_root_reference, relation, self.configuration,
        )
        candidate, distribution = self._candidate(
            relation, randomness_reference, retry_count,
        )
        lower = self.configuration.minimum_causal_increment_s
        realized = max(candidate, lower)
        intervention_reason: str | None = (
            "minimum_causal_increment"
            if candidate < lower else None
        )
        upper_delay_exclusive: float | None = None

        # TOKEN_ISSUED establishes a new interval only after this delay has
        # elapsed. Never constrain issuance using the token it supersedes.
        last_required = (
            self._last_required_before_replacement(self._next_index)
            if token_expiry is not None
            and relation.event_type is not EventType.TOKEN_ISSUED
            else None
        )
        if last_required is not None:
            remaining_relations = last_required - self._next_index
            upper_delay_exclusive = (
                token_expiry
                - current_time
                - self.configuration.token_validity_safety_margin_s
                - remaining_relations * lower
            )
            if not math.isfinite(upper_delay_exclusive) or upper_delay_exclusive <= lower:
                raise EmptyTimingEnvelopeError(
                    relation.relation_identity,
                    "insufficient remaining token validity for selected structure",
                )
            if realized >= upper_delay_exclusive:
                realized = math.nextafter(upper_delay_exclusive, 0.0)
                intervention_reason = "preserve_existing_token_validity"

        if not math.isfinite(realized) or realized < lower:
            raise EmptyTimingEnvelopeError(
                relation.relation_identity,
                "no finite positive delay remains after envelope application",
            )

        self._records.append({
            "relation_identity": relation.relation_identity,
            "segment": relation.segment,
            "target_event_type": relation.event_type.value,
            "timing_family": relation.family.value,
            "relation_randomness_reference": randomness_reference,
            **distribution,
            "candidate_delay_s": candidate,
            "lower_delay_bound_s": lower,
            "upper_delay_exclusive_s": upper_delay_exclusive,
            "realized_delay_s": realized,
            "envelope_intervention": intervention_reason is not None,
            "envelope_intervention_reason": intervention_reason,
        })
        self._next_index += 1
        return realized

    def assert_complete(self) -> None:
        if self._next_index != len(self.relations):
            raise ValueError(
                f"timing realizer consumed {self._next_index}/{len(self.relations)} relations"
            )

    def provenance_record(self) -> dict[str, object]:
        self.assert_complete()
        reasons: dict[str, int] = {}
        for record in self._records:
            reason = record["envelope_intervention_reason"]
            if reason:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        family_order = tuple(family.value for family in TimingFamily)
        applicable = tuple(
            family for family in family_order
            if any(record["timing_family"] == family for record in self._records)
        )
        return {
            "timing_provenance_schema_version": TIMING_PROVENANCE_SCHEMA_VERSION,
            "timing_configuration_version": self.configuration.configuration_version,
            "timing_configuration_fingerprint": self.configuration.fingerprint(),
            "calibration_status": self.configuration.calibration_status,
            "family_mapping_version": self.configuration.family_mapping_version,
            "timing_realization_algorithm_version": (
                self.configuration.realization_algorithm_version
            ),
            "timing_rng_derivation_version": self.configuration.rng_derivation_version,
            "semantic_envelope_version": self.configuration.semantic_envelope_version,
            "timing_decision_slot": TIMING_ROOT_SLOT,
            "timing_root_reference": self.timing_root_reference,
            "applicable_timing_families": list(applicable),
            "realized_relation_count": len(self._records),
            "envelope_intervention_count": sum(reasons.values()),
            "envelope_intervention_reasons": reasons,
            "relations": list(self._records),
            "profile_tendency_reference": "device_profile_companion via provenance_trace.device_id",
        }
