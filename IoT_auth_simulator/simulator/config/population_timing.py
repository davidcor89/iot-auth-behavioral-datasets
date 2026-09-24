"""Controlled-synthetic configuration for profile-conditioned timing.

Concrete seconds and realization policy belong here, not in DeviceProfile.T.
The defaults reproduce the active historical EventEngine baselines; they are
invariant constrained, uncalibrated, and make no physical-network claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from simulator.event_model import EventType


TIMING_CONFIGURATION_VERSION = "4c5-controlled-synthetic-timing-v1"
TIMING_CONFIGURATION_FINGERPRINT_VERSION = "4c5-timing-config-sha256-v1"
TIMING_CALIBRATION_STATUS = (
    "controlled_synthetic_invariant_constrained_uncalibrated_calibration_ready"
)


@dataclass(frozen=True)
class GaussianDelayBaseline:
    """One clamped-Gaussian baseline in synthetic seconds."""

    event_type: EventType
    center_s: float
    dispersion_s: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.center_s)
            or not math.isfinite(self.dispersion_s)
            or self.center_s < 0
            or self.dispersion_s < 0
        ):
            raise ValueError("timing baselines must be finite and nonnegative")


DEFAULT_GAUSSIAN_BASELINES: tuple[GaussianDelayBaseline, ...] = (
    GaussianDelayBaseline(EventType.DISCOVERY, 0.0, 0.0),
    GaussianDelayBaseline(EventType.GATEWAY_ADVERTISEMENT, 0.12, 0.04),
    GaussianDelayBaseline(EventType.PAIRING_REQUEST, 0.10, 0.03),
    GaussianDelayBaseline(EventType.PAIRING_RESPONSE, 0.18, 0.06),
    GaussianDelayBaseline(EventType.ENROLLMENT_REQUEST, 0.15, 0.05),
    GaussianDelayBaseline(EventType.ENROLLMENT_CONFIRMED, 0.20, 0.07),
    GaussianDelayBaseline(EventType.REGISTRATION_REQUEST, 0.0, 0.0),
    GaussianDelayBaseline(EventType.REGISTRATION_CONFIRMED, 0.08, 0.02),
    GaussianDelayBaseline(EventType.AUTHENTICATION_REQUEST, 0.50, 0.30),
    GaussianDelayBaseline(EventType.CHALLENGE_SENT, 0.07, 0.02),
    GaussianDelayBaseline(EventType.NONCE_RECEIVED, 0.15, 0.05),
    GaussianDelayBaseline(EventType.RESPONSE_SENT, 0.25, 0.08),
    GaussianDelayBaseline(EventType.AUTHENTICATION_SUCCESS, 0.06, 0.02),
    GaussianDelayBaseline(EventType.AUTHENTICATION_FAILURE, 0.06, 0.02),
    GaussianDelayBaseline(EventType.TOKEN_ISSUED, 0.08, 0.02),
    GaussianDelayBaseline(EventType.TOKEN_PRESENTED, 0.12, 0.04),
    GaussianDelayBaseline(EventType.TOKEN_VALIDATED, 0.06, 0.02),
    GaussianDelayBaseline(EventType.SESSION_OPENED, 0.05, 0.01),
    GaussianDelayBaseline(EventType.ACCESS_REQUEST, 2.00, 1.50),
    GaussianDelayBaseline(EventType.ACCESS_GRANTED, 0.05, 0.01),
    GaussianDelayBaseline(EventType.ACCESS_DENIED, 0.05, 0.01),
    GaussianDelayBaseline(EventType.SESSION_CLOSED, 30.0, 20.0),
    GaussianDelayBaseline(EventType.RENEWAL_REQUEST, 240.0, 30.0),
    GaussianDelayBaseline(EventType.TOKEN_EXPIRED, 300.0, 10.0),
    GaussianDelayBaseline(EventType.TIMEOUT, 30.0, 5.0),
    GaussianDelayBaseline(EventType.DISCONNECT, 5.0, 2.0),
)


@dataclass(frozen=True)
class PopulationTimingConfiguration:
    """Versioned physical/synthetic timing policy, separate from profile T."""

    configuration_version: str = TIMING_CONFIGURATION_VERSION
    calibration_status: str = TIMING_CALIBRATION_STATUS
    family_mapping_version: str = "4c5-structural-relation-family-map-v1"
    realization_algorithm_version: str = "4c5-relation-aware-realizer-v1"
    rng_derivation_version: str = "4c5-timing-sha256-box-muller-v1"
    semantic_envelope_version: str = "4c5-existing-validity-envelope-v1"
    lower_support_policy: str = "clamp_nonnegative_then_minimum_causal_increment"
    token_validity_safety_margin_s: float = 0.000001
    minimum_causal_increment_s: float = 0.000001
    retry_base_backoff_s: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_jitter_fraction: float = 0.20
    retry_positive_floor_s: float = 0.05
    gaussian_baselines: tuple[GaussianDelayBaseline, ...] = DEFAULT_GAUSSIAN_BASELINES

    def __post_init__(self) -> None:
        identifiers = (
            self.configuration_version,
            self.calibration_status,
            self.family_mapping_version,
            self.realization_algorithm_version,
            self.rng_derivation_version,
            self.semantic_envelope_version,
            self.lower_support_policy,
        )
        if not all(identifiers):
            raise ValueError("timing policy identifiers are required")
        if self.calibration_status != TIMING_CALIBRATION_STATUS:
            raise ValueError("4C.5 timing is controlled synthetic and uncalibrated")
        positive = (
            self.token_validity_safety_margin_s,
            self.minimum_causal_increment_s,
            self.retry_base_backoff_s,
            self.retry_backoff_factor,
            self.retry_jitter_fraction,
            self.retry_positive_floor_s,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("timing policy numeric controls must be finite and positive")
        event_types = [baseline.event_type for baseline in self.gaussian_baselines]
        if len(event_types) != len(set(event_types)):
            raise ValueError("timing baseline event types must be unique")

    def baseline_for(self, event_type: EventType) -> GaussianDelayBaseline:
        for baseline in self.gaussian_baselines:
            if baseline.event_type is event_type:
                return baseline
        raise KeyError(f"no profile-conditioned timing baseline for {event_type.value}")

    def normalized_record(self) -> dict[str, object]:
        record = asdict(self)
        record["gaussian_baselines"] = [
            {
                "event_type": baseline.event_type.value,
                "center_s": baseline.center_s,
                "dispersion_s": baseline.dispersion_s,
            }
            for baseline in self.gaussian_baselines
        ]
        return record

    def fingerprint(self) -> str:
        payload = {
            "fingerprint_version": TIMING_CONFIGURATION_FINGERPRINT_VERSION,
            "configuration": self.normalized_record(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
