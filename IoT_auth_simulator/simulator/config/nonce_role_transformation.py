"""Versioned controls for 4D.3.4 cross-attempt nonce integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


NONCE_ROLE_TRANSFORMATION_CONFIGURATION_VERSION = "4d3.4-nonce-config-v1"


@dataclass(frozen=True)
class NonceRoleTransformationConfiguration:
    """Construction identity only; nonce reuse itself consumes no randomness."""

    supported_scope: str = "within_trace_across_authentication_attempts"
    configuration_version: str = NONCE_ROLE_TRANSFORMATION_CONFIGURATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.configuration_version != NONCE_ROLE_TRANSFORMATION_CONFIGURATION_VERSION:
            raise ValueError("unsupported nonce-role transformation configuration version")
        if self.supported_scope != "within_trace_across_authentication_attempts":
            raise ValueError("4D.3.4 supports only the frozen within-trace attempt scope")

    @property
    def configuration_reference(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
