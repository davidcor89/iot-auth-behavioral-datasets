"""Versioned controls for 4D.3.3 access-boundary transformations.

These are local construction controls, not protocol timeouts, population
calibration, prevalence settings, or detector configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


ACCESS_ROLE_TRANSFORMATION_CONFIGURATION_VERSION = "4d3.3-access-config-v1"


@dataclass(frozen=True)
class AccessRoleTransformationConfiguration:
    """Isolated deterministic timing for role-added access evidence."""

    local_event_spacing_s: float = 0.001
    configuration_version: str = ACCESS_ROLE_TRANSFORMATION_CONFIGURATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.configuration_version != ACCESS_ROLE_TRANSFORMATION_CONFIGURATION_VERSION:
            raise ValueError("unsupported access-role transformation configuration version")
        if self.local_event_spacing_s <= 0:
            raise ValueError("local_event_spacing_s must be positive")

    @property
    def configuration_reference(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
