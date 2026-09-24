"""Controlled-synthetic configuration for 4C.4 activity realization.

This run/planning configuration is deliberately separate from both the latent
``DeviceProfile`` and the DP.1 population configuration.  Its defaults define
the supported implementation repertoire, not an empirically calibrated
activity distribution.
"""

from __future__ import annotations

from dataclasses import dataclass


ACTIVITY_MAPPING_VERSION = "4c4-binomial-3-inverse-cdf-v1"


@dataclass(frozen=True)
class ActivityPlanningConfiguration:
    """Versioned bounds for the restricted clean-operation repertoire."""

    configuration_version: str = "4c4-controlled-activity-planning-v1"
    mapping_version: str = ACTIVITY_MAPPING_VERSION
    supported_minimum: int = 1
    supported_maximum: int = 4
    calibration_status: str = "synthetic_controlled_uncalibrated"

    def __post_init__(self) -> None:
        if not self.configuration_version or not self.mapping_version:
            raise ValueError("activity configuration and mapping versions are required")
        if self.mapping_version != ACTIVITY_MAPPING_VERSION:
            raise ValueError("unsupported activity realization mapping")
        if self.supported_minimum != 1 or self.supported_maximum != 4:
            raise ValueError("4C.4 supports exactly 1..4 clean protected operations")
        if self.calibration_status != "synthetic_controlled_uncalibrated":
            raise ValueError("4C.4 activity realization is not empirically calibrated")
