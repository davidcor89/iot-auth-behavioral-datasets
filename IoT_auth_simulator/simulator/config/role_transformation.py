"""Versioned engineering controls for the first concrete role adapters.

These values define a controlled synthetic construction surface.  They are not
protocol validity limits, empirical calibration, or detector configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


ROLE_TRANSFORMATION_CONFIGURATION_VERSION = "4d3.2-direct-timing-config-v1"


@dataclass(frozen=True)
class DirectTimingTransformationConfiguration:
    """Local transformation controls kept outside ordinary trace generation."""

    # HN-v1 engineering campaigns already exercise 0.01 as an explicit
    # candidate-specific boundary; this remains configurable and noncanonical.
    epsilon_hn_s: float = 0.010
    hn_margin_lower_fraction: float = 0.20
    hn_margin_upper_fraction: float = 0.80
    anomaly_negative_margin_s: float = 0.001
    configuration_version: str = ROLE_TRANSFORMATION_CONFIGURATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.configuration_version != ROLE_TRANSFORMATION_CONFIGURATION_VERSION:
            raise ValueError("unsupported role-transformation configuration version")
        if self.epsilon_hn_s <= 0:
            raise ValueError("epsilon_hn_s must be positive")
        if not 0 < self.hn_margin_lower_fraction <= self.hn_margin_upper_fraction < 1:
            raise ValueError("HN margin fractions must satisfy 0 < lower <= upper < 1")
        if self.anomaly_negative_margin_s <= 0:
            raise ValueError("anomaly_negative_margin_s must be positive")

    @property
    def configuration_reference(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
