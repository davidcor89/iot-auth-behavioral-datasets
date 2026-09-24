"""Versioned, controlled-synthetic inputs for dormant DP.1 profile sampling.

This configuration is deliberately separate from the legacy global ``cfg``.
Its initial numerical bounds exercise infrastructure; they are neither an
empirical calibration nor final dataset-population parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationConfiguration:
    """Population construction controls, never behavioral validity rules."""

    contract_version: str = "device-profile-v1"
    configuration_version: str = "dp1-controlled-synthetic-v1"
    population_namespace: str = "dp1-infrastructure"
    population_seed: int = 42
    strata_count: int = 4
    timing_center_modifier_bounds: tuple[float, float] = (0.75, 1.25)
    timing_dispersion_modifier_bounds: tuple[float, float] = (0.75, 1.25)
    calibration_status: str = "synthetic_controlled_uncalibrated"

    def __post_init__(self) -> None:
        if not all((self.contract_version, self.configuration_version, self.population_namespace)):
            raise ValueError("profile contract/configuration/namespace must be identified")
        if not isinstance(self.population_seed, int) or isinstance(self.population_seed, bool):
            raise ValueError("population_seed must be an integer")
        if not isinstance(self.strata_count, int) or isinstance(self.strata_count, bool) or self.strata_count < 2:
            raise ValueError("strata_count must be an integer >= 2")
        for name, bounds in (
            ("timing_center_modifier_bounds", self.timing_center_modifier_bounds),
            ("timing_dispersion_modifier_bounds", self.timing_dispersion_modifier_bounds),
        ):
            if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds):
                raise ValueError(f"{name} must contain two finite values")
            if bounds[0] <= 0 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name} must be positive and increasing")
        if self.calibration_status != "synthetic_controlled_uncalibrated":
            raise ValueError("DP.1 does not support empirical calibration")
