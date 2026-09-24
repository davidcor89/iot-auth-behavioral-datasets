"""Versioned controls for the fixed-cohort scientific Device Profile mode.

These controls describe synthetic coverage, not an empirical device density.
They are deliberately separate from both the legacy global configuration and
the append-stable DP.1 infrastructure sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


SCIENTIFIC_POPULATION_CONTRACT_VERSION = "4d2-scientific-population-contract-v1"
SCIENTIFIC_POPULATION_CONFIGURATION_VERSION = "4d2-scientific-population-config-v1"
SCIENTIFIC_POPULATION_SAMPLER_VERSION = "4d2-balanced-scientific-population-v1"
SCIENTIFIC_POPULATION_RNG_VERSION = "4d2-component-sha256-v1"
SCALAR_COVERAGE_POLICY_VERSION = "4d2-balanced-randomized-strata-v1"
LIFECYCLE_SIMPLEX_PARTITION_VERSION = "4d2-symmetric-triangular-simplex-v1"
LIFECYCLE_WITHIN_CELL_POLICY_VERSION = "4d2-open-uniform-triangle-v1"
SCIENTIFIC_ENDPOINT_POLICY_VERSION = "4d2-strict-interior-v1"
SCIENTIFIC_PAIRING_POLICY_VERSION = "4d2-independent-dimension-permutations-v1"
SCIENTIFIC_CALIBRATION_STATUS = "synthetic_controlled_uncalibrated"


@dataclass(frozen=True)
class ScientificScalarSpace:
    """One bounded scalar marginal and its balanced coverage resolution."""

    bounds: tuple[float, float]
    strata_count: int = 4

    def __post_init__(self) -> None:
        if (
            len(self.bounds) != 2
            or not all(math.isfinite(value) for value in self.bounds)
            or self.bounds[1] <= self.bounds[0]
        ):
            raise ValueError("scientific scalar bounds must be finite and increasing")
        if (
            not isinstance(self.strata_count, int)
            or isinstance(self.strata_count, bool)
            or self.strata_count < 2
        ):
            raise ValueError("scientific scalar strata_count must be an integer >= 2")


def _timing_space() -> ScientificScalarSpace:
    return ScientificScalarSpace((0.75, 1.25), 4)


@dataclass(frozen=True)
class ScientificTimingSpace:
    """Independent scientific coverage controls for all eight T coordinates."""

    auth_center: ScientificScalarSpace = field(default_factory=_timing_space)
    auth_dispersion: ScientificScalarSpace = field(default_factory=_timing_space)
    transition_center: ScientificScalarSpace = field(default_factory=_timing_space)
    transition_dispersion: ScientificScalarSpace = field(default_factory=_timing_space)
    access_center: ScientificScalarSpace = field(default_factory=_timing_space)
    access_dispersion: ScientificScalarSpace = field(default_factory=_timing_space)
    renewal_center: ScientificScalarSpace = field(default_factory=_timing_space)
    renewal_dispersion: ScientificScalarSpace = field(default_factory=_timing_space)


@dataclass(frozen=True)
class ScientificLifecycleSpace:
    """Symmetric triangular partition of the two-dimensional lifecycle simplex."""

    subdivision: int = 4
    partition_version: str = LIFECYCLE_SIMPLEX_PARTITION_VERSION
    within_cell_policy: str = LIFECYCLE_WITHIN_CELL_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subdivision, int)
            or isinstance(self.subdivision, bool)
            or self.subdivision < 2
        ):
            raise ValueError("lifecycle simplex subdivision must be an integer >= 2")
        if self.partition_version != LIFECYCLE_SIMPLEX_PARTITION_VERSION:
            raise ValueError("unsupported lifecycle simplex partition")
        if self.within_cell_policy != LIFECYCLE_WITHIN_CELL_POLICY_VERSION:
            raise ValueError("unsupported lifecycle within-cell policy")

    @property
    def cell_count(self) -> int:
        return self.subdivision * self.subdivision


def _unit_space() -> ScientificScalarSpace:
    return ScientificScalarSpace((0.0, 1.0), 4)


@dataclass(frozen=True)
class ScientificPopulationConfiguration:
    """One immutable controlled-synthetic cohort specification."""

    device_count: int
    population_seed: int = 4101
    profile_contract_version: str = "device-profile-v1"
    population_contract_version: str = SCIENTIFIC_POPULATION_CONTRACT_VERSION
    configuration_version: str = SCIENTIFIC_POPULATION_CONFIGURATION_VERSION
    sampler_version: str = SCIENTIFIC_POPULATION_SAMPLER_VERSION
    rng_derivation_version: str = SCIENTIFIC_POPULATION_RNG_VERSION
    population_namespace: str = "4d2-scientific-population"
    scalar_coverage_policy: str = SCALAR_COVERAGE_POLICY_VERSION
    endpoint_policy: str = SCIENTIFIC_ENDPOINT_POLICY_VERSION
    cross_dimension_pairing_policy: str = SCIENTIFIC_PAIRING_POLICY_VERSION
    calibration_status: str = SCIENTIFIC_CALIBRATION_STATUS
    timing: ScientificTimingSpace = field(default_factory=ScientificTimingSpace)
    activity: ScientificScalarSpace = field(default_factory=_unit_space)
    recovery: ScientificScalarSpace = field(default_factory=_unit_space)
    renewal: ScientificScalarSpace = field(default_factory=_unit_space)
    lifecycle: ScientificLifecycleSpace = field(default_factory=ScientificLifecycleSpace)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.device_count, int)
            or isinstance(self.device_count, bool)
            or self.device_count < 1
        ):
            raise ValueError("scientific population device_count must be positive")
        if not isinstance(self.population_seed, int) or isinstance(self.population_seed, bool):
            raise ValueError("scientific population_seed must be an integer")
        expected = {
            "population_contract_version": SCIENTIFIC_POPULATION_CONTRACT_VERSION,
            "configuration_version": SCIENTIFIC_POPULATION_CONFIGURATION_VERSION,
            "sampler_version": SCIENTIFIC_POPULATION_SAMPLER_VERSION,
            "rng_derivation_version": SCIENTIFIC_POPULATION_RNG_VERSION,
            "scalar_coverage_policy": SCALAR_COVERAGE_POLICY_VERSION,
            "endpoint_policy": SCIENTIFIC_ENDPOINT_POLICY_VERSION,
            "cross_dimension_pairing_policy": SCIENTIFIC_PAIRING_POLICY_VERSION,
            "calibration_status": SCIENTIFIC_CALIBRATION_STATUS,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"unsupported scientific population {name}")
        if not self.profile_contract_version or not self.population_namespace:
            raise ValueError("scientific population identity fields are required")
        scalar_spaces = (
            *self.timing.__dict__.values(), self.activity, self.recovery, self.renewal,
        )
        if any(self.device_count < space.strata_count for space in scalar_spaces):
            raise ValueError("device_count must occupy every configured scalar stratum")
        if self.device_count < self.lifecycle.cell_count:
            raise ValueError("device_count must occupy every lifecycle simplex cell")
