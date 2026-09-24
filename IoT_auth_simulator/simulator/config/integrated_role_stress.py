"""Versioned, non-dataset controls for the 4D.3.7 integrated stress audit."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.config.opportunity_pool import OpportunityPoolConfiguration
from simulator.config.scientific_population import ScientificPopulationConfiguration


INTEGRATED_STRESS_CONFIGURATION_VERSION = "4d3.7-integrated-stress-config-v1"
INTEGRATED_STRESS_PRIMARY_SEED = 4701
INTEGRATED_STRESS_ALTERNATIVE_SEED = 4707


@dataclass(frozen=True)
class IntegratedRoleStressConfiguration:
    """One predetermined controlled-synthetic characterization campaign.

    Sixty-four devices with four opportunities each is a deliberate engineering
    compromise: 256 bases substantially exceed the 48-base development pool,
    occupy all scientific profile cells/strata, and keep the all-variant audit
    practical in the repository test environment.  It is not an ICDS size.
    """

    device_count: int = 64
    opportunities_per_device: int = 4
    population_seed: int = INTEGRATED_STRESS_PRIMARY_SEED
    device_construction_seed: int = 4700
    pool_seed: int = 4702
    planning_root_seed: int = 4703
    role_assignment_seed: int = 4704
    transformation_seed: int = 4705
    semantic_time_origin: float = 1_710_000_000.0
    trace_start_spacing_s: float = 1_000.0
    configuration_version: str = INTEGRATED_STRESS_CONFIGURATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.device_count < 16:
            raise ValueError("scientific stress population must occupy all lifecycle cells")
        if self.opportunities_per_device < 1:
            raise ValueError("stress opportunities_per_device must be positive")
        seeds = (
            self.population_seed, self.device_construction_seed, self.pool_seed,
            self.planning_root_seed, self.role_assignment_seed,
            self.transformation_seed,
        )
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("stress seeds must be integers")
        if self.configuration_version != INTEGRATED_STRESS_CONFIGURATION_VERSION:
            raise ValueError("unsupported integrated-stress configuration version")

    def population_configuration(self) -> ScientificPopulationConfiguration:
        return ScientificPopulationConfiguration(
            device_count=self.device_count, population_seed=self.population_seed,
        )

    def opportunity_configuration(self) -> OpportunityPoolConfiguration:
        return OpportunityPoolConfiguration(
            opportunities_per_device=self.opportunities_per_device,
            pool_seed=self.pool_seed,
            planning_root_seed=self.planning_root_seed,
            role_assignment_seed=self.role_assignment_seed,
            transformation_seed=self.transformation_seed,
            semantic_time_origin=self.semantic_time_origin,
            trace_start_spacing_s=self.trace_start_spacing_s,
        )
