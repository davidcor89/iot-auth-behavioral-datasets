"""Versioned controls for role-independent valid-base opportunity pools."""

from __future__ import annotations

from dataclasses import asdict, dataclass


OPPORTUNITY_POOL_CONTRACT_VERSION = "4d3.1-base-opportunity-pool-v1"
OPPORTUNITY_POOL_CONFIGURATION_VERSION = "4d3.1-opportunity-pool-config-v1"
OPPORTUNITY_RNG_DERIVATION_VERSION = "4d3.1-opportunity-sha256-v1"
ROLE_RNG_DERIVATION_VERSION = "4d3.1-future-role-sha256-v1"
TRANSFORMATION_RNG_DERIVATION_VERSION = "4d3.1-future-transformation-sha256-v1"


@dataclass(frozen=True)
class OpportunityPoolConfiguration:
    """A non-final controlled campaign specification.

    The role and transformation seeds establish isolated future namespaces but
    are deliberately excluded from base construction and pool identity.
    """

    opportunities_per_device: int = 2
    pool_seed: int = 4301
    planning_root_seed: int = 4302
    role_assignment_seed: int = 4303
    transformation_seed: int = 4304
    semantic_time_origin: float = 1_700_000_000.0
    trace_start_spacing_s: float = 1_000.0
    configuration_version: str = OPPORTUNITY_POOL_CONFIGURATION_VERSION
    contract_version: str = OPPORTUNITY_POOL_CONTRACT_VERSION
    rng_derivation_version: str = OPPORTUNITY_RNG_DERIVATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.opportunities_per_device < 1:
            raise ValueError("opportunities_per_device must be positive")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            self.pool_seed, self.planning_root_seed,
            self.role_assignment_seed, self.transformation_seed,
        )):
            raise ValueError("opportunity-pool seeds must be integers")
        if self.trace_start_spacing_s <= 0:
            raise ValueError("trace_start_spacing_s must be positive")
        if self.configuration_version != OPPORTUNITY_POOL_CONFIGURATION_VERSION:
            raise ValueError("unsupported opportunity-pool configuration version")
        if self.contract_version != OPPORTUNITY_POOL_CONTRACT_VERSION:
            raise ValueError("unsupported opportunity-pool contract version")
        if self.rng_derivation_version != OPPORTUNITY_RNG_DERIVATION_VERSION:
            raise ValueError("unsupported opportunity RNG derivation version")

    def base_identity_record(self) -> dict[str, object]:
        """Configuration affecting population-to-base construction only."""
        return {
            key: value for key, value in asdict(self).items()
            if key not in {"role_assignment_seed", "transformation_seed"}
        }
