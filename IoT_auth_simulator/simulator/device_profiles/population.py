"""Run-scoped sibling registry for deterministic Device Profiles.

Each device has an explicit stable population key distinct from its UUID.
Sampling uses only local ``random.Random`` instances. Equal-width marginal
strata and independent within-stratum draws are initial controlled-synthetic
infrastructure defaults, not a final population-composition policy or
empirical distribution. Downstream causal use is owned explicitly by the
device-aware planning layer rather than by this registry.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.device_profiles.profile import (
    DeviceProfile,
    LifecycleWeights,
    RelativeTimingTendency,
    TimingTendencies,
)


SEED_DERIVATION_VERSION = "dp1-sha256-config-key-v1"


def population_device_key_from_index(index: int) -> str:
    """Explicit append-stable runner key; not a UUID, IP, or device type."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("population device index must be a nonnegative integer")
    return f"population-device-{index:08d}"


def _digest(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _configuration_reference(config: PopulationConfiguration) -> str:
    return "sha256:" + _digest(asdict(config)).hex()


def _device_seed(config: PopulationConfiguration, key: str) -> tuple[int, str]:
    digest = _digest({
        "derivation_version": SEED_DERIVATION_VERSION,
        "population_namespace": config.population_namespace,
        "configuration_reference": _configuration_reference(config),
        "population_seed": config.population_seed,
        "population_device_key": key,
    })
    return int.from_bytes(digest, "big"), "sha256:" + digest.hex()


def _profile_id(config: PopulationConfiguration, key: str) -> str:
    digest = _digest({
        "kind": "behavioral_profile_id",
        "configuration_reference": _configuration_reference(config),
        "population_device_key": key,
    })
    return "behavioral-profile-v1-" + digest.hex()[:24]


def _stratified_value(rng: random.Random, strata: int, bounds: tuple[float, float]) -> float:
    """Choose an independent marginal stratum, then draw within its interval."""
    stratum = rng.randrange(strata)
    position = (stratum + rng.random()) / strata
    return bounds[0] + (bounds[1] - bounds[0]) * position


def _sample_profile(config: PopulationConfiguration, key: str) -> tuple[DeviceProfile, str]:
    seed, seed_reference = _device_seed(config, key)
    rng = random.Random(seed)  # Does not seed or consume the legacy global stream.

    def relative_timing() -> RelativeTimingTendency:
        return RelativeTimingTendency(
            center_modifier=_stratified_value(
                rng, config.strata_count, config.timing_center_modifier_bounds,
            ),
            dispersion_modifier=_stratified_value(
                rng, config.strata_count, config.timing_dispersion_modifier_bounds,
            ),
        )

    timing = TimingTendencies(
        auth=relative_timing(), transition=relative_timing(),
        access=relative_timing(), renewal=relative_timing(),
    )
    activity = _stratified_value(rng, config.strata_count, (0.0, 1.0))
    recovery = _stratified_value(rng, config.strata_count, (0.0, 1.0))
    renewal = _stratified_value(rng, config.strata_count, (0.0, 1.0))
    # Independently sampled positive depth masses avoid rigid named classes.
    masses = [max(_stratified_value(rng, config.strata_count, (0.0, 1.0)), 1e-12)
              for _ in range(3)]
    total = sum(masses)
    lifecycle = LifecycleWeights(*(mass / total for mass in masses))
    return DeviceProfile(
        behavioral_profile_id=_profile_id(config, key),
        contract_version=config.contract_version,
        timing=timing, activity=activity, recovery=recovery,
        renewal=renewal, lifecycle=lifecycle,
    ), seed_reference


@dataclass(frozen=True)
class ProfileAssignment:
    """Privileged internal device/profile association, not an export record."""

    device_id: str
    population_device_key: str
    profile_seed_reference: str
    profile: DeviceProfile


@dataclass(frozen=True)
class DeviceProfilePopulation:
    """One immutable assignment per runner-created population device."""

    configuration: PopulationConfiguration
    assignments: tuple[ProfileAssignment, ...]
    _by_device_id: Mapping[str, ProfileAssignment] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        index = {assignment.device_id: assignment for assignment in self.assignments}
        if len(index) != len(self.assignments):
            raise ValueError("population device IDs must be unique")
        if len({item.population_device_key for item in self.assignments}) != len(self.assignments):
            raise ValueError("population device keys must be unique")
        object.__setattr__(self, "_by_device_id", MappingProxyType(index))

    def profile_for_device(self, device_id: str) -> DeviceProfile:
        return self._by_device_id[device_id].profile

    def assignment_for_device(self, device_id: str) -> ProfileAssignment:
        return self._by_device_id[device_id]

    @property
    def behavioral_profiles(self) -> Mapping[str, DeviceProfile]:
        return MappingProxyType({key: item.profile for key, item in self._by_device_id.items()})

    def internal_reconstruction_record(self) -> dict[str, object]:
        """Privileged in-memory audit data; DP.1 does not serialize it."""
        return {
            "configuration": asdict(self.configuration),
            "configuration_reference": _configuration_reference(self.configuration),
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "assignments": [
                {
                    "device_id": item.device_id,
                    "population_device_key": item.population_device_key,
                    "profile_seed_reference": item.profile_seed_reference,
                    "behavioral_profile_id": item.profile.behavioral_profile_id,
                    "latent_profile": {
                        "timing": asdict(item.profile.timing),
                        "activity": item.profile.activity,
                        "recovery": item.profile.recovery,
                        "renewal": item.profile.renewal,
                        "lifecycle": asdict(item.profile.lifecycle),
                    },
                }
                for item in self.assignments
            ],
        }


def build_profile_population(
    device_bindings: Iterable[tuple[str, str]],
    configuration: PopulationConfiguration,
) -> DeviceProfilePopulation:
    """Sample once per stable key without consulting device type or GT role."""
    assignments = []
    for device_id, key in device_bindings:
        if not device_id or not key:
            raise ValueError("device ID and stable population key are required")
        profile, seed_reference = _sample_profile(configuration, key)
        assignments.append(ProfileAssignment(device_id, key, seed_reference, profile))
    return DeviceProfilePopulation(configuration, tuple(assignments))
