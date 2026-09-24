"""Immutable Device Profile v1 values; no trace or validity behavior."""

from __future__ import annotations

import math
from dataclasses import dataclass


LATENT_COMPONENTS = ("T", "A", "R", "N", "L")
TIMING_RELATION_FAMILIES = ("AUTH", "TRANSITION", "ACCESS", "RENEWAL")
LIFECYCLE_DEPTHS = ("AUTH", "SESSION", "ACCESS")


@dataclass(frozen=True)
class RelativeTimingTendency:
    """Unitless relative center/dispersion modifiers, not durations or limits."""

    center_modifier: float
    dispersion_modifier: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value > 0 for value in (
            self.center_modifier, self.dispersion_modifier,
        )):
            raise ValueError("relative timing modifiers must be finite and positive")


@dataclass(frozen=True)
class TimingTendencies:
    """T: exactly four persistent relation-family timing tendencies."""

    auth: RelativeTimingTendency
    transition: RelativeTimingTendency
    access: RelativeTimingTendency
    renewal: RelativeTimingTendency

    def __post_init__(self) -> None:
        if not all(isinstance(value, RelativeTimingTendency) for value in (
            self.auth, self.transition, self.access, self.renewal,
        )):
            raise ValueError("each timing family requires a relative tendency")


@dataclass(frozen=True)
class LifecycleWeights:
    """L: weights for valid generation depths, not security states."""

    auth: float
    session: float
    access: float

    def __post_init__(self) -> None:
        values = (self.auth, self.session, self.access)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("lifecycle weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("lifecycle weights must sum to one")


@dataclass(frozen=True)
class DeviceProfile:
    """Exactly T/A/R/N/L plus privileged identity/version metadata.

    The value is independent of semantic state and GT construction role. All
    five components can causally influence the opt-in ordinary
    profile-conditioned generation path, subject to semantic applicability.
    Semantic contexts do not own this value.
    """

    behavioral_profile_id: str
    contract_version: str
    timing: TimingTendencies  # T
    activity: float          # A
    recovery: float          # R
    renewal: float           # N
    lifecycle: LifecycleWeights  # L

    def __post_init__(self) -> None:
        if not self.behavioral_profile_id or not self.contract_version:
            raise ValueError("profile ID and contract version are required")
        if not isinstance(self.timing, TimingTendencies) or not isinstance(self.lifecycle, LifecycleWeights):
            raise ValueError("profile timing and lifecycle require v1 value types")
        for name in ("activity", "recovery", "renewal"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a normalized tendency in [0, 1]")
