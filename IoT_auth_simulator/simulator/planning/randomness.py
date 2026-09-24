"""Isolated deterministic planning-stream derivation for trace choices."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import IntEnum


PLANNING_RNG_DERIVATION_VERSION = "4c-inf-planning-sha256-v1"
PLANNING_RNG_NAMESPACE = "device-aware-trace-planning"
PLANNING_DECISION_SLOT_ORDER_VERSION = "4c5-fixed-planning-slots-v4"


class PlanningDecisionSlot(IntEnum):
    """Frozen positions in the isolated per-trace planning stream."""

    LIFECYCLE = 1
    RECOVERY = 2
    RENEWAL = 3
    ACTIVITY = 4
    TIMING = 5


@dataclass(frozen=True)
class PlanningRandomness:
    """Stable stream identity; consumers explicitly own any planning draws."""

    seed: int
    reference: str
    derivation_version: str = PLANNING_RNG_DERIVATION_VERSION
    namespace: str = PLANNING_RNG_NAMESPACE

    def new_rng(self) -> random.Random:
        """Instantiate an isolated RNG without touching module-level random."""
        return random.Random(self.seed)

    def variate_at(self, slot: PlanningDecisionSlot | int) -> float:
        """Return one fixed positional variate without conditional stream shifts."""
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
            raise ValueError("planning decision slot must be a positive integer")
        rng = self.new_rng()
        draw = 0.0
        for _ in range(slot):
            draw = rng.random()
        return draw

    @classmethod
    def from_reference(cls, reference: str) -> "PlanningRandomness":
        """Reconstruct the stream seed from its privileged SHA-256 reference."""
        prefix = "sha256:"
        digest_hex = reference[len(prefix):] if reference.startswith(prefix) else ""
        if len(digest_hex) != 64:
            raise ValueError("planning RNG reference must contain one SHA-256 digest")
        try:
            seed = int(digest_hex, 16)
        except ValueError as error:
            raise ValueError("planning RNG reference is not hexadecimal") from error
        return cls(seed=seed, reference=reference)


def derive_planning_randomness(
    *,
    planning_root_seed: int,
    population_device_key: str,
    trace_construction_key: str,
) -> PlanningRandomness:
    """Derive an append-stable per-device/per-request stream using SHA-256."""
    if not isinstance(planning_root_seed, int) or isinstance(planning_root_seed, bool):
        raise ValueError("planning root seed must be an integer")
    if not population_device_key or not trace_construction_key:
        raise ValueError("stable population and trace construction keys are required")
    payload = {
        "derivation_version": PLANNING_RNG_DERIVATION_VERSION,
        "planning_namespace": PLANNING_RNG_NAMESPACE,
        "planning_root_seed": planning_root_seed,
        "population_device_key": population_device_key,
        "trace_construction_key": trace_construction_key,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return PlanningRandomness(
        seed=int.from_bytes(digest, "big"),
        reference="sha256:" + digest.hex(),
    )
