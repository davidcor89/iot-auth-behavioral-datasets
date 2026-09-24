"""Abstract family-specific transformation boundary; no variants live here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from simulator.role_integration.applicability import ApplicabilityResult
from simulator.role_integration.models import ValidBaseTrace


ROLE_TRANSFORMATION_INTERFACE_VERSION = "4d3.1-family-transformation-interface-v1"


class TransformationFamily(str, Enum):
    TIMING = "timing"
    ACCESS_ENFORCEMENT = "access_enforcement"
    NONCE = "nonce"
    TOKEN_CURRENTNESS_REPLAY = "token_currentness_replay"
    IDENTITY = "identity"
    BINDING = "binding"


class TransformationStatus(str, Enum):
    APPLIED = "APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"
    CONSTRUCTION_ERROR = "CONSTRUCTION_ERROR"


@dataclass(frozen=True)
class TimingTransformationEvidence:
    """Future local timing-adapter evidence; construction metadata, never GT."""

    original_observed_time: float
    transformed_observed_time: float
    semantic_time: float
    targeted_relation: str
    transformation_randomness_reference: str


@dataclass(frozen=True)
class TransformationOutcome:
    """Future construction outcome; status carries no GT interpretation."""

    interface_version: str
    family: TransformationFamily
    transformation_id: str
    base_trace_reference: str
    status: TransformationStatus
    derived_event_records_json: tuple[str, ...]
    transformation_provenance_json: str
    reason: str

    def __post_init__(self) -> None:
        if self.interface_version != ROLE_TRANSFORMATION_INTERFACE_VERSION:
            raise ValueError("unsupported role-transformation interface version")


class RoleTransformation(Protocol):
    """Adapter protocol; implementations remain family-specific and future."""

    family: TransformationFamily
    transformation_id: str

    def applicability(
        self, base: ValidBaseTrace,
    ) -> tuple[ApplicabilityResult, ...]: ...

    def transform(
        self, base: ValidBaseTrace, *, randomness_reference: str,
    ) -> TransformationOutcome: ...
