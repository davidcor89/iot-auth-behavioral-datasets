"""Versioned controls for 4D.3.5 token-currentness integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


TOKEN_ROLE_TRANSFORMATION_CONFIGURATION_VERSION = "4d3.5-token-config-v1"


@dataclass(frozen=True)
class TokenRoleTransformationConfiguration:
    """Construction identity only; token adapters consume no randomness."""

    supported_scope: str = "within_continuing_protected_session"
    configuration_version: str = TOKEN_ROLE_TRANSFORMATION_CONFIGURATION_VERSION
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if self.configuration_version != TOKEN_ROLE_TRANSFORMATION_CONFIGURATION_VERSION:
            raise ValueError("unsupported token-role transformation configuration version")
        if self.supported_scope != "within_continuing_protected_session":
            raise ValueError("4D.3.5 supports only the frozen continuing-session scope")

    @property
    def configuration_reference(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
