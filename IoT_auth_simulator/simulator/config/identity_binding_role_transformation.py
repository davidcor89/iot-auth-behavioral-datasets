"""Versioned controls for 4D.3.6 identity and token-binding integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


IDENTITY_BINDING_TRANSFORMATION_CONFIGURATION_VERSION = (
    "4d3.6-identity-binding-config-v1"
)


@dataclass(frozen=True)
class IdentityBindingTransformationConfiguration:
    """Construction identity only; the adapters consume no random draws."""

    supported_scope: str = "within_trace_represented_identity_and_token_binding"
    configuration_version: str = (
        IDENTITY_BINDING_TRANSFORMATION_CONFIGURATION_VERSION
    )
    calibration_status: str = "controlled_synthetic_non_final_uncalibrated"

    def __post_init__(self) -> None:
        if (
            self.configuration_version
            != IDENTITY_BINDING_TRANSFORMATION_CONFIGURATION_VERSION
        ):
            raise ValueError("unsupported identity/binding configuration version")
        if self.supported_scope != "within_trace_represented_identity_and_token_binding":
            raise ValueError("4D.3.6 supports only represented within-trace relations")

    @property
    def configuration_reference(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
