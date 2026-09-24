"""Version and run provenance for the synchronized v1.1 release."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from simulator.semantic_time import temporal_contract_metadata
from simulator.anomaly_contract import synchronized_capability_report
from simulator.invariant_registry import invariant_registry_report
from simulator.hard_negative import hard_negative_capability_report


BEHAVIOR_MODEL_VERSION = "behavioral-model-v1.1"
# Final C2 release identifiers. The historical feature schema is intentionally
# unchanged because C2.6 does not alter detector observations or features.
GENERATOR_SOFTWARE_VERSION = "sync-v1.1.0"
EVENT_SCHEMA_VERSION = "sync-v1.1.0-event-schema"
FEATURE_SCHEMA_VERSION = "c1.6-development-derived-schema"
CONFIGURATION_VERSION = "historical-defaults-27135bf"


def _json_compatible(value: Any) -> Any:
    """Convert configuration values into deterministic JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def configuration_snapshot(config: Any) -> dict[str, Any]:
    """Return a serializable configuration snapshot without mutating ``config``."""
    snapshot = _json_compatible(config)
    if not isinstance(snapshot, dict):
        raise TypeError("generation configuration must serialize to a mapping")
    return snapshot


def configuration_snapshot_reference(snapshot: Mapping[str, Any]) -> str:
    """Return a stable content reference for a serializable config snapshot."""
    encoded = json.dumps(
        _json_compatible(snapshot), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def discover_generator_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Resolve the generator commit from the environment or local Git checkout.

    ``GENERATOR_GIT_COMMIT`` supports packaged/CI execution where ``.git`` is
    unavailable.  Failure to inspect Git is represented explicitly as ``None``.
    """
    supplied = os.environ.get("GENERATOR_GIT_COMMIT")
    if supplied:
        return supplied

    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def discover_generator_git_dirty(repo_root: Optional[Path] = None) -> Optional[bool]:
    """Report tracked-file dirtiness; preserved untracked audit docs are excluded."""
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


@dataclass(frozen=True)
class GenerationProvenance:
    """Traceability metadata for one generation run.

    Synchronized outputs attach this structure to their provenance view. It
    remains absent from historical compatibility schemas.
    """

    behavior_model_version: str
    generator_software_version: str
    generator_git_commit: Optional[str]
    generator_git_dirty: Optional[bool]
    generator_git_state_scope: str
    event_schema_version: str
    feature_schema_version: str
    event_ground_truth_schema_version: str
    trace_ground_truth_schema_version: str
    validation_schema_version: str
    field_manifest_version: str
    dataset_interface_version: str
    dataset_manifest_version: str
    dataset_version: str
    invariant_registry_version: str
    run_id: str
    configuration_version: str
    configuration_snapshot_reference: str
    configuration_snapshot: dict[str, Any]
    temporal_contract: dict[str, Any]
    anomaly_capability_contract: dict[str, Any]
    invariant_registry: dict[str, Any]
    randomness_contract: dict[str, Any]
    hard_negative_capability_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_generation_provenance(
    config: Any,
    *,
    run_id: Optional[str] = None,
    generator_git_commit: Optional[str] = None,
    generator_git_dirty: Optional[bool] = None,
) -> GenerationProvenance:
    """Build a run record without starting or changing simulation execution."""
    # Local imports avoid a module-level provenance/dataset-contract cycle.
    from simulator.data.dataset_contract import (
        DATASET_INTERFACE_VERSION,
        DATASET_MANIFEST_VERSION,
        DATASET_VERSION,
        FIELD_MANIFEST_VERSION,
    )
    from simulator.validation import (
        EVENT_GROUND_TRUTH_SCHEMA_VERSION,
        TRACE_GROUND_TRUTH_SCHEMA_VERSION,
        VALIDATION_SCHEMA_VERSION,
    )

    snapshot = configuration_snapshot(config)
    return GenerationProvenance(
        behavior_model_version=BEHAVIOR_MODEL_VERSION,
        generator_software_version=GENERATOR_SOFTWARE_VERSION,
        generator_git_commit=(
            generator_git_commit
            if generator_git_commit is not None
            else discover_generator_git_commit()
        ),
        generator_git_dirty=(
            generator_git_dirty
            if generator_git_dirty is not None
            else discover_generator_git_dirty()
        ),
        generator_git_state_scope="tracked_files_only_untracked_files_excluded",
        event_schema_version=EVENT_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        event_ground_truth_schema_version=EVENT_GROUND_TRUTH_SCHEMA_VERSION,
        trace_ground_truth_schema_version=TRACE_GROUND_TRUTH_SCHEMA_VERSION,
        validation_schema_version=VALIDATION_SCHEMA_VERSION,
        field_manifest_version=FIELD_MANIFEST_VERSION,
        dataset_interface_version=DATASET_INTERFACE_VERSION,
        dataset_manifest_version=DATASET_MANIFEST_VERSION,
        dataset_version=DATASET_VERSION,
        invariant_registry_version=invariant_registry_report()["registry_version"],
        run_id=run_id or str(uuid.uuid4()),
        configuration_version=CONFIGURATION_VERSION,
        configuration_snapshot_reference=configuration_snapshot_reference(snapshot),
        configuration_snapshot=snapshot,
        temporal_contract=temporal_contract_metadata(config),
        anomaly_capability_contract=synchronized_capability_report(),
        invariant_registry=invariant_registry_report(),
        randomness_contract={
            "master_seed_path": "configuration_snapshot.simulation.random_seed",
            "stream_model": (
                "legacy_global_event_stream_with_optional_isolated_trace_planning_"
                "and_relation_timing_derivation"
            ),
            "trace_planning_stream": {
                "namespace": "device-aware-trace-planning",
                "derivation_version": "4c-inf-planning-sha256-v1",
                "compatibility_mode_draws": 0,
                "profile_conditioned_lifecycle_draws": 1,
                "profile_conditioned_lifecycle_recovery_draws": 2,
                "profile_conditioned_lifecycle_recovery_renewal_draws": 3,
                "profile_conditioned_lifecycle_recovery_renewal_activity_draws": 4,
                "timing_root_derivation": (
                    "slot_5_identity_with_deterministic_per_relation_hashes"
                ),
                "decision_slot_order_version": "4c5-fixed-planning-slots-v4",
                "decision_slots": {
                    "lifecycle": 1,
                    "recovery": 2,
                    "renewal": 3,
                    "activity": 4,
                    "timing_root": 5,
                },
                "profile_conditioned_mode_is_opt_in": True,
                "timing_realization": {
                    "profile_conditioned_mode_is_opt_in": True,
                    "decision_slot": 5,
                    "stream_form": "deterministic_per_relation_hashes",
                    "latent_timing_not_repeated": True,
                },
            },
            "semantic_reproducibility": "supported_for_configuration_and_seed",
            "byte_identical_reproducibility": False,
            "known_limitation": (
                "UUIDs, generated credentials/nonces/PSKs, optional wall-clock "
                "anchors, and serialization details may differ byte-for-byte."
            ),
        },
        hard_negative_capability_contract=hard_negative_capability_report(),
    )
