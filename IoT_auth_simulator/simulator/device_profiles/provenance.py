"""Dedicated, privileged DP.2 population/profile provenance.

This artifact is neither a detector observation, ground-truth record, nor
trace/event debug view. The normalized fingerprint deliberately excludes
emitted UUIDs and run identifiers; those remain useful only for joining a
particular run's traces to its profile records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.scientific_population import (
    SCIENTIFIC_POPULATION_RNG_VERSION,
    SCIENTIFIC_POPULATION_SAMPLER_VERSION,
    ScientificLifecycleSpace,
    ScientificPopulationConfiguration,
    ScientificScalarSpace,
    ScientificTimingSpace,
)
from simulator.device_profiles.population import (
    SEED_DERIVATION_VERSION,
    build_profile_population,
)
from simulator.device_profiles.profile import (
    DeviceProfile,
    LifecycleWeights,
    RelativeTimingTendency,
    TimingTendencies,
)
from simulator.device_profiles.scientific_population import (
    SCIENTIFIC_POPULATION_FINGERPRINT_VERSION,
    ScientificDeviceProfilePopulation,
    build_scientific_profile_population,
    scientific_configuration_record,
)

if TYPE_CHECKING:
    from simulator.runner import RunPopulation
    from simulator.data.synchronized_views import SequencePairs


PROFILE_PROVENANCE_SCHEMA_VERSION = "dp2-device-profile-provenance-v1"
SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION = "4d2-device-profile-provenance-v2"
SUPPORTED_PROFILE_PROVENANCE_SCHEMA_VERSIONS = frozenset({
    PROFILE_PROVENANCE_SCHEMA_VERSION,
    SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION,
})
POPULATION_FINGERPRINT_VERSION = "dp2-normalized-population-sha256-v1"
PROFILE_PROVENANCE_AUTHORITY = "privileged_population_generation_provenance"
PROFILE_SAMPLER_VERSION = "dp1-strata-with-replacement-v1"


def _configuration_record(configuration: PopulationConfiguration) -> dict[str, Any]:
    """Serialize only population controls, not the legacy generator cfg."""
    return {
        "profile_contract_version": configuration.contract_version,
        "population_configuration_version": configuration.configuration_version,
        "sampler_namespace": configuration.population_namespace,
        "sampler_version": PROFILE_SAMPLER_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "population_seed": configuration.population_seed,
        "calibration_status": configuration.calibration_status,
        "strata_count": configuration.strata_count,
        "timing_center_modifier_bounds": list(configuration.timing_center_modifier_bounds),
        "timing_dispersion_modifier_bounds": list(configuration.timing_dispersion_modifier_bounds),
    }


def _timing_record(value: RelativeTimingTendency) -> dict[str, float]:
    return {
        "center_modifier": value.center_modifier,
        "dispersion_modifier": value.dispersion_modifier,
    }


def _latent_record(profile: DeviceProfile) -> dict[str, Any]:
    """Exactly the scientific T/A/R/N/L components; no profile metadata."""
    return {
        "T": {
            "AUTH": _timing_record(profile.timing.auth),
            "TRANSITION": _timing_record(profile.timing.transition),
            "ACCESS": _timing_record(profile.timing.access),
            "RENEWAL": _timing_record(profile.timing.renewal),
        },
        "A": profile.activity,
        "R": profile.recovery,
        "N": profile.renewal,
        "L": {
            "AUTH": profile.lifecycle.auth,
            "SESSION": profile.lifecycle.session,
            "ACCESS": profile.lifecycle.access,
        },
    }


def _device_record(assignment: Any) -> dict[str, Any]:
    return {
        "device_id": assignment.device_id,
        "population_device_key": assignment.population_device_key,
        "behavioral_profile_id": assignment.profile.behavioral_profile_id,
        "profile_seed_reference": assignment.profile_seed_reference,
        "profile_contract_version": assignment.profile.contract_version,
        "latent_tendencies": _latent_record(assignment.profile),
    }


def _scientific_device_record(assignment: Any) -> dict[str, Any]:
    return {
        **_device_record(assignment),
        "scalar_strata": dict(assignment.scalar_strata),
        "lifecycle_cell_id": assignment.lifecycle_cell_id,
        "component_references": dict(assignment.component_references),
    }


def population_fingerprint(
    configuration_record: Mapping[str, Any],
    device_records: list[Mapping[str, Any]],
) -> str:
    """SHA-256 over version, full profile configuration and sorted stable assignments.

    Each assignment contributes its stable population key, profile ID,
    contract version and exact T/A/R/N/L floats. UUID device IDs, per-run IDs,
    seed references, physical attributes, credentials and row order are absent.
    The seed derivation version and whole configuration enter via the population
    configuration record. This is a content fingerprint, not a uniqueness proof.
    """
    assignments = [
        {
            "population_device_key": record["population_device_key"],
            "behavioral_profile_id": record["behavioral_profile_id"],
            "profile_contract_version": record["profile_contract_version"],
            "latent_tendencies": record["latent_tendencies"],
        }
        for record in device_records
    ]
    assignments.sort(key=lambda record: record["population_device_key"])
    normalized = {
        "fingerprint_version": POPULATION_FINGERPRINT_VERSION,
        "population_configuration": dict(configuration_record),
        "assignments": assignments,
    }
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def assert_trace_profile_association(
    sequences: "SequencePairs", profile_record: Mapping[str, Any],
) -> None:
    """Require every runner trace/event device ID to resolve to one profile."""
    devices = profile_record["devices"]
    ids = [record["device_id"] for record in devices]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate profile device IDs")
    associated = set(ids)
    for events, context in sequences:
        if context.device_id not in associated:
            raise ValueError(f"trace device has no profile record: {context.device_id}")
        if any(event.device_id != context.device_id for event in events):
            raise ValueError("trace event device ID disagrees with trace context")


def to_device_profile_provenance(
    run_population: "RunPopulation", *, run_id: str,
    sequences: "SequencePairs | None" = None,
) -> dict[str, Any]:
    """Build a separate two-level artifact from the runner-owned registries."""
    if not run_id:
        raise ValueError("run_id is required for profile artifact association")
    assignments = run_population.behavioral_profiles.assignments
    device_ids = {device.device_id for device in run_population.devices}
    assignment_ids = {item.device_id for item in assignments}
    semantic_ids = set(run_population.semantic_contexts)
    indexed_device_ids = set(run_population.devices_by_id)
    if not (len(device_ids) == len(assignments) == len(assignment_ids)
            and device_ids == assignment_ids == semantic_ids == indexed_device_ids):
        raise ValueError("runner device, semantic context and profile registries disagree")
    profiles = [item.profile.behavioral_profile_id for item in assignments]
    if len(profiles) != len(set(profiles)):
        raise ValueError("behavioral profile IDs must be unique")

    profiles = run_population.behavioral_profiles
    if isinstance(profiles, ScientificDeviceProfilePopulation):
        configuration_record = scientific_configuration_record(profiles.configuration)
        devices = sorted((_scientific_device_record(item) for item in assignments),
                         key=lambda record: record["population_device_key"])
        artifact = {
            "authority": PROFILE_PROVENANCE_AUTHORITY,
            "provenance_schema_version": SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION,
            "run_id": run_id,
            "population": {
                "configuration": configuration_record,
                "device_count": len(devices),
                "stable_population_device_keys": [
                    record["population_device_key"] for record in devices
                ],
                "sampler_version": SCIENTIFIC_POPULATION_SAMPLER_VERSION,
                "rng_derivation_version": SCIENTIFIC_POPULATION_RNG_VERSION,
                "fingerprint_version": SCIENTIFIC_POPULATION_FINGERPRINT_VERSION,
                "population_fingerprint": profiles.population_fingerprint,
                "coverage_diagnostics": dict(profiles.diagnostics),
            },
            "devices": devices,
        }
    else:
        configuration_record = _configuration_record(profiles.configuration)
        devices = sorted((_device_record(item) for item in assignments),
                         key=lambda record: record["population_device_key"])
        artifact = {
            "authority": PROFILE_PROVENANCE_AUTHORITY,
            "provenance_schema_version": PROFILE_PROVENANCE_SCHEMA_VERSION,
            "run_id": run_id,
            "population": {
                "configuration": configuration_record,
                "device_count": len(devices),
                "population_fingerprint": population_fingerprint(configuration_record, devices),
            },
            "devices": devices,
        }
    if sequences is not None:
        assert_trace_profile_association(sequences, artifact)
    return artifact


def _require_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from DP.2 schema")


def _reconstructed_configuration(record: Mapping[str, Any]) -> PopulationConfiguration:
    _require_fields(record, {
        "profile_contract_version", "population_configuration_version",
        "sampler_namespace", "sampler_version", "seed_derivation_version", "population_seed",
        "calibration_status", "strata_count", "timing_center_modifier_bounds",
        "timing_dispersion_modifier_bounds",
    }, "population configuration")
    if record["seed_derivation_version"] != SEED_DERIVATION_VERSION:
        raise ValueError("unsupported profile seed derivation version")
    if record["sampler_version"] != PROFILE_SAMPLER_VERSION:
        raise ValueError("unsupported profile sampler version")
    configuration = PopulationConfiguration(
        contract_version=record["profile_contract_version"],
        configuration_version=record["population_configuration_version"],
        population_namespace=record["sampler_namespace"],
        population_seed=record["population_seed"],
        calibration_status=record["calibration_status"],
        strata_count=record["strata_count"],
        timing_center_modifier_bounds=tuple(record["timing_center_modifier_bounds"]),
        timing_dispersion_modifier_bounds=tuple(record["timing_dispersion_modifier_bounds"]),
    )
    if _configuration_record(configuration) != dict(record):
        raise ValueError("population configuration does not round-trip")
    return configuration


def _reconstructed_profile(
    record: Mapping[str, Any], *, profile_id: str, contract_version: str,
) -> DeviceProfile:
    _require_fields(record, {"T", "A", "R", "N", "L"}, "latent profile")
    timing = record["T"]
    _require_fields(timing, {"AUTH", "TRANSITION", "ACCESS", "RENEWAL"}, "timing families")

    def relation(family: str) -> RelativeTimingTendency:
        value = timing[family]
        _require_fields(value, {"center_modifier", "dispersion_modifier"}, "timing tendency")
        return RelativeTimingTendency(value["center_modifier"], value["dispersion_modifier"])

    lifecycle = record["L"]
    _require_fields(lifecycle, {"AUTH", "SESSION", "ACCESS"}, "lifecycle depths")
    return DeviceProfile(
        behavioral_profile_id=profile_id,
        contract_version=contract_version,
        timing=TimingTendencies(
            auth=relation("AUTH"), transition=relation("TRANSITION"),
            access=relation("ACCESS"), renewal=relation("RENEWAL"),
        ),
        activity=record["A"], recovery=record["R"], renewal=record["N"],
        lifecycle=LifecycleWeights(
            lifecycle["AUTH"], lifecycle["SESSION"], lifecycle["ACCESS"],
        ),
    )


def _validate_legacy_device_profile_provenance(artifact: Mapping[str, Any]) -> None:
    """Validate the unchanged DP.2-v1/DP.1 replay path."""
    _require_fields(artifact, {
        "authority", "provenance_schema_version", "run_id", "population", "devices",
    }, "profile artifact")
    if artifact["authority"] != PROFILE_PROVENANCE_AUTHORITY:
        raise ValueError("incorrect profile provenance authority")
    if artifact["provenance_schema_version"] != PROFILE_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("incorrect legacy profile provenance schema")
    if not artifact["run_id"]:
        raise ValueError("profile artifact run_id is required")
    population = artifact["population"]
    _require_fields(population, {"configuration", "device_count", "population_fingerprint"},
                    "population provenance")
    configuration = _reconstructed_configuration(population["configuration"])
    devices = artifact["devices"]
    if not isinstance(devices, list) or population["device_count"] != len(devices):
        raise ValueError("population device count is inconsistent")
    ids, keys, profile_ids = set(), set(), set()
    bindings = []
    for record in devices:
        _require_fields(record, {
            "device_id", "population_device_key", "behavioral_profile_id",
            "profile_seed_reference", "profile_contract_version", "latent_tendencies",
        }, "device profile")
        if not all(record[name] for name in (
            "device_id", "population_device_key", "behavioral_profile_id",
            "profile_seed_reference", "profile_contract_version",
        )):
            raise ValueError("device profile identity fields are required")
        if record["profile_contract_version"] != configuration.contract_version:
            raise ValueError("device profile contract version mismatch")
        _reconstructed_profile(
            record["latent_tendencies"],
            profile_id=record["behavioral_profile_id"],
            contract_version=record["profile_contract_version"],
        )
        ids.add(record["device_id"])
        keys.add(record["population_device_key"])
        profile_ids.add(record["behavioral_profile_id"])
        bindings.append((record["device_id"], record["population_device_key"]))
    if not (len(ids) == len(keys) == len(profile_ids) == len(devices)):
        raise ValueError("duplicate device ID, population key or profile ID")
    expected = build_profile_population(bindings, configuration)
    by_key = {item.population_device_key: item for item in expected.assignments}
    for record in devices:
        item = by_key[record["population_device_key"]]
        if (record["behavioral_profile_id"] != item.profile.behavioral_profile_id
                or record["profile_seed_reference"] != item.profile_seed_reference
                or record["latent_tendencies"] != _latent_record(item.profile)):
            raise ValueError("profile record does not reconstruct from DP.1 sampler")
    if population["population_fingerprint"] != population_fingerprint(
        population["configuration"], devices,
    ):
        raise ValueError("population fingerprint mismatch")


def _scientific_scalar(record: Mapping[str, Any]) -> ScientificScalarSpace:
    _require_fields(record, {"bounds", "strata_count"}, "scientific scalar space")
    return ScientificScalarSpace(tuple(record["bounds"]), record["strata_count"])


def _reconstructed_scientific_configuration(
    record: Mapping[str, Any],
) -> ScientificPopulationConfiguration:
    expected = {
        "device_count", "population_seed", "profile_contract_version",
        "population_contract_version", "configuration_version", "sampler_version",
        "rng_derivation_version", "population_namespace", "scalar_coverage_policy",
        "endpoint_policy", "cross_dimension_pairing_policy", "calibration_status",
        "timing", "activity", "recovery", "renewal", "lifecycle",
    }
    _require_fields(record, expected, "scientific population configuration")
    timing_record = record["timing"]
    timing_fields = {
        "auth_center", "auth_dispersion", "transition_center", "transition_dispersion",
        "access_center", "access_dispersion", "renewal_center", "renewal_dispersion",
    }
    _require_fields(timing_record, timing_fields, "scientific timing space")
    lifecycle_record = record["lifecycle"]
    _require_fields(
        lifecycle_record, {"subdivision", "partition_version", "within_cell_policy"},
        "scientific lifecycle space",
    )
    configuration = ScientificPopulationConfiguration(
        device_count=record["device_count"],
        population_seed=record["population_seed"],
        profile_contract_version=record["profile_contract_version"],
        population_contract_version=record["population_contract_version"],
        configuration_version=record["configuration_version"],
        sampler_version=record["sampler_version"],
        rng_derivation_version=record["rng_derivation_version"],
        population_namespace=record["population_namespace"],
        scalar_coverage_policy=record["scalar_coverage_policy"],
        endpoint_policy=record["endpoint_policy"],
        cross_dimension_pairing_policy=record["cross_dimension_pairing_policy"],
        calibration_status=record["calibration_status"],
        timing=ScientificTimingSpace(**{
            name: _scientific_scalar(timing_record[name]) for name in timing_fields
        }),
        activity=_scientific_scalar(record["activity"]),
        recovery=_scientific_scalar(record["recovery"]),
        renewal=_scientific_scalar(record["renewal"]),
        lifecycle=ScientificLifecycleSpace(**lifecycle_record),
    )
    if scientific_configuration_record(configuration) != dict(record):
        raise ValueError("scientific population configuration does not round-trip")
    return configuration


def _validate_scientific_device_profile_provenance(artifact: Mapping[str, Any]) -> None:
    _require_fields(artifact, {
        "authority", "provenance_schema_version", "run_id", "population", "devices",
    }, "scientific profile artifact")
    if artifact["authority"] != PROFILE_PROVENANCE_AUTHORITY:
        raise ValueError("incorrect profile provenance authority")
    if artifact["provenance_schema_version"] != SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("incorrect scientific profile provenance schema")
    if not artifact["run_id"]:
        raise ValueError("profile artifact run_id is required")
    population = artifact["population"]
    _require_fields(population, {
        "configuration", "device_count", "stable_population_device_keys",
        "sampler_version", "rng_derivation_version", "fingerprint_version",
        "population_fingerprint", "coverage_diagnostics",
    }, "scientific population provenance")
    configuration = _reconstructed_scientific_configuration(population["configuration"])
    if population["sampler_version"] != SCIENTIFIC_POPULATION_SAMPLER_VERSION:
        raise ValueError("unsupported scientific population sampler")
    if population["rng_derivation_version"] != SCIENTIFIC_POPULATION_RNG_VERSION:
        raise ValueError("unsupported scientific population RNG derivation")
    if population["fingerprint_version"] != SCIENTIFIC_POPULATION_FINGERPRINT_VERSION:
        raise ValueError("unsupported scientific population fingerprint")
    devices = artifact["devices"]
    if not isinstance(devices, list) or len(devices) != population["device_count"]:
        raise ValueError("scientific population device count is inconsistent")
    ids, keys, profile_ids = set(), set(), set()
    bindings = []
    for record in devices:
        _require_fields(record, {
            "device_id", "population_device_key", "behavioral_profile_id",
            "profile_seed_reference", "profile_contract_version", "latent_tendencies",
            "scalar_strata", "lifecycle_cell_id", "component_references",
        }, "scientific device profile")
        _reconstructed_profile(
            record["latent_tendencies"], profile_id=record["behavioral_profile_id"],
            contract_version=record["profile_contract_version"],
        )
        ids.add(record["device_id"])
        keys.add(record["population_device_key"])
        profile_ids.add(record["behavioral_profile_id"])
        bindings.append((record["device_id"], record["population_device_key"]))
    if not (len(ids) == len(keys) == len(profile_ids) == len(devices)):
        raise ValueError("duplicate scientific device ID, population key or profile ID")
    if sorted(keys) != population["stable_population_device_keys"]:
        raise ValueError("scientific population stable key set is inconsistent")
    expected = build_scientific_profile_population(bindings, configuration)
    expected_by_key = {
        item.population_device_key: _scientific_device_record(item)
        for item in expected.assignments
    }
    for record in devices:
        if record != expected_by_key[record["population_device_key"]]:
            raise ValueError("scientific profile record does not reconstruct")
    if population["coverage_diagnostics"] != dict(expected.diagnostics):
        raise ValueError("scientific population diagnostics do not reconstruct")
    if population["population_fingerprint"] != expected.population_fingerprint:
        raise ValueError("scientific population fingerprint mismatch")


def validate_device_profile_provenance(artifact: Mapping[str, Any]) -> None:
    """Dispatch exact reconstruction by recorded provenance schema version."""
    schema = artifact.get("provenance_schema_version")
    if schema == PROFILE_PROVENANCE_SCHEMA_VERSION:
        _validate_legacy_device_profile_provenance(artifact)
    elif schema == SCIENTIFIC_PROFILE_PROVENANCE_SCHEMA_VERSION:
        _validate_scientific_device_profile_provenance(artifact)
    else:
        raise ValueError("unsupported profile provenance schema")


def load_device_profile_provenance(path: str | Path) -> dict[str, Any]:
    """Reload and independently validate the dedicated JSON artifact."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_device_profile_provenance(artifact)
    artifact["devices"].sort(key=lambda record: record["population_device_key"])
    return artifact
