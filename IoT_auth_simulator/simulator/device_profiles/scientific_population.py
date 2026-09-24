"""Fixed-cohort scientific Device Profile construction and diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from simulator.config.scientific_population import (
    SCIENTIFIC_POPULATION_RNG_VERSION,
    SCIENTIFIC_POPULATION_SAMPLER_VERSION,
    ScientificPopulationConfiguration,
    ScientificScalarSpace,
)
from simulator.device_profiles.profile import (
    DeviceProfile,
    LifecycleWeights,
    RelativeTimingTendency,
    TimingTendencies,
)


SCIENTIFIC_POPULATION_FINGERPRINT_VERSION = "4d2-scientific-population-sha256-v1"
SCIENTIFIC_DIAGNOSTICS_VERSION = "4d2-scientific-population-diagnostics-v1"

SCALAR_DIMENSIONS = (
    "T.AUTH.center", "T.AUTH.dispersion",
    "T.TRANSITION.center", "T.TRANSITION.dispersion",
    "T.ACCESS.center", "T.ACCESS.dispersion",
    "T.RENEWAL.center", "T.RENEWAL.dispersion",
    "A", "R", "N",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _reference(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _open_unit(reference: str, component: str) -> float:
    digest = hashlib.sha256(f"{reference}|{component}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:7], "big") >> 3  # 53 deterministic bits.
    return (integer + 1) / ((2**53) + 2)


def scientific_configuration_record(
    configuration: ScientificPopulationConfiguration,
) -> dict[str, object]:
    """Canonical JSON-compatible scientific configuration."""
    return json.loads(json.dumps(asdict(configuration), allow_nan=False))


def _cohort_reference(
    configuration: ScientificPopulationConfiguration,
    stable_keys: Sequence[str],
) -> str:
    return _reference({
        "sampler_version": SCIENTIFIC_POPULATION_SAMPLER_VERSION,
        "rng_derivation_version": SCIENTIFIC_POPULATION_RNG_VERSION,
        "population_namespace": configuration.population_namespace,
        "population_seed": configuration.population_seed,
        "stable_population_device_keys": list(stable_keys),
    })


def _component_reference(
    configuration: ScientificPopulationConfiguration,
    stable_keys: Sequence[str],
    component: str,
    component_configuration: object,
    operation: str,
) -> str:
    """Isolate component changes while retaining fixed-cohort identity."""
    return _reference({
        "cohort_reference": _cohort_reference(configuration, stable_keys),
        "rng_derivation_version": configuration.rng_derivation_version,
        "component": component,
        "component_configuration": component_configuration,
        "operation": operation,
    })


def _deterministic_permutation(values: Sequence[object], reference: str) -> list[object]:
    decorated = []
    for index, value in enumerate(values):
        digest = hashlib.sha256(
            _canonical_json({"reference": reference, "index": index, "value": value})
        ).digest()
        decorated.append((digest, index, value))
    return [value for _digest, _index, value in sorted(decorated)]


def _balanced_labels(count: int, strata: int, assignment_reference: str) -> list[int]:
    base, remainder = divmod(count, strata)
    extra_order = _deterministic_permutation(list(range(strata)), assignment_reference + "|extra")
    extras = set(extra_order[:remainder])
    tokens: list[tuple[int, int]] = []
    for stratum in range(strata):
        for occurrence in range(base + int(stratum in extras)):
            tokens.append((stratum, occurrence))
    shuffled = _deterministic_permutation(tokens, assignment_reference + "|tokens")
    return [stratum for stratum, _occurrence in shuffled]


def scientific_scalar_from_reference(
    space: ScientificScalarSpace,
    stratum: int,
    within_reference: str,
) -> float:
    unit = _open_unit(within_reference, "within-stratum")
    position = (stratum + unit) / space.strata_count
    value = space.bounds[0] + (space.bounds[1] - space.bounds[0]) * position
    if not space.bounds[0] < value < space.bounds[1]:
        raise ValueError("scientific scalar realization is not strictly interior")
    return value


@dataclass(frozen=True)
class LifecycleSimplexCell:
    """One equal-area triangle in a symmetric barycentric grid partition."""

    cell_id: str
    orientation: str
    lattice_index: tuple[int, int, int]
    vertices: tuple[tuple[float, float, float], ...]


def lifecycle_simplex_cells(subdivision: int) -> tuple[LifecycleSimplexCell, ...]:
    """Return the m^2 explicit grid triangles covering the simplex.

    Upward cells use integer ``a+b+c=m-1`` and vertices obtained by adding
    one lattice unit to each coordinate in turn. Downward cells use
    ``a+b+c=m-2`` and add one unit to coordinate pairs. Coordinate permutation
    maps every declared cell onto another declared cell.
    """
    if subdivision < 2:
        raise ValueError("simplex subdivision must be >= 2")
    cells: list[LifecycleSimplexCell] = []
    m = subdivision
    for orientation, total in (("U", m - 1), ("D", m - 2)):
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                if orientation == "U":
                    integer_vertices = (
                        (a + 1, b, c), (a, b + 1, c), (a, b, c + 1),
                    )
                else:
                    integer_vertices = (
                        (a + 1, b + 1, c),
                        (a + 1, b, c + 1),
                        (a, b + 1, c + 1),
                    )
                vertices = tuple(
                    tuple(value / m for value in vertex)
                    for vertex in integer_vertices
                )
                cells.append(LifecycleSimplexCell(
                    cell_id=f"{orientation}-{a}-{b}-{c}",
                    orientation=orientation,
                    lattice_index=(a, b, c),
                    vertices=vertices,
                ))
    cells.sort(key=lambda cell: cell.cell_id)
    if len(cells) != m * m:
        raise AssertionError("triangular simplex partition must contain m^2 cells")
    return tuple(cells)


def lifecycle_point_from_reference(
    cell: LifecycleSimplexCell,
    within_reference: str,
) -> tuple[float, float, float]:
    u = _open_unit(within_reference, "triangle-u")
    v = _open_unit(within_reference, "triangle-v")
    root = math.sqrt(u)
    coefficients = (1.0 - root, root * (1.0 - v), root * v)
    point = tuple(
        sum(coefficients[index] * cell.vertices[index][coordinate] for index in range(3))
        for coordinate in range(3)
    )
    # Normalize one floating-point accumulation without changing cell geometry.
    total = sum(point)
    point = tuple(value / total for value in point)
    if not all(0.0 < value < 1.0 for value in point):
        raise ValueError("scientific lifecycle point must be strictly inside the simplex")
    return point


def _scalar_spaces(
    configuration: ScientificPopulationConfiguration,
) -> dict[str, ScientificScalarSpace]:
    timing = configuration.timing
    return {
        "T.AUTH.center": timing.auth_center,
        "T.AUTH.dispersion": timing.auth_dispersion,
        "T.TRANSITION.center": timing.transition_center,
        "T.TRANSITION.dispersion": timing.transition_dispersion,
        "T.ACCESS.center": timing.access_center,
        "T.ACCESS.dispersion": timing.access_dispersion,
        "T.RENEWAL.center": timing.renewal_center,
        "T.RENEWAL.dispersion": timing.renewal_dispersion,
        "A": configuration.activity,
        "R": configuration.recovery,
        "N": configuration.renewal,
    }


@dataclass(frozen=True)
class ScientificProfileAssignment:
    device_id: str
    population_device_key: str
    profile_seed_reference: str
    profile: DeviceProfile
    scalar_strata: Mapping[str, int]
    lifecycle_cell_id: str
    component_references: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalar_strata", MappingProxyType(dict(self.scalar_strata)))
        object.__setattr__(
            self, "component_references", MappingProxyType(dict(self.component_references)),
        )


@dataclass(frozen=True)
class ScientificDeviceProfilePopulation:
    configuration: ScientificPopulationConfiguration
    assignments: tuple[ScientificProfileAssignment, ...]
    population_fingerprint: str
    diagnostics: Mapping[str, object]
    _by_device_id: Mapping[str, ScientificProfileAssignment] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        index = {item.device_id: item for item in self.assignments}
        if len(index) != len(self.assignments):
            raise ValueError("scientific population device IDs must be unique")
        keys = [item.population_device_key for item in self.assignments]
        if len(set(keys)) != len(keys):
            raise ValueError("scientific population device keys must be unique")
        if len(keys) != self.configuration.device_count:
            raise ValueError("scientific population configuration/device count mismatch")
        object.__setattr__(self, "_by_device_id", MappingProxyType(index))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def profile_for_device(self, device_id: str) -> DeviceProfile:
        return self._by_device_id[device_id].profile

    def assignment_for_device(self, device_id: str) -> ScientificProfileAssignment:
        return self._by_device_id[device_id]

    @property
    def behavioral_profiles(self) -> Mapping[str, DeviceProfile]:
        return MappingProxyType({key: item.profile for key, item in self._by_device_id.items()})


def _latent_record(profile: DeviceProfile) -> dict[str, object]:
    return {
        "T": {
            "AUTH": asdict(profile.timing.auth),
            "TRANSITION": asdict(profile.timing.transition),
            "ACCESS": asdict(profile.timing.access),
            "RENEWAL": asdict(profile.timing.renewal),
        },
        "A": profile.activity,
        "R": profile.recovery,
        "N": profile.renewal,
        "L": asdict(profile.lifecycle),
    }


def scientific_population_fingerprint(
    configuration: ScientificPopulationConfiguration,
    assignments: Sequence[ScientificProfileAssignment],
) -> str:
    normalized = {
        "fingerprint_version": SCIENTIFIC_POPULATION_FINGERPRINT_VERSION,
        "configuration": scientific_configuration_record(configuration),
        "stable_population_device_keys": sorted(
            item.population_device_key for item in assignments
        ),
        "assignments": sorted(({
            "population_device_key": item.population_device_key,
            "behavioral_profile_id": item.profile.behavioral_profile_id,
            "profile_contract_version": item.profile.contract_version,
            "latent_tendencies": _latent_record(item.profile),
        } for item in assignments), key=lambda record: record["population_device_key"]),
    }
    return _reference(normalized)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    def value_at(fraction: float) -> float:
        location = fraction * (len(ordered) - 1)
        low = math.floor(location)
        high = math.ceil(location)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (location - low)
    return {"q25": value_at(0.25), "q50": value_at(0.5), "q75": value_at(0.75)}


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_sum = sum((a - left_mean) ** 2 for a in left)
    right_sum = sum((b - right_mean) ** 2 for b in right)
    return 0.0 if not left_sum or not right_sum else numerator / math.sqrt(left_sum * right_sum)


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index, _value in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def scientific_population_diagnostics(
    configuration: ScientificPopulationConfiguration,
    assignments: Sequence[ScientificProfileAssignment],
) -> dict[str, object]:
    scalar_spaces = _scalar_spaces(configuration)
    scalar_values: dict[str, list[float]] = {name: [] for name in SCALAR_DIMENSIONS}
    lifecycle_values: list[tuple[float, float, float]] = []
    for item in assignments:
        profile = item.profile
        values = {
            "T.AUTH.center": profile.timing.auth.center_modifier,
            "T.AUTH.dispersion": profile.timing.auth.dispersion_modifier,
            "T.TRANSITION.center": profile.timing.transition.center_modifier,
            "T.TRANSITION.dispersion": profile.timing.transition.dispersion_modifier,
            "T.ACCESS.center": profile.timing.access.center_modifier,
            "T.ACCESS.dispersion": profile.timing.access.dispersion_modifier,
            "T.RENEWAL.center": profile.timing.renewal.center_modifier,
            "T.RENEWAL.dispersion": profile.timing.renewal.dispersion_modifier,
            "A": profile.activity, "R": profile.recovery, "N": profile.renewal,
        }
        for name, value in values.items():
            scalar_values[name].append(value)
        lifecycle_values.append((
            profile.lifecycle.auth, profile.lifecycle.session, profile.lifecycle.access,
        ))
    scalar = {}
    for name in SCALAR_DIMENSIONS:
        values = scalar_values[name]
        space = scalar_spaces[name]
        occupancy = [0] * space.strata_count
        for item in assignments:
            occupancy[item.scalar_strata[name]] += 1
        scalar[name] = {
            "stratum_occupancy": occupancy,
            "expected_minimum_occupancy": len(assignments) // space.strata_count,
            "expected_maximum_occupancy": math.ceil(len(assignments) / space.strata_count),
            "minimum": min(values), "maximum": max(values),
            "quantiles": _quantiles(values),
            "endpoint_violations": sum(
                not space.bounds[0] < value < space.bounds[1] for value in values
            ),
        }
    cell_ids = [cell.cell_id for cell in lifecycle_simplex_cells(
        configuration.lifecycle.subdivision
    )]
    cell_occupancy = {cell_id: 0 for cell_id in cell_ids}
    for item in assignments:
        cell_occupancy[item.lifecycle_cell_id] += 1
    coordinate_names = ("L.AUTH", "L.SESSION", "L.ACCESS")
    lifecycle_columns = {
        name: [value[index] for value in lifecycle_values]
        for index, name in enumerate(coordinate_names)
    }
    effective = {**scalar_values, "L.AUTH": lifecycle_columns["L.AUTH"],
                 "L.SESSION": lifecycle_columns["L.SESSION"]}
    names = list(effective)
    pearson = {name: {} for name in names}
    spearman = {name: {} for name in names}
    max_pearson = max_spearman = 0.0
    for left in names:
        for right in names:
            p = _pearson(effective[left], effective[right])
            s = _pearson(_ranks(effective[left]), _ranks(effective[right]))
            pearson[left][right] = p
            spearman[left][right] = s
            if left != right:
                max_pearson = max(max_pearson, abs(p))
                max_spearman = max(max_spearman, abs(s))
    normalized_profiles = [_canonical_json(_latent_record(item.profile)) for item in assignments]
    duplicate_count = len(normalized_profiles) - len(set(normalized_profiles))
    vectors = [
        [*(scalar_values[name][index] for name in SCALAR_DIMENSIONS),
         *lifecycle_values[index]]
        for index in range(len(assignments))
    ]
    distances = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        for index, left in enumerate(vectors) for right in vectors[index + 1:]
    ]
    return {
        "diagnostics_version": SCIENTIFIC_DIAGNOSTICS_VERSION,
        "descriptive_only_no_resampling": True,
        "scalar_marginals": scalar,
        "lifecycle": {
            "cell_occupancy": cell_occupancy,
            "coordinate_minimums": {
                name: min(values) for name, values in lifecycle_columns.items()
            },
            "coordinate_maximums": {
                name: max(values) for name, values in lifecycle_columns.items()
            },
            "minimum_weight": min(min(value) for value in lifecycle_values),
            "maximum_normalization_error": max(
                abs(sum(value) - 1.0) for value in lifecycle_values
            ),
        },
        "cross_dimension": {
            "effective_dimensions": names,
            "pearson": pearson, "spearman": spearman,
            "maximum_absolute_off_diagonal_pearson": max_pearson,
            "maximum_absolute_off_diagonal_spearman": max_spearman,
            "acceptance_threshold": None,
        },
        "identity": {
            "duplicate_exact_profile_count": duplicate_count,
            "pairwise_distance_minimum": min(distances) if distances else None,
            "pairwise_distance_maximum": max(distances) if distances else None,
            "pairwise_distance_mean": statistics.fmean(distances) if distances else None,
        },
    }


def build_scientific_profile_population(
    device_bindings: Iterable[tuple[str, str]],
    configuration: ScientificPopulationConfiguration,
) -> ScientificDeviceProfilePopulation:
    """Build one immutable balanced cohort, canonicalized by stable key."""
    bindings = list(device_bindings)
    if len(bindings) != configuration.device_count:
        raise ValueError("scientific population configuration/device count mismatch")
    if any(not device_id or not key for device_id, key in bindings):
        raise ValueError("device ID and stable population key are required")
    if len({device_id for device_id, _key in bindings}) != len(bindings):
        raise ValueError("scientific population device IDs must be unique")
    if len({key for _device_id, key in bindings}) != len(bindings):
        raise ValueError("scientific population stable keys must be unique")
    by_key = {key: device_id for device_id, key in bindings}
    stable_keys = tuple(sorted(by_key))
    scalar_spaces = _scalar_spaces(configuration)
    scalar_values: dict[str, dict[str, float]] = {}
    scalar_strata: dict[str, dict[str, int]] = {}
    component_references: dict[str, dict[str, str]] = {key: {} for key in stable_keys}
    for dimension, space in scalar_spaces.items():
        component_config = asdict(space)
        assignment_reference = _component_reference(
            configuration, stable_keys, dimension, component_config, "assignment",
        )
        labels = _balanced_labels(len(stable_keys), space.strata_count, assignment_reference)
        scalar_values[dimension] = {}
        scalar_strata[dimension] = {}
        for key, stratum in zip(stable_keys, labels):
            within_reference = _component_reference(
                configuration, stable_keys, dimension,
                {**component_config, "population_device_key": key}, "within-stratum",
            )
            scalar_values[dimension][key] = scientific_scalar_from_reference(
                space, stratum, within_reference,
            )
            scalar_strata[dimension][key] = stratum
            component_references[key][dimension] = within_reference

    cells = lifecycle_simplex_cells(configuration.lifecycle.subdivision)
    lifecycle_config = asdict(configuration.lifecycle)
    lifecycle_assignment_reference = _component_reference(
        configuration, stable_keys, "L", lifecycle_config, "cell-assignment",
    )
    cell_labels = _balanced_labels(
        len(stable_keys), len(cells), lifecycle_assignment_reference,
    )
    assignments: list[ScientificProfileAssignment] = []
    configuration_reference = _reference(scientific_configuration_record(configuration))
    cohort_reference = _cohort_reference(configuration, stable_keys)
    for key, cell_index in zip(stable_keys, cell_labels):
        cell = cells[cell_index]
        lifecycle_reference = _component_reference(
            configuration, stable_keys, "L",
            {**lifecycle_config, "population_device_key": key}, "within-cell",
        )
        lifecycle = lifecycle_point_from_reference(cell, lifecycle_reference)
        component_references[key]["L"] = lifecycle_reference
        profile_id = "behavioral-profile-scientific-v1-" + _reference({
            "configuration_reference": configuration_reference,
            "cohort_reference": cohort_reference,
            "population_device_key": key,
        }).split(":", 1)[1][:24]
        profile = DeviceProfile(
            behavioral_profile_id=profile_id,
            contract_version=configuration.profile_contract_version,
            timing=TimingTendencies(
                auth=RelativeTimingTendency(
                    scalar_values["T.AUTH.center"][key],
                    scalar_values["T.AUTH.dispersion"][key],
                ),
                transition=RelativeTimingTendency(
                    scalar_values["T.TRANSITION.center"][key],
                    scalar_values["T.TRANSITION.dispersion"][key],
                ),
                access=RelativeTimingTendency(
                    scalar_values["T.ACCESS.center"][key],
                    scalar_values["T.ACCESS.dispersion"][key],
                ),
                renewal=RelativeTimingTendency(
                    scalar_values["T.RENEWAL.center"][key],
                    scalar_values["T.RENEWAL.dispersion"][key],
                ),
            ),
            activity=scalar_values["A"][key],
            recovery=scalar_values["R"][key],
            renewal=scalar_values["N"][key],
            lifecycle=LifecycleWeights(*lifecycle),
        )
        assignments.append(ScientificProfileAssignment(
            device_id=by_key[key],
            population_device_key=key,
            profile_seed_reference=_reference({
                "kind": "scientific-profile-root",
                "cohort_reference": cohort_reference,
                "population_device_key": key,
            }),
            profile=profile,
            scalar_strata={name: scalar_strata[name][key] for name in SCALAR_DIMENSIONS},
            lifecycle_cell_id=cell.cell_id,
            component_references=component_references[key],
        ))
    assignment_tuple = tuple(assignments)
    fingerprint = scientific_population_fingerprint(configuration, assignment_tuple)
    diagnostics = scientific_population_diagnostics(configuration, assignment_tuple)
    # Structural coverage gate; correlations are deliberately descriptive only.
    for record in diagnostics["scalar_marginals"].values():
        occupancy = record["stratum_occupancy"]
        if min(occupancy) < 1 or max(occupancy) - min(occupancy) > 1:
            raise ValueError("scientific scalar coverage is not balanced")
        if record["endpoint_violations"]:
            raise ValueError("scientific scalar endpoint policy was violated")
    lifecycle_occupancy = diagnostics["lifecycle"]["cell_occupancy"].values()
    if min(lifecycle_occupancy) < 1 or max(lifecycle_occupancy) - min(lifecycle_occupancy) > 1:
        raise ValueError("scientific lifecycle-cell coverage is not balanced")
    if diagnostics["lifecycle"]["minimum_weight"] <= 0:
        raise ValueError("scientific lifecycle endpoint policy was violated")
    if diagnostics["lifecycle"]["maximum_normalization_error"] > 1e-12:
        raise ValueError("scientific lifecycle normalization failed")
    return ScientificDeviceProfilePopulation(
        configuration, assignment_tuple, fingerprint, diagnostics,
    )
