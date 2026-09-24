"""Immutable scientific-composition values; all are privileged metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional


COMPOSITION_SPECIFICATION_SCHEMA_VERSION = "4d4b-composition-specification-v1"
SCIENTIFIC_ROLE_SCHEMA_VERSION = "4d4b-scientific-role-v1"
CONTROL_MEMBERSHIP_SCHEMA_VERSION = "4d4b-control-membership-v1"
DUPLICATE_EVIDENCE_POLICY_VERSION = "4d4b-duplicate-evidence-policy-v1"
ELIGIBILITY_EDGE_SCHEMA_VERSION = "4d4b-eligibility-edge-v1"
COMPOSITION_PLAN_SCHEMA_VERSION = "4d4b-composition-plan-v1"
INFEASIBILITY_REPORT_SCHEMA_VERSION = "4d4b-infeasibility-report-v1"


class RoleKind(str, Enum):
    ORDINARY_NORMAL = "ordinary_normal"
    HARD_NEGATIVE = "hard_negative"
    ANOMALY = "anomaly"


class DuplicateEvidencePolicy(str, Enum):
    """Post-build policy; conflicting truth is never policy-overridable."""

    REPORT_ONLY = "REPORT_ONLY"
    REJECT_DUPLICATE_EVIDENCE = "REJECT_DUPLICATE_EVIDENCE"


@dataclass(frozen=True, order=True)
class ScientificRole:
    kind: RoleKind
    variant_id: Optional[str] = None
    schema_version: str = field(default=SCIENTIFIC_ROLE_SCHEMA_VERSION, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCIENTIFIC_ROLE_SCHEMA_VERSION:
            raise ValueError("unsupported scientific-role schema")
        if (self.kind is RoleKind.ORDINARY_NORMAL) != (self.variant_id is None):
            raise ValueError("ordinary role has no variant; HN/anomaly roles require one")

    @property
    def key(self) -> str:
        return self.kind.value if self.variant_id is None else f"{self.kind.value}:{self.variant_id}"

    def to_record(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind.value,
                "variant_id": self.variant_id, "key": self.key}


ORDINARY_NORMAL = ScientificRole(RoleKind.ORDINARY_NORMAL)


class ControlMembership(str, Enum):
    BROAD_POPULATION = "broad_population_normal"
    ACCESS = "access_applicable_control"
    MULTI_ATTEMPT = "multi_attempt_control"
    REPEATED_CURRENT_TOKEN = "repeated_current_token_control"
    REPLACEMENT = "replacement_control"
    IDENTITY_LIFECYCLE = "identity_lifecycle_control"
    BINDING = "binding_control"


@dataclass(frozen=True)
class RoleRequest:
    role: ScientificRole
    count: int
    minimum_distinct_devices: int = 0
    maximum_per_device: Optional[int] = None

    def __post_init__(self) -> None:
        if self.count < 0 or self.minimum_distinct_devices < 0:
            raise ValueError("role counts and device minima must be nonnegative")
        if self.minimum_distinct_devices > self.count:
            raise ValueError("device minimum cannot exceed requested count")
        if self.maximum_per_device is not None and self.maximum_per_device < 1:
            raise ValueError("maximum_per_device must be positive")


@dataclass(frozen=True)
class ControlReserveRequirement:
    membership: ControlMembership
    minimum_ordinary_count: int

    def __post_init__(self) -> None:
        if self.minimum_ordinary_count < 0:
            raise ValueError("reserve count must be nonnegative")


@dataclass(frozen=True)
class CompositionSpecification:
    role_requests: tuple[RoleRequest, ...]
    schedule_seed: int
    ordinary_control_reserves: tuple[ControlReserveRequirement, ...] = ()
    structural_matching: Mapping[str, str] = field(default_factory=dict)
    profile_balancing: Mapping[str, str] = field(default_factory=dict)
    specification_version: str = COMPOSITION_SPECIFICATION_SCHEMA_VERSION
    standard_one_role_per_base: bool = True
    duplicate_evidence_policy: DuplicateEvidencePolicy = DuplicateEvidencePolicy.REPORT_ONLY
    purpose: str = "non_final_engineering_composition"

    def __post_init__(self) -> None:
        if self.specification_version != COMPOSITION_SPECIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported composition specification")
        if isinstance(self.schedule_seed, bool) or not isinstance(self.schedule_seed, int):
            raise ValueError("schedule_seed must be an integer")
        if not self.standard_one_role_per_base:
            raise ValueError("4D.4B implements standard one-role-per-base mode only")
        keys = [request.role.key for request in self.role_requests]
        if len(keys) != len(set(keys)):
            raise ValueError("each scientific role may be requested once")
        if not isinstance(self.duplicate_evidence_policy, DuplicateEvidencePolicy):
            raise ValueError("duplicate evidence requires an explicit versioned policy")
        object.__setattr__(self, "structural_matching", MappingProxyType(dict(self.structural_matching)))
        object.__setattr__(self, "profile_balancing", MappingProxyType(dict(self.profile_balancing)))

    def to_record(self) -> dict[str, Any]:
        return {
            "specification_version": self.specification_version,
            "schedule_seed": self.schedule_seed,
            "standard_one_role_per_base": self.standard_one_role_per_base,
            "duplicate_evidence_policy_version": DUPLICATE_EVIDENCE_POLICY_VERSION,
            "duplicate_evidence_policy": self.duplicate_evidence_policy.value,
            "purpose": self.purpose,
            "role_requests": [
                {**asdict(item), "role": item.role.to_record()}
                for item in sorted(self.role_requests, key=lambda value: value.role.key)
            ],
            "ordinary_control_reserves": [
                {"membership": item.membership.value,
                 "minimum_ordinary_count": item.minimum_ordinary_count}
                for item in sorted(
                    self.ordinary_control_reserves, key=lambda value: value.membership.value
                )
            ],
            "structural_matching": dict(self.structural_matching),
            "profile_balancing": dict(self.profile_balancing),
        }


@dataclass(frozen=True)
class EligibleRoleEdge:
    base_trace_reference: str
    role: ScientificRole
    device_key: str
    family: str
    applicability_reference: str
    control_memberships: tuple[str, ...]
    structural_descriptors: Mapping[str, Any]
    profile_descriptors: Mapping[str, Any]
    scheduler_tie_break_reference: str
    schema_version: str = ELIGIBILITY_EDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_descriptors", MappingProxyType(dict(self.structural_descriptors)))
        object.__setattr__(self, "profile_descriptors", MappingProxyType(dict(self.profile_descriptors)))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_trace_reference": self.base_trace_reference,
            "role": self.role.to_record(), "device_key": self.device_key,
            "family": self.family, "applicability_reference": self.applicability_reference,
            "control_memberships": list(self.control_memberships),
            "structural_descriptors": dict(self.structural_descriptors),
            "profile_descriptors": dict(self.profile_descriptors),
            "scheduler_tie_break_reference": self.scheduler_tie_break_reference,
        }


@dataclass(frozen=True)
class CompositionAssignment:
    base_trace_reference: str
    role: ScientificRole
    variant_id: Optional[str]
    family: str
    device_key: str
    applicability_reference: str
    control_memberships: tuple[str, ...]
    structural_stratum_reference: str
    profile_stratum_reference: str
    scheduler_tie_break_reference: str
    rationale: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["role"] = self.role.to_record()
        return record


@dataclass(frozen=True)
class CompositionInfeasibilityReport:
    requested_counts: Mapping[str, int]
    maximum_immediately_eligible_support: Mapping[str, int]
    distinct_eligible_devices: Mapping[str, int]
    eligible_overlap: Mapping[str, int]
    bases_blocked_by_exclusivity: Mapping[str, int]
    reserve_conflicts: tuple[str, ...]
    structural_conflicts: tuple[str, ...]
    profile_conflicts: tuple[str, ...]
    unsatisfied_constraints: tuple[str, ...]
    matched_counts: Mapping[str, int]
    schema_version: str = INFEASIBILITY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("requested_counts", "maximum_immediately_eligible_support",
                     "distinct_eligible_devices", "eligible_overlap",
                     "bases_blocked_by_exclusivity", "matched_counts"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_counts": dict(self.requested_counts),
            "maximum_immediately_eligible_support": dict(
                self.maximum_immediately_eligible_support
            ),
            "distinct_eligible_devices": dict(self.distinct_eligible_devices),
            "eligible_overlap": dict(self.eligible_overlap),
            "bases_blocked_by_exclusivity": dict(self.bases_blocked_by_exclusivity),
            "reserve_conflicts": list(self.reserve_conflicts),
            "structural_conflicts": list(self.structural_conflicts),
            "profile_conflicts": list(self.profile_conflicts),
            "unsatisfied_constraints": list(self.unsatisfied_constraints),
            "matched_counts": dict(self.matched_counts),
        }


@dataclass(frozen=True)
class CompositionPlan:
    specification: CompositionSpecification
    population_fingerprint: str
    pool_fingerprint: str
    scheduler_version: str
    rng_derivation_version: str
    eligibility_digest: str
    assignments: tuple[CompositionAssignment, ...]
    unassigned_base_references: tuple[str, ...]
    feasibility_status: str
    pre_transformation_diagnostics_json: str
    composition_fingerprint: str
    infeasibility: Optional[CompositionInfeasibilityReport] = None
    schema_version: str = COMPOSITION_PLAN_SCHEMA_VERSION

    @property
    def diagnostics(self) -> dict[str, Any]:
        return json.loads(self.pre_transformation_diagnostics_json)

    def normalized_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "specification": self.specification.to_record(),
            "population_fingerprint": self.population_fingerprint,
            "pool_fingerprint": self.pool_fingerprint,
            "scheduler_version": self.scheduler_version,
            "rng_derivation_version": self.rng_derivation_version,
            "eligibility_digest": self.eligibility_digest,
            "assignments": [item.to_record() for item in self.assignments],
            "unassigned_base_references": list(self.unassigned_base_references),
            "feasibility_status": self.feasibility_status,
            "pre_transformation_diagnostics": self.diagnostics,
            "infeasibility": self.infeasibility.to_record() if self.infeasibility else None,
        }
