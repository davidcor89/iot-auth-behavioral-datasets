"""Small immutable planning values for applicability-first lifecycle choice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

from simulator.event_model import EventType

if TYPE_CHECKING:
    from simulator.engines.scenario_engine import ScenarioSpec


TRACE_PLANNING_PROVENANCE_SCHEMA_VERSION = "4c5-trace-planning-provenance-v6"


class LifecycleDepth(str, Enum):
    """Generator lifecycle-depth abstractions, not BM states or invariants."""

    AUTH = "AUTH"
    SESSION = "SESSION"
    ACCESS = "ACCESS"


class RecoveryChoice(str, Enum):
    """Supported successful ordinary authentication trajectories."""

    DIRECT = "DIRECT"
    ONE_RETRY = "ONE_RETRY"


class RenewalChoice(str, Enum):
    """Supported post-ACCESS continuations; not lifecycle depths."""

    NO_RENEWAL = "NO_RENEWAL"
    RENEWAL = "RENEWAL"


class ActivityPlacement(str, Enum):
    """Supported placement of protected operations in a planned trace."""

    NONE = "NONE"
    PRE_RENEWAL = "PRE_RENEWAL"


class ConstructionRole(str, Enum):
    """Construction role is independent from the requested lifecycle depth."""

    ORDINARY = "ordinary"
    HARD_NEGATIVE = "hard_negative"
    ANOMALY = "anomaly"


class CompletionMode(str, Enum):
    """Why a trace ends; intentional lifecycle endpoints are not partials."""

    INTENTIONAL_LIFECYCLE_ENDPOINT = "intentional_lifecycle_endpoint"
    VALID_PREFIX = "valid_prefix"
    LEGACY_COMPATIBILITY = "legacy_compatibility"


class PlanningMode(str, Enum):
    """Explicit migration modes keep causal profile planning opt-in."""

    LEGACY_EQUIVALENT = "legacy_equivalent"
    EXPLICIT_LIFECYCLE = "explicit_lifecycle_repertoire"
    PROFILE_CONDITIONED_LIFECYCLE = "profile_conditioned_lifecycle"


class TracePlanningErrorReason(str, Enum):
    NO_SUPPORTED_APPLICABLE_LIFECYCLE = "no_supported_applicable_lifecycle"
    UNSUPPORTED_REQUESTED_LIFECYCLE = "unsupported_requested_lifecycle_or_template"
    INVALID_CONSTRUCTION_ROLE = "invalid_construction_role_for_planner"
    ZERO_APPLICABLE_WEIGHT = "zero_applicable_weight"
    MALFORMED_LIFECYCLE_WEIGHTS = "malformed_lifecycle_weights"
    NO_APPLICABLE_RECOVERY_CHOICE = "no_applicable_recovery_choice"
    MALFORMED_RECOVERY_TENDENCY = "malformed_recovery_tendency"
    UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION = (
        "unsupported_lifecycle_recovery_combination"
    )
    NO_APPLICABLE_RENEWAL_CHOICE = "no_applicable_renewal_choice"
    MALFORMED_RENEWAL_TENDENCY = "malformed_renewal_tendency"
    UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION = (
        "unsupported_lifecycle_renewal_combination"
    )
    NO_APPLICABLE_ACTIVITY_REALIZATION = "no_applicable_activity_realization"
    MALFORMED_ACTIVITY_TENDENCY = "malformed_activity_tendency"
    UNSUPPORTED_ACTIVITY_COMBINATION = "unsupported_activity_combination"


class TracePlanningError(ValueError):
    """Stable planning failure with no silent lifecycle substitution."""

    def __init__(self, reason: TracePlanningErrorReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


@dataclass(frozen=True)
class LegacyScenarioIntent:
    """Immutable ordinary ScenarioSpec snapshot for migration neutrality."""

    scenario_id: str
    scenario_type: str
    generation_profile: str
    normal_steps: tuple[EventType, ...]

    @classmethod
    def from_scenario_spec(cls, spec: "ScenarioSpec") -> "LegacyScenarioIntent":
        if (
            spec.scenario_type != "normal"
            or spec.is_anomaly
            or spec.anomaly_type is not None
            or spec.injection_event is not None
            or spec.rejected_access_prerequisite is not None
        ):
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                "4C.INF legacy planning accepts ordinary ScenarioSpec values only",
            )
        return cls(
            scenario_id=spec.scenario_id,
            scenario_type=spec.scenario_type,
            generation_profile=spec.generation_profile,
            normal_steps=tuple(spec.normal_steps),
        )


@dataclass(frozen=True)
class TraceConstructionRequest:
    """Experiment/build intent before a device-aware ordinary realization."""

    trace_construction_key: str
    scenario_id: str
    construction_role: ConstructionRole
    completion_mode: CompletionMode
    planning_mode: PlanningMode
    requested_lifecycle_depth: Optional[LifecycleDepth]
    legacy_scenario_intent: Optional[LegacyScenarioIntent] = None

    def __post_init__(self) -> None:
        if not self.trace_construction_key or not self.scenario_id:
            raise ValueError("trace construction key and scenario ID are required")
        if self.planning_mode is PlanningMode.LEGACY_EQUIVALENT:
            if self.legacy_scenario_intent is None:
                raise ValueError("legacy-equivalent planning requires legacy scenario intent")
            if self.completion_mode is not CompletionMode.LEGACY_COMPATIBILITY:
                raise ValueError("legacy-equivalent planning requires legacy completion mode")
            if self.requested_lifecycle_depth is None:
                raise ValueError("legacy-equivalent planning requires a lifecycle reference")
        elif self.legacy_scenario_intent is not None:
            raise ValueError("explicit lifecycle planning does not accept legacy scenario intent")
        elif self.completion_mode is not CompletionMode.INTENTIONAL_LIFECYCLE_ENDPOINT:
            raise ValueError("explicit lifecycle repertoire uses intentional endpoints")
        elif self.planning_mode is PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            if self.requested_lifecycle_depth is not None:
                raise ValueError("profile-conditioned planning selects rather than requests depth")
        elif self.requested_lifecycle_depth is None:
            raise ValueError("explicit lifecycle planning requires a requested depth")

    @classmethod
    def explicit_lifecycle(
        cls,
        *,
        trace_construction_key: str,
        scenario_id: str,
        lifecycle_depth: LifecycleDepth,
    ) -> "TraceConstructionRequest":
        return cls(
            trace_construction_key=trace_construction_key,
            scenario_id=scenario_id,
            construction_role=ConstructionRole.ORDINARY,
            completion_mode=CompletionMode.INTENTIONAL_LIFECYCLE_ENDPOINT,
            planning_mode=PlanningMode.EXPLICIT_LIFECYCLE,
            requested_lifecycle_depth=lifecycle_depth,
        )

    @classmethod
    def legacy_equivalent(
        cls,
        *,
        trace_construction_key: str,
        lifecycle_depth: LifecycleDepth,
        scenario_spec: "ScenarioSpec",
    ) -> "TraceConstructionRequest":
        intent = LegacyScenarioIntent.from_scenario_spec(scenario_spec)
        return cls(
            trace_construction_key=trace_construction_key,
            scenario_id=intent.scenario_id,
            construction_role=ConstructionRole.ORDINARY,
            completion_mode=CompletionMode.LEGACY_COMPATIBILITY,
            planning_mode=PlanningMode.LEGACY_EQUIVALENT,
            requested_lifecycle_depth=lifecycle_depth,
            legacy_scenario_intent=intent,
        )

    @classmethod
    def profile_conditioned_lifecycle(
        cls,
        *,
        trace_construction_key: str,
        scenario_id: str,
    ) -> "TraceConstructionRequest":
        return cls(
            trace_construction_key=trace_construction_key,
            scenario_id=scenario_id,
            construction_role=ConstructionRole.ORDINARY,
            completion_mode=CompletionMode.INTENTIONAL_LIFECYCLE_ENDPOINT,
            planning_mode=PlanningMode.PROFILE_CONDITIONED_LIFECYCLE,
            requested_lifecycle_depth=None,
        )


@dataclass(frozen=True)
class ApplicableChoices:
    """Pure intersection of semantic applicability and supported repertoire."""

    lifecycle_depths: tuple[LifecycleDepth, ...]
    supported_repertoire_version: str
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApplicableRecoveryChoices:
    """Profile-independent recovery options after lifecycle selection."""

    lifecycle_depth: LifecycleDepth
    recovery_choices: tuple[RecoveryChoice, ...]
    supported_repertoire_version: str
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApplicableRenewalChoices:
    """Profile-independent continuation options after lifecycle/recovery."""

    lifecycle_depth: LifecycleDepth
    recovery_choice: RecoveryChoice
    renewal_choices: tuple[RenewalChoice, ...]
    supported_repertoire_version: str
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActivityApplicability:
    """Profile-independent activity support after L/R/N resolution."""

    lifecycle_depth: LifecycleDepth
    recovery_choice: RecoveryChoice
    renewal_choice: RenewalChoice
    applicable: bool
    structural_minimum: int
    supported_minimum: int
    supported_maximum: int
    placement: ActivityPlacement
    supported_repertoire_version: str
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LifecycleSelection:
    """One auditable categorical result from applicable normalized weights."""

    applicable_lifecycle_depths: tuple[LifecycleDepth, ...]
    normalized_weights: tuple[tuple[LifecycleDepth, float], ...]
    selected_lifecycle_depth: LifecycleDepth
    random_variate: float


@dataclass(frozen=True)
class RecoverySelection:
    """Recovery result separated from applicability and persistent tendency."""

    applicable_recovery_choices: tuple[RecoveryChoice, ...]
    selected_recovery_choice: RecoveryChoice
    decision_slot: int
    stochastic_decision_used: bool


@dataclass(frozen=True)
class RenewalSelection:
    """Renewal result separated from applicability and persistent tendency."""

    applicable_renewal_choices: tuple[RenewalChoice, ...]
    selected_renewal_choice: RenewalChoice
    decision_slot: int
    stochastic_decision_used: bool


@dataclass(frozen=True)
class ActivityRealization:
    """Protected-operation count selected without retaining A or its draw."""

    selected_protected_operation_count: int
    decision_slot: int
    stochastic_decision_used: bool
    mapping_version: str


@dataclass(frozen=True)
class TracePlan:
    """Resolved ordinary realization; contains no T/A/R/N/L values."""

    trace_construction_key: str
    scenario_id: str
    construction_role: ConstructionRole
    lifecycle_depth: LifecycleDepth
    completion_mode: CompletionMode
    planning_mode: PlanningMode
    supported_repertoire_version: str
    template_reference: str
    event_steps: tuple[EventType, ...]
    generation_profile: str
    planning_rng_reference: str
    planning_rng_derivation_version: str
    planning_rng_namespace: str
    planner_version: str
    applicable_lifecycle_depths: tuple[LifecycleDepth, ...]
    planning_draw_count: int
    profile_conditioned_selection: bool
    recovery_choice: RecoveryChoice
    supported_recovery_repertoire_version: str
    applicable_recovery_choices: tuple[RecoveryChoice, ...]
    recovery_template_reference: str
    profile_conditioned_recovery_selection: bool
    recovery_decision_slot: int
    recovery_decision_stochastic: bool
    renewal_choice: RenewalChoice
    supported_renewal_repertoire_version: str
    applicable_renewal_choices: tuple[RenewalChoice, ...]
    renewal_extension_reference: str
    profile_conditioned_renewal_selection: bool
    renewal_decision_slot: int
    renewal_decision_stochastic: bool
    base_lifecycle_endpoint: EventType
    final_planned_endpoint: EventType
    planning_decision_slot_order_version: str
    activity_applicable: bool = False
    activity_structural_minimum: int = 0
    activity_supported_minimum: int = 0
    activity_supported_maximum: int = 0
    activity_placement: ActivityPlacement = ActivityPlacement.NONE
    supported_activity_repertoire_version: str = "activity-not-enabled"
    activity_mapping_version: str = "activity-not-enabled"
    selected_protected_operation_count: int = 0
    profile_conditioned_activity_selection: bool = False
    activity_decision_slot: int = 4
    activity_decision_stochastic: bool = False
    activity_configuration_version: str = "activity-not-enabled"
    activity_calibration_status: str = "not_applicable"
    activity_exclusion_reasons: tuple[str, ...] = ()

    def provenance_record(self) -> dict[str, object]:
        """Privileged trace-realization record, deliberately without latents."""
        return {
            "planning_provenance_schema_version": TRACE_PLANNING_PROVENANCE_SCHEMA_VERSION,
            "supported_lifecycle_repertoire_version": self.supported_repertoire_version,
            "planner_version": self.planner_version,
            "construction_role": self.construction_role.value,
            "completion_mode": self.completion_mode.value,
            "applicable_lifecycle_depths": [
                depth.value for depth in self.applicable_lifecycle_depths
            ],
            "selected_lifecycle_depth": self.lifecycle_depth.value,
            "planning_mode": self.planning_mode.value,
            "template_reference": self.template_reference,
            "trace_construction_key": self.trace_construction_key,
            "planning_rng_reference": self.planning_rng_reference,
            "planning_rng_derivation_version": self.planning_rng_derivation_version,
            "planning_rng_namespace": self.planning_rng_namespace,
            "planning_draw_count": self.planning_draw_count,
            "profile_association_reference": "provenance_trace.device_id",
            "profile_conditioned_selection": self.profile_conditioned_selection,
            "supported_recovery_repertoire_version": (
                self.supported_recovery_repertoire_version
            ),
            "applicable_recovery_choices": [
                choice.value for choice in self.applicable_recovery_choices
            ],
            "selected_recovery_choice": self.recovery_choice.value,
            "recovery_template_reference": self.recovery_template_reference,
            "profile_conditioned_recovery_selection": (
                self.profile_conditioned_recovery_selection
            ),
            "recovery_decision_slot": self.recovery_decision_slot,
            "recovery_decision_stochastic": self.recovery_decision_stochastic,
            "supported_renewal_repertoire_version": (
                self.supported_renewal_repertoire_version
            ),
            "applicable_renewal_choices": [
                choice.value for choice in self.applicable_renewal_choices
            ],
            "selected_renewal_choice": self.renewal_choice.value,
            "renewal_extension_reference": self.renewal_extension_reference,
            "profile_conditioned_renewal_selection": (
                self.profile_conditioned_renewal_selection
            ),
            "renewal_decision_slot": self.renewal_decision_slot,
            "renewal_decision_stochastic": self.renewal_decision_stochastic,
            "base_lifecycle_endpoint": self.base_lifecycle_endpoint.value,
            "final_planned_endpoint": self.final_planned_endpoint.value,
            "planning_decision_slot_order_version": (
                self.planning_decision_slot_order_version
            ),
            "activity_applicable": self.activity_applicable,
            "activity_structural_minimum": self.activity_structural_minimum,
            "activity_supported_range": [
                self.activity_supported_minimum,
                self.activity_supported_maximum,
            ],
            "activity_placement": self.activity_placement.value,
            "supported_activity_repertoire_version": (
                self.supported_activity_repertoire_version
            ),
            "activity_mapping_version": self.activity_mapping_version,
            "selected_protected_operation_count": (
                self.selected_protected_operation_count
            ),
            "profile_conditioned_activity_selection": (
                self.profile_conditioned_activity_selection
            ),
            "activity_decision_slot": self.activity_decision_slot,
            "activity_decision_stochastic": self.activity_decision_stochastic,
            "activity_configuration_version": self.activity_configuration_version,
            "activity_calibration_status": self.activity_calibration_status,
            "activity_exclusion_reasons": list(self.activity_exclusion_reasons),
        }
