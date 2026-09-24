"""Applicability-first ordinary planning with opt-in lifecycle tendency use."""

from __future__ import annotations

import math
import random

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.activity_planning import ActivityPlanningConfiguration
from simulator.device_profiles.profile import DeviceProfile, LifecycleWeights
from simulator.engines.scenario_engine import ScenarioSpec
from simulator.planning.models import (
    ApplicableChoices,
    ApplicableRecoveryChoices,
    ApplicableRenewalChoices,
    ActivityApplicability,
    ActivityPlacement,
    ActivityRealization,
    ConstructionRole,
    LifecycleDepth,
    LifecycleSelection,
    PlanningMode,
    RecoveryChoice,
    RecoverySelection,
    RenewalChoice,
    RenewalSelection,
    TraceConstructionRequest,
    TracePlan,
    TracePlanningError,
    TracePlanningErrorReason,
)
from simulator.planning.randomness import (
    PLANNING_DECISION_SLOT_ORDER_VERSION,
    PlanningDecisionSlot,
    PlanningRandomness,
)
from simulator.planning.repertoire import (
    AUTHENTICATION_RECOVERY_REPERTOIRE,
    LIFECYCLE_REPERTOIRE_VERSION,
    ORDINARY_LIFECYCLE_REPERTOIRE,
    RECOVERY_REPERTOIRE_VERSION,
    RENEWAL_EXTENSION_REPERTOIRE,
    RENEWAL_REPERTOIRE_VERSION,
    ACTIVITY_REPERTOIRE_VERSION,
    compose_ordinary_lifecycle,
    compose_ordinary_trace,
    recovery_template_reference,
    renewal_extension_reference,
)
from simulator.security_context import PersistentDeviceContext


LEGACY_COMPATIBILITY_PLANNER_VERSION = "4c-inf-deterministic-compatibility-v1"
PROFILE_CONDITIONED_PLANNER_VERSION = "4c1-masked-categorical-lifecycle-v1"
PROFILE_CONDITIONED_RECOVERY_PLANNER_VERSION = (
    "4c2-lifecycle-recovery-fixed-slots-v1"
)
PROFILE_CONDITIONED_RENEWAL_PLANNER_VERSION = (
    "4c3-lifecycle-recovery-renewal-fixed-slots-v1"
)
PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION = (
    "4c4-lifecycle-recovery-renewal-activity-fixed-slots-v1"
)


_LIFECYCLE_WEIGHT_ATTRIBUTES = {
    LifecycleDepth.AUTH: "auth",
    LifecycleDepth.SESSION: "session",
    LifecycleDepth.ACCESS: "access",
}


def normalized_applicable_lifecycle_weights(
    lifecycle_weights: LifecycleWeights,
    applicable: ApplicableChoices,
) -> tuple[tuple[LifecycleDepth, float], ...]:
    """Mask non-applicable depths and normalize only valid applicable mass."""
    if not applicable.lifecycle_depths:
        detail = ", ".join(applicable.exclusion_reasons) or "no ordinary lifecycle available"
        raise TracePlanningError(
            TracePlanningErrorReason.NO_SUPPORTED_APPLICABLE_LIFECYCLE,
            detail,
        )
    if len(set(applicable.lifecycle_depths)) != len(applicable.lifecycle_depths):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
            "applicable lifecycle depths must be unique",
        )

    validated: dict[LifecycleDepth, float] = {}
    for depth, attribute in _LIFECYCLE_WEIGHT_ATTRIBUTES.items():
        if not hasattr(lifecycle_weights, attribute):
            raise TracePlanningError(
                TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
                f"missing lifecycle coordinate for {depth.value}",
            )
        value = getattr(lifecycle_weights, attribute)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise TracePlanningError(
                TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
                f"{depth.value} lifecycle weight must be finite and nonnegative",
            )
        validated[depth] = float(value)
    if not math.isclose(sum(validated.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
            "persistent lifecycle weights must sum to one",
        )

    masked = [(depth, validated[depth]) for depth in applicable.lifecycle_depths]

    total = sum(value for _, value in masked)
    if not math.isfinite(total):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_LIFECYCLE_WEIGHTS,
            "applicable lifecycle weight sum must be finite",
        )
    if total <= 0:
        raise TracePlanningError(
            TracePlanningErrorReason.ZERO_APPLICABLE_WEIGHT,
            "sum of applicable lifecycle weights is zero",
        )
    return tuple((depth, value / total) for depth, value in masked)


def select_profile_conditioned_lifecycle(
    lifecycle_weights: LifecycleWeights,
    applicable: ApplicableChoices,
    planning_rng: random.Random,
) -> LifecycleSelection:
    """Consume exactly one variate for an explicit cumulative categorical draw."""
    normalized = normalized_applicable_lifecycle_weights(lifecycle_weights, applicable)
    draw = planning_rng.random()
    if not math.isfinite(draw) or not 0 <= draw < 1:
        raise ValueError("planning RNG must produce a finite variate in [0, 1)")
    cumulative = 0.0
    # Rounding can leave the cumulative sum infinitesimally below one. Use the
    # final positive-mass depth in that case; a zero-weight depth is never selected.
    selected = next(depth for depth, probability in reversed(normalized) if probability > 0)
    for depth, probability in normalized:
        cumulative += probability
        if draw < cumulative:
            selected = depth
            break
    return LifecycleSelection(
        applicable_lifecycle_depths=applicable.lifecycle_depths,
        normalized_weights=normalized,
        selected_lifecycle_depth=selected,
        random_variate=draw,
    )


def reconstruct_profile_conditioned_lifecycle(
    lifecycle_weights: LifecycleWeights,
    applicable: ApplicableChoices,
    planning_randomness: PlanningRandomness,
) -> LifecycleSelection:
    """Reproduce a planning decision without EventEngine execution."""
    return select_profile_conditioned_lifecycle(
        lifecycle_weights, applicable, planning_randomness.new_rng(),
    )


def _validated_recovery_tendency(recovery_tendency: float) -> float:
    if (
        isinstance(recovery_tendency, bool)
        or not isinstance(recovery_tendency, (int, float))
        or not math.isfinite(recovery_tendency)
        or not 0 <= recovery_tendency <= 1
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_RECOVERY_TENDENCY,
            "recovery tendency must be a finite numeric value in [0, 1]",
        )
    return float(recovery_tendency)


def select_profile_conditioned_recovery(
    recovery_tendency: float,
    applicable: ApplicableRecoveryChoices,
    planning_randomness: PlanningRandomness,
) -> RecoverySelection:
    """Use fixed slot 2 only when recovery applicability offers a choice."""
    tendency = _validated_recovery_tendency(recovery_tendency)
    choices = applicable.recovery_choices
    if not choices:
        detail = ", ".join(applicable.exclusion_reasons) or "no recovery choice available"
        raise TracePlanningError(
            TracePlanningErrorReason.NO_APPLICABLE_RECOVERY_CHOICE,
            detail,
        )
    if len(set(choices)) != len(choices) or any(
        choice not in AUTHENTICATION_RECOVERY_REPERTOIRE for choice in choices
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
            "recovery applicability contains duplicate or unsupported choices",
        )
    if len(choices) == 1:
        return RecoverySelection(
            applicable_recovery_choices=choices,
            selected_recovery_choice=choices[0],
            decision_slot=int(PlanningDecisionSlot.RECOVERY),
            stochastic_decision_used=False,
        )
    if set(choices) != {RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY}:
        raise TracePlanningError(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
            "4C.2 supports exactly DIRECT and ONE_RETRY when recovery is stochastic",
        )
    draw = planning_randomness.variate_at(PlanningDecisionSlot.RECOVERY)
    selected = (
        RecoveryChoice.ONE_RETRY
        if draw < tendency
        else RecoveryChoice.DIRECT
    )
    return RecoverySelection(
        applicable_recovery_choices=choices,
        selected_recovery_choice=selected,
        decision_slot=int(PlanningDecisionSlot.RECOVERY),
        stochastic_decision_used=True,
    )


def reconstruct_profile_conditioned_recovery(
    recovery_tendency: float,
    applicable: ApplicableRecoveryChoices,
    planning_randomness: PlanningRandomness,
) -> RecoverySelection:
    """Reconstruct recovery independently of EventEngine realization."""
    return select_profile_conditioned_recovery(
        recovery_tendency, applicable, planning_randomness,
    )


def _validated_renewal_tendency(renewal_tendency: float) -> float:
    if (
        isinstance(renewal_tendency, bool)
        or not isinstance(renewal_tendency, (int, float))
        or not math.isfinite(renewal_tendency)
        or not 0 <= renewal_tendency <= 1
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_RENEWAL_TENDENCY,
            "renewal tendency must be a finite numeric value in [0, 1]",
        )
    return float(renewal_tendency)


def select_profile_conditioned_renewal(
    renewal_tendency: float,
    applicable: ApplicableRenewalChoices,
    planning_randomness: PlanningRandomness,
) -> RenewalSelection:
    """Use fixed slot 3 only when the approved continuation is applicable."""
    tendency = _validated_renewal_tendency(renewal_tendency)
    choices = applicable.renewal_choices
    if not choices:
        detail = ", ".join(applicable.exclusion_reasons) or "no renewal choice available"
        raise TracePlanningError(
            TracePlanningErrorReason.NO_APPLICABLE_RENEWAL_CHOICE,
            detail,
        )
    if len(set(choices)) != len(choices) or any(
        choice not in RENEWAL_EXTENSION_REPERTOIRE for choice in choices
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION,
            "renewal applicability contains duplicate or unsupported choices",
        )
    if len(choices) == 1:
        if choices != (RenewalChoice.NO_RENEWAL,):
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION,
                "singleton renewal applicability must be NO_RENEWAL",
            )
        return RenewalSelection(
            applicable_renewal_choices=choices,
            selected_renewal_choice=RenewalChoice.NO_RENEWAL,
            decision_slot=int(PlanningDecisionSlot.RENEWAL),
            stochastic_decision_used=False,
        )
    if set(choices) != {RenewalChoice.NO_RENEWAL, RenewalChoice.RENEWAL}:
        raise TracePlanningError(
            TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION,
            "4C.3 supports exactly NO_RENEWAL and RENEWAL when stochastic",
        )
    draw = planning_randomness.variate_at(PlanningDecisionSlot.RENEWAL)
    selected = (
        RenewalChoice.RENEWAL
        if draw < tendency
        else RenewalChoice.NO_RENEWAL
    )
    return RenewalSelection(
        applicable_renewal_choices=choices,
        selected_renewal_choice=selected,
        decision_slot=int(PlanningDecisionSlot.RENEWAL),
        stochastic_decision_used=True,
    )


def reconstruct_profile_conditioned_renewal(
    renewal_tendency: float,
    applicable: ApplicableRenewalChoices,
    planning_randomness: PlanningRandomness,
) -> RenewalSelection:
    """Reconstruct renewal independently of EventEngine realization."""
    return select_profile_conditioned_renewal(
        renewal_tendency, applicable, planning_randomness,
    )


def _validated_activity_tendency(activity_tendency: float) -> float:
    if (
        isinstance(activity_tendency, bool)
        or not isinstance(activity_tendency, (int, float))
        or not math.isfinite(activity_tendency)
        or not 0 <= activity_tendency <= 1
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.MALFORMED_ACTIVITY_TENDENCY,
            "activity tendency must be a finite numeric value in [0, 1]",
        )
    return float(activity_tendency)


def activity_count_from_variate(activity_tendency: float, variate: float) -> int:
    """Realize 1 + Binomial(3, A) with half-open inverse-CDF intervals."""
    tendency = _validated_activity_tendency(activity_tendency)
    if (
        isinstance(variate, bool)
        or not isinstance(variate, (int, float))
        or not math.isfinite(variate)
        or not 0 <= variate < 1
    ):
        raise ValueError("activity planning variate must be finite and in [0, 1)")
    u = float(variate)
    complement = 1.0 - tendency
    cumulative_zero = complement ** 3
    cumulative_one = cumulative_zero + 3.0 * tendency * complement ** 2
    cumulative_two = cumulative_one + 3.0 * tendency ** 2 * complement
    if u < cumulative_zero:
        return 1
    if u < cumulative_one:
        return 2
    if u < cumulative_two:
        return 3
    return 4


def select_profile_conditioned_activity(
    activity_tendency: float,
    applicable: ActivityApplicability,
    planning_randomness: PlanningRandomness,
    configuration: ActivityPlanningConfiguration | None = None,
) -> ActivityRealization:
    """Resolve structural zero or use fixed slot 4 for supported ACCESS activity."""
    config = configuration or ActivityPlanningConfiguration()
    if not applicable.applicable:
        if (
            applicable.structural_minimum != 0
            or applicable.supported_minimum != 0
            or applicable.supported_maximum != 0
            or applicable.placement is not ActivityPlacement.NONE
        ):
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_ACTIVITY_COMBINATION,
                "inapplicable activity must be the structural zero realization",
            )
        return ActivityRealization(
            selected_protected_operation_count=0,
            decision_slot=int(PlanningDecisionSlot.ACTIVITY),
            stochastic_decision_used=False,
            mapping_version=config.mapping_version,
        )
    if (
        applicable.structural_minimum != 1
        or applicable.supported_minimum != config.supported_minimum
        or applicable.supported_maximum != config.supported_maximum
        or applicable.placement is not ActivityPlacement.PRE_RENEWAL
        or applicable.supported_repertoire_version != ACTIVITY_REPERTOIRE_VERSION
    ):
        raise TracePlanningError(
            TracePlanningErrorReason.UNSUPPORTED_ACTIVITY_COMBINATION,
            "activity applicability is outside the 4C.4 clean-operation repertoire",
        )
    tendency = _validated_activity_tendency(activity_tendency)
    draw = planning_randomness.variate_at(PlanningDecisionSlot.ACTIVITY)
    return ActivityRealization(
        selected_protected_operation_count=activity_count_from_variate(tendency, draw),
        decision_slot=int(PlanningDecisionSlot.ACTIVITY),
        stochastic_decision_used=0 < tendency < 1,
        mapping_version=config.mapping_version,
    )


def reconstruct_profile_conditioned_activity(
    activity_tendency: float,
    applicable: ActivityApplicability,
    planning_randomness: PlanningRandomness,
    configuration: ActivityPlanningConfiguration | None = None,
) -> ActivityRealization:
    """Reconstruct activity independently of EventEngine realization."""
    return select_profile_conditioned_activity(
        activity_tendency, applicable, planning_randomness, configuration,
    )


class ApplicabilityResolver:
    """Pure lifecycle applicability; profile values and RNG are out of scope."""

    def resolve(
        self,
        context: PersistentDeviceContext,
        construction_role: ConstructionRole,
    ) -> ApplicableChoices:
        if construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.INF ordinary planner does not accept {construction_role.value}",
            )
        exclusions = []
        if context.revoked:
            exclusions.append("persistent_device_revoked")
        if context.blocked:
            exclusions.append("persistent_device_blocked")
        choices = () if exclusions else tuple(ORDINARY_LIFECYCLE_REPERTOIRE)
        return ApplicableChoices(
            lifecycle_depths=choices,
            supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
            exclusion_reasons=tuple(exclusions),
        )

    def resolve_recovery(
        self,
        context: PersistentDeviceContext,
        construction_role: ConstructionRole,
        lifecycle_depth: LifecycleDepth,
    ) -> ApplicableRecoveryChoices:
        """Resolve recovery only after lifecycle, without profile values or RNG."""
        if construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.2 recovery planner does not accept {construction_role.value}",
            )
        exclusions = []
        if context.revoked:
            exclusions.append("persistent_device_revoked")
        if context.blocked:
            exclusions.append("persistent_device_blocked")
        if lifecycle_depth not in ORDINARY_LIFECYCLE_REPERTOIRE:
            exclusions.append("unsupported_lifecycle_depth")
        choices = (
            ()
            if exclusions
            else (RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY)
        )
        return ApplicableRecoveryChoices(
            lifecycle_depth=lifecycle_depth,
            recovery_choices=choices,
            supported_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
            exclusion_reasons=tuple(exclusions),
        )

    def resolve_renewal(
        self,
        context: PersistentDeviceContext,
        construction_role: ConstructionRole,
        lifecycle_depth: LifecycleDepth,
        recovery_choice: RecoveryChoice,
        base_event_steps: tuple,
    ) -> ApplicableRenewalChoices:
        """Resolve continuation after L/R without reading N or planning RNG."""
        if construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.3 renewal planner does not accept {construction_role.value}",
            )
        exclusions = []
        if context.revoked:
            exclusions.append("persistent_device_revoked")
        if context.blocked:
            exclusions.append("persistent_device_blocked")
        try:
            supported_base = compose_ordinary_lifecycle(
                lifecycle_depth, recovery_choice,
            )
        except (KeyError, TypeError):
            supported_base = ()
            exclusions.append("unsupported_lifecycle_recovery_base")
        if tuple(base_event_steps) != supported_base:
            exclusions.append("unsupported_base_plan")
        if exclusions:
            choices = ()
        elif lifecycle_depth is LifecycleDepth.ACCESS:
            choices = (RenewalChoice.NO_RENEWAL, RenewalChoice.RENEWAL)
        else:
            choices = (RenewalChoice.NO_RENEWAL,)
        return ApplicableRenewalChoices(
            lifecycle_depth=lifecycle_depth,
            recovery_choice=recovery_choice,
            renewal_choices=choices,
            supported_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
            exclusion_reasons=tuple(exclusions),
        )

    def resolve_activity(
        self,
        context: PersistentDeviceContext,
        construction_role: ConstructionRole,
        lifecycle_depth: LifecycleDepth,
        recovery_choice: RecoveryChoice,
        renewal_choice: RenewalChoice,
        base_event_steps: tuple,
        configuration: ActivityPlanningConfiguration | None = None,
    ) -> ActivityApplicability:
        """Resolve structural activity support after L/R/N without reading A."""
        config = configuration or ActivityPlanningConfiguration()
        if construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.4 activity planner does not accept {construction_role.value}",
            )
        exclusions = []
        if context.revoked:
            exclusions.append("persistent_device_revoked")
        if context.blocked:
            exclusions.append("persistent_device_blocked")
        try:
            supported_base = compose_ordinary_lifecycle(
                lifecycle_depth, recovery_choice,
            )
        except (KeyError, TypeError):
            supported_base = ()
            exclusions.append("unsupported_lifecycle_recovery_base")
        if tuple(base_event_steps) != supported_base:
            exclusions.append("unsupported_base_plan")
        if renewal_choice not in RENEWAL_EXTENSION_REPERTOIRE:
            exclusions.append("unsupported_renewal_choice")
        if (
            renewal_choice is RenewalChoice.RENEWAL
            and lifecycle_depth is not LifecycleDepth.ACCESS
        ):
            exclusions.append("renewal_requires_access_lifecycle")
        if exclusions:
            raise TracePlanningError(
                TracePlanningErrorReason.NO_APPLICABLE_ACTIVITY_REALIZATION,
                ", ".join(exclusions),
            )
        if lifecycle_depth is LifecycleDepth.ACCESS:
            return ActivityApplicability(
                lifecycle_depth=lifecycle_depth,
                recovery_choice=recovery_choice,
                renewal_choice=renewal_choice,
                applicable=True,
                structural_minimum=1,
                supported_minimum=config.supported_minimum,
                supported_maximum=config.supported_maximum,
                placement=ActivityPlacement.PRE_RENEWAL,
                supported_repertoire_version=ACTIVITY_REPERTOIRE_VERSION,
            )
        return ActivityApplicability(
            lifecycle_depth=lifecycle_depth,
            recovery_choice=recovery_choice,
            renewal_choice=renewal_choice,
            applicable=False,
            structural_minimum=0,
            supported_minimum=0,
            supported_maximum=0,
            placement=ActivityPlacement.NONE,
            supported_repertoire_version=ACTIVITY_REPERTOIRE_VERSION,
            exclusion_reasons=("lifecycle_has_no_protected_activity",),
        )


class DeviceAwareTracePlanner:
    """Ordinary planner with separate compatibility and L-conditioned paths."""

    def plan(
        self,
        request: TraceConstructionRequest,
        applicable: ApplicableChoices,
        profile: DeviceProfile,
        planning_randomness: PlanningRandomness,
    ) -> TracePlan:
        if request.planning_mode is PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "profile-conditioned requests require lifecycle_weights boundary",
            )
        if request.construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.INF ordinary planner does not accept {request.construction_role.value}",
            )
        if not applicable.lifecycle_depths:
            detail = ", ".join(applicable.exclusion_reasons) or "no ordinary lifecycle available"
            raise TracePlanningError(
                TracePlanningErrorReason.NO_SUPPORTED_APPLICABLE_LIFECYCLE,
                detail,
            )
        depth = request.requested_lifecycle_depth
        if depth is None:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "compatibility planning requires a requested lifecycle depth",
            )
        template = ORDINARY_LIFECYCLE_REPERTOIRE.get(depth)
        if template is None or depth not in applicable.lifecycle_depths:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                f"requested lifecycle {depth.value} is not in applicable supported choices",
            )

        if request.planning_mode is PlanningMode.LEGACY_EQUIVALENT:
            intent = request.legacy_scenario_intent
            if intent is None:  # guarded by the request, retained as a safe boundary
                raise TracePlanningError(
                    TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                    "legacy scenario intent is missing",
                )
            event_steps = intent.normal_steps
            generation_profile = intent.generation_profile
            template_reference = "legacy-ordinary-scenario-compatibility-v1"
        else:
            event_steps = template.event_steps
            generation_profile = SYNCHRONIZED_GENERATION_PROFILE
            template_reference = template.template_reference

        # The profile is intentionally not inspected. The runner has already
        # resolved exactly one assignment for the selected device; the existing
        # privileged trace device_id joins to the separate DP.2 companion.
        _ = profile
        return TracePlan(
            trace_construction_key=request.trace_construction_key,
            scenario_id=request.scenario_id,
            construction_role=request.construction_role,
            lifecycle_depth=depth,
            completion_mode=request.completion_mode,
            planning_mode=request.planning_mode,
            supported_repertoire_version=applicable.supported_repertoire_version,
            template_reference=template_reference,
            event_steps=event_steps,
            generation_profile=generation_profile,
            planning_rng_reference=planning_randomness.reference,
            planning_rng_derivation_version=planning_randomness.derivation_version,
            planning_rng_namespace=planning_randomness.namespace,
            planner_version=LEGACY_COMPATIBILITY_PLANNER_VERSION,
            applicable_lifecycle_depths=applicable.lifecycle_depths,
            planning_draw_count=0,
            profile_conditioned_selection=False,
            recovery_choice=RecoveryChoice.DIRECT,
            supported_recovery_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
            applicable_recovery_choices=(RecoveryChoice.DIRECT,),
            recovery_template_reference="compatibility-direct-recovery-v1",
            profile_conditioned_recovery_selection=False,
            recovery_decision_slot=int(PlanningDecisionSlot.RECOVERY),
            recovery_decision_stochastic=False,
            renewal_choice=RenewalChoice.NO_RENEWAL,
            supported_renewal_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
            applicable_renewal_choices=(RenewalChoice.NO_RENEWAL,),
            renewal_extension_reference="compatibility-no-renewal-v1",
            profile_conditioned_renewal_selection=False,
            renewal_decision_slot=int(PlanningDecisionSlot.RENEWAL),
            renewal_decision_stochastic=False,
            base_lifecycle_endpoint=template.intentional_endpoint,
            final_planned_endpoint=event_steps[-1],
            planning_decision_slot_order_version=(
                PLANNING_DECISION_SLOT_ORDER_VERSION
            ),
        )

    def plan_profile_conditioned_lifecycle(
        self,
        request: TraceConstructionRequest,
        applicable: ApplicableChoices,
        lifecycle_weights: LifecycleWeights,
        planning_randomness: PlanningRandomness,
    ) -> TracePlan:
        """Select one applicable depth from L using exactly one isolated draw."""
        if request.construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.1 ordinary planner does not accept {request.construction_role.value}",
            )
        if request.planning_mode is not PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "L-conditioned planner requires profile-conditioned lifecycle mode",
            )
        selection = reconstruct_profile_conditioned_lifecycle(
            lifecycle_weights, applicable, planning_randomness,
        )
        depth = selection.selected_lifecycle_depth
        template = ORDINARY_LIFECYCLE_REPERTOIRE.get(depth)
        if template is None:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                f"selected lifecycle {depth.value} has no supported template",
            )
        return TracePlan(
            trace_construction_key=request.trace_construction_key,
            scenario_id=request.scenario_id,
            construction_role=request.construction_role,
            lifecycle_depth=depth,
            completion_mode=request.completion_mode,
            planning_mode=request.planning_mode,
            supported_repertoire_version=applicable.supported_repertoire_version,
            template_reference=template.template_reference,
            event_steps=template.event_steps,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            planning_rng_reference=planning_randomness.reference,
            planning_rng_derivation_version=planning_randomness.derivation_version,
            planning_rng_namespace=planning_randomness.namespace,
            planner_version=PROFILE_CONDITIONED_PLANNER_VERSION,
            applicable_lifecycle_depths=selection.applicable_lifecycle_depths,
            planning_draw_count=1,
            profile_conditioned_selection=True,
            recovery_choice=RecoveryChoice.DIRECT,
            supported_recovery_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
            applicable_recovery_choices=(RecoveryChoice.DIRECT,),
            recovery_template_reference="4c1-direct-recovery-compatibility-v1",
            profile_conditioned_recovery_selection=False,
            recovery_decision_slot=int(PlanningDecisionSlot.RECOVERY),
            recovery_decision_stochastic=False,
            renewal_choice=RenewalChoice.NO_RENEWAL,
            supported_renewal_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
            applicable_renewal_choices=(RenewalChoice.NO_RENEWAL,),
            renewal_extension_reference="4c1-no-renewal-compatibility-v1",
            profile_conditioned_renewal_selection=False,
            renewal_decision_slot=int(PlanningDecisionSlot.RENEWAL),
            renewal_decision_stochastic=False,
            base_lifecycle_endpoint=template.intentional_endpoint,
            final_planned_endpoint=template.intentional_endpoint,
            planning_decision_slot_order_version=(
                PLANNING_DECISION_SLOT_ORDER_VERSION
            ),
        )

    def plan_profile_conditioned_lifecycle_and_recovery(
        self,
        request: TraceConstructionRequest,
        lifecycle_selection: LifecycleSelection,
        recovery_applicable: ApplicableRecoveryChoices,
        recovery_tendency: float,
        planning_randomness: PlanningRandomness,
    ) -> TracePlan:
        """Compose slot-1 lifecycle with slot-2 R-conditioned recovery."""
        if request.construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.2 ordinary planner does not accept {request.construction_role.value}",
            )
        if request.planning_mode is not PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "L/R-conditioned planner requires profile-conditioned lifecycle mode",
            )
        depth = lifecycle_selection.selected_lifecycle_depth
        if recovery_applicable.lifecycle_depth is not depth:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
                "recovery applicability was resolved for a different lifecycle depth",
            )
        recovery = reconstruct_profile_conditioned_recovery(
            recovery_tendency, recovery_applicable, planning_randomness,
        )
        try:
            event_steps = compose_ordinary_lifecycle(
                depth, recovery.selected_recovery_choice,
            )
            template_reference = recovery_template_reference(
                depth, recovery.selected_recovery_choice,
            )
        except (KeyError, TypeError) as error:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RECOVERY_COMBINATION,
                f"unsupported {depth!r}/{recovery.selected_recovery_choice.value} composition",
            ) from error
        return TracePlan(
            trace_construction_key=request.trace_construction_key,
            scenario_id=request.scenario_id,
            construction_role=request.construction_role,
            lifecycle_depth=depth,
            completion_mode=request.completion_mode,
            planning_mode=request.planning_mode,
            supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
            template_reference=template_reference,
            event_steps=event_steps,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            planning_rng_reference=planning_randomness.reference,
            planning_rng_derivation_version=planning_randomness.derivation_version,
            planning_rng_namespace=planning_randomness.namespace,
            planner_version=PROFILE_CONDITIONED_RECOVERY_PLANNER_VERSION,
            applicable_lifecycle_depths=(
                lifecycle_selection.applicable_lifecycle_depths
            ),
            planning_draw_count=(
                2 if recovery.stochastic_decision_used else 1
            ),
            profile_conditioned_selection=True,
            recovery_choice=recovery.selected_recovery_choice,
            supported_recovery_repertoire_version=(
                recovery_applicable.supported_repertoire_version
            ),
            applicable_recovery_choices=(
                recovery.applicable_recovery_choices
            ),
            recovery_template_reference=template_reference,
            profile_conditioned_recovery_selection=(
                recovery.stochastic_decision_used
            ),
            recovery_decision_slot=recovery.decision_slot,
            recovery_decision_stochastic=recovery.stochastic_decision_used,
            renewal_choice=RenewalChoice.NO_RENEWAL,
            supported_renewal_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
            applicable_renewal_choices=(RenewalChoice.NO_RENEWAL,),
            renewal_extension_reference="4c2-no-renewal-compatibility-v1",
            profile_conditioned_renewal_selection=False,
            renewal_decision_slot=int(PlanningDecisionSlot.RENEWAL),
            renewal_decision_stochastic=False,
            base_lifecycle_endpoint=(
                ORDINARY_LIFECYCLE_REPERTOIRE[depth].intentional_endpoint
            ),
            final_planned_endpoint=event_steps[-1],
            planning_decision_slot_order_version=(
                PLANNING_DECISION_SLOT_ORDER_VERSION
            ),
        )

    def plan_profile_conditioned_lifecycle_recovery_and_renewal(
        self,
        request: TraceConstructionRequest,
        lifecycle_selection: LifecycleSelection,
        recovery_selection: RecoverySelection,
        renewal_applicable: ApplicableRenewalChoices,
        renewal_tendency: float,
        planning_randomness: PlanningRandomness,
    ) -> TracePlan:
        """Compose slots 1/2 with an applicability-gated slot-3 continuation."""
        if request.construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.3 ordinary planner does not accept {request.construction_role.value}",
            )
        if request.planning_mode is not PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "L/R/N-conditioned planner requires profile-conditioned lifecycle mode",
            )
        depth = lifecycle_selection.selected_lifecycle_depth
        recovery_choice = recovery_selection.selected_recovery_choice
        if (
            renewal_applicable.lifecycle_depth is not depth
            or renewal_applicable.recovery_choice is not recovery_choice
        ):
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION,
                "renewal applicability was resolved for a different base plan",
            )
        renewal = reconstruct_profile_conditioned_renewal(
            renewal_tendency, renewal_applicable, planning_randomness,
        )
        try:
            event_steps = compose_ordinary_trace(
                depth, recovery_choice, renewal.selected_renewal_choice,
            )
            template_reference = recovery_template_reference(depth, recovery_choice)
            extension_reference = renewal_extension_reference(
                renewal.selected_renewal_choice
            )
        except (KeyError, TypeError) as error:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_LIFECYCLE_RENEWAL_COMBINATION,
                f"unsupported {depth!r}/{recovery_choice.value}/"
                f"{renewal.selected_renewal_choice.value} composition",
            ) from error
        return TracePlan(
            trace_construction_key=request.trace_construction_key,
            scenario_id=request.scenario_id,
            construction_role=request.construction_role,
            lifecycle_depth=depth,
            completion_mode=request.completion_mode,
            planning_mode=request.planning_mode,
            supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
            template_reference=template_reference,
            event_steps=event_steps,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            planning_rng_reference=planning_randomness.reference,
            planning_rng_derivation_version=planning_randomness.derivation_version,
            planning_rng_namespace=planning_randomness.namespace,
            planner_version=PROFILE_CONDITIONED_RENEWAL_PLANNER_VERSION,
            applicable_lifecycle_depths=lifecycle_selection.applicable_lifecycle_depths,
            planning_draw_count=(
                int(PlanningDecisionSlot.RENEWAL)
                if renewal.stochastic_decision_used
                else int(PlanningDecisionSlot.RECOVERY)
            ),
            profile_conditioned_selection=True,
            recovery_choice=recovery_choice,
            supported_recovery_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
            applicable_recovery_choices=recovery_selection.applicable_recovery_choices,
            recovery_template_reference=template_reference,
            profile_conditioned_recovery_selection=(
                recovery_selection.stochastic_decision_used
            ),
            recovery_decision_slot=recovery_selection.decision_slot,
            recovery_decision_stochastic=recovery_selection.stochastic_decision_used,
            renewal_choice=renewal.selected_renewal_choice,
            supported_renewal_repertoire_version=(
                renewal_applicable.supported_repertoire_version
            ),
            applicable_renewal_choices=renewal.applicable_renewal_choices,
            renewal_extension_reference=extension_reference,
            profile_conditioned_renewal_selection=renewal.stochastic_decision_used,
            renewal_decision_slot=renewal.decision_slot,
            renewal_decision_stochastic=renewal.stochastic_decision_used,
            base_lifecycle_endpoint=(
                ORDINARY_LIFECYCLE_REPERTOIRE[depth].intentional_endpoint
            ),
            final_planned_endpoint=event_steps[-1],
            planning_decision_slot_order_version=PLANNING_DECISION_SLOT_ORDER_VERSION,
        )

    def plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
        self,
        request: TraceConstructionRequest,
        lifecycle_selection: LifecycleSelection,
        recovery_selection: RecoverySelection,
        renewal_selection: RenewalSelection,
        activity_applicable: ActivityApplicability,
        activity_tendency: float,
        planning_randomness: PlanningRandomness,
        activity_configuration: ActivityPlanningConfiguration | None = None,
    ) -> TracePlan:
        """Compose frozen L/R/N choices with slot-4 activity realization."""
        if request.construction_role is not ConstructionRole.ORDINARY:
            raise TracePlanningError(
                TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
                f"4C.4 ordinary planner does not accept {request.construction_role.value}",
            )
        if request.planning_mode is not PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_REQUESTED_LIFECYCLE,
                "L/R/N/A-conditioned planner requires profile-conditioned lifecycle mode",
            )
        depth = lifecycle_selection.selected_lifecycle_depth
        recovery_choice = recovery_selection.selected_recovery_choice
        renewal_choice = renewal_selection.selected_renewal_choice
        config = activity_configuration or ActivityPlanningConfiguration()
        activity = reconstruct_profile_conditioned_activity(
            activity_tendency,
            activity_applicable,
            planning_randomness,
            config,
        )
        expected_applicable = depth is LifecycleDepth.ACCESS
        if (
            activity_applicable.lifecycle_depth is not depth
            or activity_applicable.recovery_choice is not recovery_choice
            or activity_applicable.renewal_choice is not renewal_choice
            or activity_applicable.applicable is not expected_applicable
        ):
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_ACTIVITY_COMBINATION,
                "activity applicability does not match the selected lifecycle",
            )
        try:
            event_steps = compose_ordinary_trace(
                depth,
                recovery_choice,
                renewal_choice,
                activity.selected_protected_operation_count,
            )
            template_reference = recovery_template_reference(depth, recovery_choice)
            extension_reference = renewal_extension_reference(renewal_choice)
        except (KeyError, TypeError) as error:
            raise TracePlanningError(
                TracePlanningErrorReason.UNSUPPORTED_ACTIVITY_COMBINATION,
                f"unsupported {depth!r}/{recovery_choice.value}/"
                f"{renewal_choice.value}/activity-"
                f"{activity.selected_protected_operation_count} composition",
            ) from error
        return TracePlan(
            trace_construction_key=request.trace_construction_key,
            scenario_id=request.scenario_id,
            construction_role=request.construction_role,
            lifecycle_depth=depth,
            completion_mode=request.completion_mode,
            planning_mode=request.planning_mode,
            supported_repertoire_version=LIFECYCLE_REPERTOIRE_VERSION,
            template_reference=template_reference,
            event_steps=event_steps,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            planning_rng_reference=planning_randomness.reference,
            planning_rng_derivation_version=planning_randomness.derivation_version,
            planning_rng_namespace=planning_randomness.namespace,
            planner_version=PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
            applicable_lifecycle_depths=lifecycle_selection.applicable_lifecycle_depths,
            planning_draw_count=(
                int(PlanningDecisionSlot.ACTIVITY)
                if activity_applicable.applicable
                else (
                    int(PlanningDecisionSlot.RENEWAL)
                    if renewal_selection.stochastic_decision_used
                    else int(PlanningDecisionSlot.RECOVERY)
                )
            ),
            profile_conditioned_selection=True,
            recovery_choice=recovery_choice,
            supported_recovery_repertoire_version=RECOVERY_REPERTOIRE_VERSION,
            applicable_recovery_choices=recovery_selection.applicable_recovery_choices,
            recovery_template_reference=template_reference,
            profile_conditioned_recovery_selection=(
                recovery_selection.stochastic_decision_used
            ),
            recovery_decision_slot=recovery_selection.decision_slot,
            recovery_decision_stochastic=recovery_selection.stochastic_decision_used,
            renewal_choice=renewal_choice,
            supported_renewal_repertoire_version=RENEWAL_REPERTOIRE_VERSION,
            applicable_renewal_choices=renewal_selection.applicable_renewal_choices,
            renewal_extension_reference=extension_reference,
            profile_conditioned_renewal_selection=(
                renewal_selection.stochastic_decision_used
            ),
            renewal_decision_slot=renewal_selection.decision_slot,
            renewal_decision_stochastic=renewal_selection.stochastic_decision_used,
            base_lifecycle_endpoint=(
                ORDINARY_LIFECYCLE_REPERTOIRE[depth].intentional_endpoint
            ),
            final_planned_endpoint=event_steps[-1],
            planning_decision_slot_order_version=PLANNING_DECISION_SLOT_ORDER_VERSION,
            activity_applicable=activity_applicable.applicable,
            activity_structural_minimum=activity_applicable.structural_minimum,
            activity_supported_minimum=activity_applicable.supported_minimum,
            activity_supported_maximum=activity_applicable.supported_maximum,
            activity_placement=activity_applicable.placement,
            supported_activity_repertoire_version=(
                activity_applicable.supported_repertoire_version
            ),
            activity_mapping_version=activity.mapping_version,
            selected_protected_operation_count=(
                activity.selected_protected_operation_count
            ),
            profile_conditioned_activity_selection=activity_applicable.applicable,
            activity_decision_slot=activity.decision_slot,
            activity_decision_stochastic=activity.stochastic_decision_used,
            activity_configuration_version=(
                config.configuration_version
            ),
            activity_calibration_status=config.calibration_status,
            activity_exclusion_reasons=activity_applicable.exclusion_reasons,
        )


def adapt_trace_plan(plan: TracePlan) -> ScenarioSpec:
    """Narrow compatibility adapter; planning metadata does not enter events."""
    if plan.construction_role is not ConstructionRole.ORDINARY:
        raise TracePlanningError(
            TracePlanningErrorReason.INVALID_CONSTRUCTION_ROLE,
            "ScenarioSpec adapter accepts ordinary plans only",
        )
    return ScenarioSpec(
        scenario_id=plan.scenario_id,
        scenario_type="normal",
        is_anomaly=False,
        generation_profile=plan.generation_profile,
        normal_steps=list(plan.event_steps),
    )
