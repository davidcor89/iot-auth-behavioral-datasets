"""4C.4 activity applicability, inverse-CDF, composition, and boundary tests."""

from __future__ import annotations

import math
import random
import unittest

from simulator.config.activity_planning import (
    ACTIVITY_MAPPING_VERSION,
    ActivityPlanningConfiguration,
)
from simulator.config.device_profile_population import PopulationConfiguration
from simulator.data.dataset_contract import FIELD_MANIFEST_VERSION
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.engines.event_engine import EventEngine
from simulator.device_profiles.population import build_profile_population
from simulator.event_model import EventType
from simulator.planning.models import (
    ActivityApplicability,
    ActivityPlacement,
    ConstructionRole,
    LifecycleDepth,
    LifecycleSelection,
    RecoveryChoice,
    RecoverySelection,
    RenewalChoice,
    RenewalSelection,
    TraceConstructionRequest,
    TracePlanningError,
    TracePlanningErrorReason,
)
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
    activity_count_from_variate,
    adapt_trace_plan,
    reconstruct_profile_conditioned_activity,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    reconstruct_profile_conditioned_renewal,
    select_profile_conditioned_activity,
)
from simulator.planning.randomness import (
    PLANNING_DECISION_SLOT_ORDER_VERSION,
    PlanningDecisionSlot,
    derive_planning_randomness,
)
from simulator.planning.repertoire import (
    ACTIVITY_REPERTOIRE_VERSION,
    compose_ordinary_lifecycle,
    compose_ordinary_trace,
)
from simulator.provenance import FEATURE_SCHEMA_VERSION
from simulator.security_context import PersistentDeviceContext


def request(key: str = "trace") -> TraceConstructionRequest:
    return TraceConstructionRequest.profile_conditioned_lifecycle(
        trace_construction_key=key, scenario_id=f"scenario-{key}",
    )


def lifecycle_selection(depth: LifecycleDepth) -> LifecycleSelection:
    return LifecycleSelection(
        applicable_lifecycle_depths=tuple(LifecycleDepth),
        normalized_weights=tuple((item, 1 / 3) for item in LifecycleDepth),
        selected_lifecycle_depth=depth,
        random_variate=0.25,
    )


def recovery_selection(choice: RecoveryChoice) -> RecoverySelection:
    return RecoverySelection(
        applicable_recovery_choices=(RecoveryChoice.DIRECT, RecoveryChoice.ONE_RETRY),
        selected_recovery_choice=choice,
        decision_slot=2,
        stochastic_decision_used=True,
    )


def renewal_selection(choice: RenewalChoice) -> RenewalSelection:
    return RenewalSelection(
        applicable_renewal_choices=(RenewalChoice.NO_RENEWAL, RenewalChoice.RENEWAL),
        selected_renewal_choice=choice,
        decision_slot=3,
        stochastic_decision_used=True,
    )


def planning_randomness(key: str = "trace"):
    return derive_planning_randomness(
        planning_root_seed=4102,
        population_device_key="population-device-000",
        trace_construction_key=key,
    )


class FixedSlotRandomness:
    def __init__(self, value: float):
        self.value = value
        self.requested_slots: list[int] = []
        self.reference = "sha256:" + "0" * 64
        self.derivation_version = "4c-inf-planning-sha256-v1"
        self.namespace = "device-aware-trace-planning"

    def variate_at(self, slot: int) -> float:
        self.requested_slots.append(int(slot))
        if int(slot) != 4:
            raise AssertionError("activity may read only fixed slot 4")
        return self.value


def applicable_activity() -> ActivityApplicability:
    return ActivityApplicability(
        lifecycle_depth=LifecycleDepth.ACCESS,
        recovery_choice=RecoveryChoice.DIRECT,
        renewal_choice=RenewalChoice.NO_RENEWAL,
        applicable=True,
        structural_minimum=1,
        supported_minimum=1,
        supported_maximum=4,
        placement=ActivityPlacement.PRE_RENEWAL,
        supported_repertoire_version=ACTIVITY_REPERTOIRE_VERSION,
    )


class ActivityMechanismTests(unittest.TestCase):
    def test_endpoints_and_inverse_cdf_half_open_boundaries(self):
        self.assertEqual(1, activity_count_from_variate(0.0, 0.0))
        self.assertEqual(1, activity_count_from_variate(0.0, math.nextafter(1.0, 0.0)))
        self.assertEqual(4, activity_count_from_variate(1.0, 0.0))
        self.assertEqual(4, activity_count_from_variate(1.0, math.nextafter(1.0, 0.0)))

        # For A=.5 the X CDF boundaries are .125, .5, and .875.
        fixtures = (
            (math.nextafter(0.125, 0.0), 1),
            (0.125, 2),
            (0.25, 2),
            (0.5, 3),
            (0.75, 3),
            (0.875, 4),
            (math.nextafter(1.0, 0.0), 4),
        )
        for draw, expected in fixtures:
            with self.subTest(draw=draw):
                self.assertEqual(expected, activity_count_from_variate(0.5, draw))

    def test_activity_uses_only_slot_four(self):
        source = FixedSlotRandomness(0.5)
        result = select_profile_conditioned_activity(0.5, applicable_activity(), source)
        self.assertEqual(3, result.selected_protected_operation_count)
        self.assertEqual([4], source.requested_slots)
        self.assertEqual(4, result.decision_slot)
        self.assertTrue(result.stochastic_decision_used)

    def test_malformed_applicable_activity_is_rejected_without_coercion(self):
        malformed = (True, False, -0.01, 1.01, math.nan, math.inf, -math.inf, None, "0.5")
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(TracePlanningError) as raised:
                select_profile_conditioned_activity(
                    value, applicable_activity(), FixedSlotRandomness(0.5),
                )
            self.assertIs(
                TracePlanningErrorReason.MALFORMED_ACTIVITY_TENDENCY,
                raised.exception.reason,
            )

    def test_structural_zero_is_applicability_first_and_does_not_read_slot(self):
        inapplicable = ActivityApplicability(
            lifecycle_depth=LifecycleDepth.AUTH,
            recovery_choice=RecoveryChoice.DIRECT,
            renewal_choice=RenewalChoice.NO_RENEWAL,
            applicable=False,
            structural_minimum=0,
            supported_minimum=0,
            supported_maximum=0,
            placement=ActivityPlacement.NONE,
            supported_repertoire_version=ACTIVITY_REPERTOIRE_VERSION,
            exclusion_reasons=("lifecycle_has_no_protected_activity",),
        )
        for value in (None, True, -1, math.nan, "missing"):
            source = FixedSlotRandomness(0.5)
            result = select_profile_conditioned_activity(value, inapplicable, source)
            self.assertEqual(0, result.selected_protected_operation_count)
            self.assertFalse(result.stochastic_decision_used)
            self.assertEqual([], source.requested_slots)


class ActivityApplicabilityAndCompositionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = ApplicabilityResolver()
        self.context = PersistentDeviceContext(device_id="device")

    def _applicability(
        self,
        depth: LifecycleDepth,
        recovery: RecoveryChoice = RecoveryChoice.DIRECT,
        renewal: RenewalChoice = RenewalChoice.NO_RENEWAL,
    ) -> ActivityApplicability:
        return self.resolver.resolve_activity(
            self.context,
            ConstructionRole.ORDINARY,
            depth,
            recovery,
            renewal,
            compose_ordinary_lifecycle(depth, recovery),
        )

    def test_applicability_is_independent_of_a(self):
        for depth in LifecycleDepth:
            before = self._applicability(depth)
            for _activity in (0.0, 0.2, 0.8, 1.0):
                self.assertEqual(before, self._applicability(depth))
            if depth is LifecycleDepth.ACCESS:
                self.assertTrue(before.applicable)
                self.assertEqual((1, 1, 4), (
                    before.structural_minimum,
                    before.supported_minimum,
                    before.supported_maximum,
                ))
                self.assertIs(ActivityPlacement.PRE_RENEWAL, before.placement)
            else:
                self.assertFalse(before.applicable)
                self.assertEqual((0, 0, 0), (
                    before.structural_minimum,
                    before.supported_minimum,
                    before.supported_maximum,
                ))

    def _plan(
        self,
        recovery: RecoveryChoice,
        renewal: RenewalChoice,
        count: int,
        key: str,
    ):
        # Pick an A/draw pair that deterministically realizes the requested count.
        draw_by_count = {1: 0.0, 2: 0.2, 3: 0.6, 4: 0.9}
        source = FixedSlotRandomness(draw_by_count[count])
        return DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
            request(key),
            lifecycle_selection(LifecycleDepth.ACCESS),
            recovery_selection(recovery),
            renewal_selection(renewal),
            self._applicability(LifecycleDepth.ACCESS, recovery, renewal),
            0.5,
            source,
        )

    def test_all_counts_compose_for_direct_retry_and_renewal_matrix(self):
        for recovery in RecoveryChoice:
            for renewal in RenewalChoice:
                for count in range(1, 5):
                    key = f"{recovery.value}-{renewal.value}-{count}"
                    with self.subTest(key=key):
                        plan = self._plan(recovery, renewal, count, key)
                        self.assertEqual(count, plan.selected_protected_operation_count)
                        self.assertEqual(
                            compose_ordinary_trace(
                                LifecycleDepth.ACCESS, recovery, renewal, count,
                            ),
                            plan.event_steps,
                        )
                        random.seed(5200)
                        pair = EventEngine(
                            device_id=key,
                            gateway_id="gateway",
                            auth_server_id="auth",
                            broker_id="broker",
                            source_ip="10.0.0.1",
                            battery_level=80,
                            firmware_version="1",
                            persistent_context=PersistentDeviceContext(device_id=key),
                            start_time=1000.0,
                        ).execute(adapt_trace_plan(plan))
                        events, trace = pair
                        bundle = build_synchronized_bundle([pair])
                        requests = [event for event in events
                                    if event.event_type is EventType.ACCESS_REQUEST]
                        grants = [event for event in events
                                  if event.event_type is EventType.ACCESS_GRANTED]
                        self.assertEqual(count, len(requests))
                        self.assertEqual(count, len(grants))
                        self.assertEqual(count, len(set(trace.access_request_ids)))
                        self.assertEqual(count, len(trace.authorization_decisions))
                        self.assertEqual(count, len(trace.resource_operation_outcomes))
                        self.assertEqual(("granted",) * count,
                                         trace.authorization_decisions)
                        self.assertEqual(("success",) * count,
                                         trace.resource_operation_outcomes)
                        self.assertEqual("validated_normal",
                            bundle["validation"]["trace_ground_truth_statuses"][trace.trace_id])
                        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
                        self.assertFalse(trace.observable_anomaly)
                        if renewal is RenewalChoice.RENEWAL:
                            renewal_index = next(i for i, event in enumerate(events)
                                if event.event_type is EventType.RENEWAL_REQUEST)
                            self.assertTrue(all(events.index(event) < renewal_index
                                                for event in requests))
                            self.assertFalse(any(
                                event.event_type is EventType.ACCESS_REQUEST
                                for event in events[renewal_index:]
                            ))

    def test_provenance_and_information_boundary(self):
        plan = self._plan(
            RecoveryChoice.DIRECT, RenewalChoice.RENEWAL, 3, "provenance",
        )
        record = plan.provenance_record()
        self.assertEqual("4c5-trace-planning-provenance-v6",
                         record["planning_provenance_schema_version"])
        self.assertEqual(PROFILE_CONDITIONED_ACTIVITY_PLANNER_VERSION,
                         record["planner_version"])
        self.assertTrue(record["activity_applicable"])
        self.assertEqual(1, record["activity_structural_minimum"])
        self.assertEqual([1, 4], record["activity_supported_range"])
        self.assertEqual("PRE_RENEWAL", record["activity_placement"])
        self.assertEqual(ACTIVITY_REPERTOIRE_VERSION,
                         record["supported_activity_repertoire_version"])
        self.assertEqual(ACTIVITY_MAPPING_VERSION, record["activity_mapping_version"])
        self.assertEqual("4c4-controlled-activity-planning-v1",
                         record["activity_configuration_version"])
        self.assertEqual("synthetic_controlled_uncalibrated",
                         record["activity_calibration_status"])
        self.assertEqual(3, record["selected_protected_operation_count"])
        self.assertEqual(4, record["activity_decision_slot"])
        self.assertTrue(record["activity_decision_stochastic"])
        self.assertEqual(PLANNING_DECISION_SLOT_ORDER_VERSION,
                         record["planning_decision_slot_order_version"])
        for forbidden in (
            "activity_tendency", "activity_variate", "behavioral_profile_id",
            "population_device_key", "message_rate", "packet_rate",
        ):
            self.assertNotIn(forbidden, record)
        self.assertEqual("c1.6-development-derived-schema", FEATURE_SCHEMA_VERSION)
        self.assertEqual("sync-v1.1.0+4c4.activity-field-manifest-v5",
                         FIELD_MANIFEST_VERSION)

    def test_slot_four_does_not_perturb_slots_one_to_three(self):
        source = planning_randomness("fixed-slots")
        stream = source.new_rng()
        draws = [stream.random() for _ in range(4)]
        for slot, expected in enumerate(draws, start=1):
            self.assertEqual(expected, source.variate_at(slot))
        before = tuple(source.variate_at(slot) for slot in (1, 2, 3))
        reconstruct_profile_conditioned_activity(
            0.5, applicable_activity(), source,
        )
        after = tuple(source.variate_at(slot) for slot in (1, 2, 3))
        self.assertEqual(before, after)
        self.assertEqual(4, int(PlanningDecisionSlot.ACTIVITY))

    def test_192_trace_campaign_preserves_lrn_and_reconstructs_activity(self):
        population = build_profile_population(
            [(f"campaign-device-{index}", f"campaign-population-{index:03d}")
             for index in range(8)],
            PopulationConfiguration(population_seed=4101),
        )
        resolver = ApplicabilityResolver()
        preserved = {"L": 0, "R": 0, "N": 0}
        realized_by_device: list[tuple[float, set[int]]] = []
        activities_by_count: dict[int, set[float]] = {}
        for assignment in population.assignments:
            profile = assignment.profile
            context = PersistentDeviceContext(device_id=assignment.device_id)
            realized: set[int] = set()
            for index in range(24):
                key = f"{assignment.population_device_key}-trace-{index:03d}"
                source = derive_planning_randomness(
                    planning_root_seed=4102,
                    population_device_key=assignment.population_device_key,
                    trace_construction_key=key,
                )
                lifecycle_applicable = resolver.resolve(
                    context, ConstructionRole.ORDINARY,
                )
                lifecycle = reconstruct_profile_conditioned_lifecycle(
                    profile.lifecycle, lifecycle_applicable, source,
                )
                recovery_applicable = resolver.resolve_recovery(
                    context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                )
                recovery = reconstruct_profile_conditioned_recovery(
                    profile.recovery, recovery_applicable, source,
                )
                base = compose_ordinary_lifecycle(
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                )
                renewal_applicable = resolver.resolve_renewal(
                    context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                    base,
                )
                renewal = reconstruct_profile_conditioned_renewal(
                    profile.renewal, renewal_applicable, source,
                )
                activity_applicable = resolver.resolve_activity(
                    context, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                    renewal.selected_renewal_choice,
                    base,
                )
                plan = DeviceAwareTracePlanner().plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
                    request(key), lifecycle, recovery, renewal,
                    activity_applicable, profile.activity, source,
                )
                self.assertIs(lifecycle.selected_lifecycle_depth, plan.lifecycle_depth)
                self.assertIs(recovery.selected_recovery_choice, plan.recovery_choice)
                self.assertIs(renewal.selected_renewal_choice, plan.renewal_choice)
                preserved["L"] += 1
                preserved["R"] += 1
                preserved["N"] += 1
                if lifecycle.selected_lifecycle_depth is LifecycleDepth.ACCESS:
                    # Independent inverse-CDF reconstruction, not the helper.
                    a = profile.activity
                    u = source.variate_at(4)
                    c0 = (1 - a) ** 3
                    c1 = c0 + 3 * a * (1 - a) ** 2
                    c2 = c1 + 3 * a ** 2 * (1 - a)
                    expected = 1 if u < c0 else 2 if u < c1 else 3 if u < c2 else 4
                    self.assertEqual(expected, plan.selected_protected_operation_count)
                    realized.add(expected)
                    activities_by_count.setdefault(expected, set()).add(profile.activity)
                else:
                    self.assertEqual(0, plan.selected_protected_operation_count)
            realized_by_device.append((profile.activity, realized))
        self.assertEqual({"L": 192, "R": 192, "N": 192}, preserved)
        self.assertTrue(any(0 < activity < 1 and len(counts) > 1
                            for activity, counts in realized_by_device))
        self.assertTrue(any(len(activities) > 1
                            for activities in activities_by_count.values()))


if __name__ == "__main__":
    unittest.main()
