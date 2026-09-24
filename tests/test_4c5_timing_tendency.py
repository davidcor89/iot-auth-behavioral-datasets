"""4C.5 relation mapping, deterministic timing, envelope, and boundaries."""

from __future__ import annotations

import math
import random
import unittest
from collections import defaultdict
from unittest.mock import patch

from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE
from simulator.config.population_timing import (
    DEFAULT_GAUSSIAN_BASELINES,
    TIMING_CALIBRATION_STATUS,
    GaussianDelayBaseline,
    PopulationTimingConfiguration,
)
from simulator.config.settings import cfg
from simulator.config.device_profile_population import PopulationConfiguration
from simulator.data.synchronized_views import build_synchronized_bundle
from simulator.device_profiles.population import build_profile_population
from simulator.device_profiles.profile import (
    RelativeTimingTendency,
    TimingTendencies,
)
from simulator.engines.event_engine import DELAY_PROFILE, EventEngine
from simulator.engines.scenario_engine import ScenarioSpec
from simulator.event_model import EventResult, EventType
from simulator.planning.models import (
    ConstructionRole,
    LifecycleDepth,
    RecoveryChoice,
    RenewalChoice,
    TraceConstructionRequest,
)
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    adapt_trace_plan,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    reconstruct_profile_conditioned_renewal,
)
from simulator.planning.randomness import (
    PLANNING_DECISION_SLOT_ORDER_VERSION,
    PlanningDecisionSlot,
    derive_planning_randomness,
)
from simulator.planning.repertoire import compose_ordinary_trace
from simulator.planning.repertoire import compose_ordinary_lifecycle
from simulator.runner import run_simulation_with_population
from simulator.security_context import PersistentDeviceContext
from simulator.timing.realizer import (
    EmptyTimingEnvelopeError,
    TimingFamily,
    TimingRealizer,
    derive_relation_randomness_reference,
    map_timing_relations,
    normal_variate_from_reference,
    signed_uniform_from_reference,
)


def tendencies(
    *,
    auth=(1.0, 1.0),
    transition=(1.0, 1.0),
    access=(1.0, 1.0),
    renewal=(1.0, 1.0),
) -> TimingTendencies:
    return TimingTendencies(
        auth=RelativeTimingTendency(*auth),
        transition=RelativeTimingTendency(*transition),
        access=RelativeTimingTendency(*access),
        renewal=RelativeTimingTendency(*renewal),
    )


def planning_reference(key: str = "trace") -> str:
    return derive_planning_randomness(
        planning_root_seed=4102,
        population_device_key="population-device-000",
        trace_construction_key=key,
    ).reference


def realizer(
    steps,
    *,
    profile: TimingTendencies | None = None,
    configuration: PopulationTimingConfiguration | None = None,
    key: str = "trace",
) -> TimingRealizer:
    return TimingRealizer(
        configuration=configuration or PopulationTimingConfiguration(),
        timing_profile=profile or tendencies(),
        planning_rng_reference=planning_reference(key),
        event_steps=steps,
    )


def baseline_override(event_type: EventType, center: float, dispersion: float = 0.0):
    values = tuple(
        GaussianDelayBaseline(event_type, center, dispersion)
        if item.event_type is event_type else item
        for item in DEFAULT_GAUSSIAN_BASELINES
    )
    return PopulationTimingConfiguration(gaussian_baselines=values)


class TimingMappingTests(unittest.TestCase):
    def test_structural_mapping_and_renewal_override(self):
        steps = compose_ordinary_trace(
            LifecycleDepth.ACCESS,
            RecoveryChoice.ONE_RETRY,
            RenewalChoice.RENEWAL,
            2,
        )
        mapped = map_timing_relations(steps)
        self.assertEqual(steps, tuple(item.event_type for item in mapped))
        by_index = {item.step_index: item for item in mapped}
        renewal_index = steps.index(EventType.RENEWAL_REQUEST)
        self.assertTrue(all(
            by_index[index].family is TimingFamily.RENEWAL
            for index in range(renewal_index, len(steps))
        ))
        access_indices = [
            index for index, event in enumerate(steps)
            if event in {EventType.ACCESS_REQUEST, EventType.ACCESS_GRANTED}
        ]
        self.assertEqual(
            {TimingFamily.ACCESS},
            {by_index[index].family for index in access_indices},
        )
        retry_index = steps.index(EventType.RETRY)
        self.assertIs(TimingFamily.AUTH, by_index[retry_index].family)
        first_token = steps.index(EventType.TOKEN_ISSUED)
        self.assertIs(TimingFamily.TRANSITION, by_index[first_token].family)

    def test_family_applicability_follows_structure(self):
        fixtures = (
            (LifecycleDepth.AUTH, RenewalChoice.NO_RENEWAL,
             {TimingFamily.TRANSITION, TimingFamily.AUTH}),
            (LifecycleDepth.SESSION, RenewalChoice.NO_RENEWAL,
             {TimingFamily.TRANSITION, TimingFamily.AUTH}),
            (LifecycleDepth.ACCESS, RenewalChoice.NO_RENEWAL,
             {TimingFamily.TRANSITION, TimingFamily.AUTH, TimingFamily.ACCESS}),
            (LifecycleDepth.ACCESS, RenewalChoice.RENEWAL, set(TimingFamily)),
        )
        for depth, renewal, expected in fixtures:
            with self.subTest(depth=depth, renewal=renewal):
                steps = compose_ordinary_trace(
                    depth, RecoveryChoice.DIRECT, renewal,
                    1 if depth is LifecycleDepth.ACCESS else 0,
                )
                self.assertEqual(expected, {item.family for item in map_timing_relations(steps)})

    def test_segment_local_relation_identities_are_append_stable(self):
        one = map_timing_relations(compose_ordinary_trace(
            LifecycleDepth.ACCESS, RecoveryChoice.DIRECT,
            RenewalChoice.RENEWAL, 1,
        ))
        four = map_timing_relations(compose_ordinary_trace(
            LifecycleDepth.ACCESS, RecoveryChoice.DIRECT,
            RenewalChoice.RENEWAL, 4,
        ))
        one_access = [item.relation_identity for item in one if item.family is TimingFamily.ACCESS]
        four_access = [item.relation_identity for item in four if item.family is TimingFamily.ACCESS]
        self.assertEqual(one_access, four_access[:2])
        one_renewal = [item.relation_identity for item in one if item.family is TimingFamily.RENEWAL]
        four_renewal = [item.relation_identity for item in four if item.family is TimingFamily.RENEWAL]
        self.assertEqual(one_renewal, four_renewal)


class TimingMechanismTests(unittest.TestCase):
    def test_configuration_and_fingerprint_are_separate_and_stable(self):
        configuration = PopulationTimingConfiguration()
        self.assertEqual(TIMING_CALIBRATION_STATUS, configuration.calibration_status)
        self.assertEqual(configuration.fingerprint(), PopulationTimingConfiguration().fingerprint())
        self.assertTrue(configuration.fingerprint().startswith("sha256:"))
        self.assertEqual(300, cfg.security.token_lifetime_s)
        configured = {
            item.event_type: (item.center_s, item.dispersion_s)
            for item in configuration.gaussian_baselines
        }
        self.assertEqual(
            {event: baseline for event, baseline in DELAY_PROFILE.items()
             if event is not EventType.RETRY},
            configured,
        )

    def test_center_and_dispersion_use_independent_multipliers(self):
        steps = (EventType.AUTHENTICATION_REQUEST,)
        configuration = PopulationTimingConfiguration()
        relation = map_timing_relations(steps)[0]
        base = configuration.baseline_for(EventType.AUTHENTICATION_REQUEST)
        reference = derive_relation_randomness_reference(
            realizer(steps).timing_root_reference, relation, configuration,
        )
        z = normal_variate_from_reference(reference)
        profile = tendencies(auth=(1.20, 0.80))
        timing = realizer(steps, profile=profile)
        delay = timing.realize_delay(
            EventType.AUTHENTICATION_REQUEST,
            current_time=0.0,
            token_expiry=None,
            retry_count=0,
        )
        expected = max(0.0, 1.20 * base.center_s + z * 0.80 * base.dispersion_s)
        expected = max(expected, configuration.minimum_causal_increment_s)
        self.assertAlmostEqual(expected, delay, places=15)

    def test_family_specific_modifier_does_not_change_other_family(self):
        steps = (EventType.DISCOVERY, EventType.AUTHENTICATION_REQUEST)
        first = realizer(steps, profile=tendencies(auth=(0.75, 0.75)), key="same")
        second = realizer(steps, profile=tendencies(auth=(1.25, 1.25)), key="same")
        first_transition = first.realize_delay(
            EventType.DISCOVERY, current_time=0, token_expiry=None, retry_count=0,
        )
        second_transition = second.realize_delay(
            EventType.DISCOVERY, current_time=0, token_expiry=None, retry_count=0,
        )
        self.assertEqual(first_transition, second_transition)
        first_auth = first.realize_delay(
            EventType.AUTHENTICATION_REQUEST,
            current_time=first_transition, token_expiry=None, retry_count=0,
        )
        second_auth = second.realize_delay(
            EventType.AUTHENTICATION_REQUEST,
            current_time=second_transition, token_expiry=None, retry_count=0,
        )
        self.assertNotEqual(first_auth, second_auth)

    def test_retry_uses_auth_center_dispersion_and_positive_floor(self):
        steps = (EventType.AUTHENTICATION_REQUEST, EventType.RETRY)
        profile = tendencies(auth=(1.25, 0.75))
        timing = realizer(steps, profile=profile)
        timing.realize_delay(
            EventType.AUTHENTICATION_REQUEST,
            current_time=0, token_expiry=None, retry_count=1,
        )
        retry = timing.relations[1]
        reference = derive_relation_randomness_reference(
            timing.timing_root_reference, retry, timing.configuration,
        )
        base = 1.0 * 2.0**1
        expected = max(0.05, 1.25 * base + signed_uniform_from_reference(reference) * 0.75 * 0.20 * base)
        observed = timing.realize_delay(
            EventType.RETRY, current_time=1, token_expiry=None, retry_count=1,
        )
        self.assertEqual(expected, observed)
        self.assertGreaterEqual(observed, 0.05)

    def test_profile_timing_does_not_consume_global_rng_or_slots_one_to_four(self):
        state = random.getstate()
        source = derive_planning_randomness(
            planning_root_seed=4102,
            population_device_key="population-device-000",
            trace_construction_key="isolation",
        )
        upstream = tuple(source.variate_at(slot) for slot in (1, 2, 3, 4))
        timing = TimingRealizer(
            configuration=PopulationTimingConfiguration(),
            timing_profile=tendencies(),
            planning_rng_reference=source.reference,
            event_steps=(EventType.DISCOVERY,),
        )
        timing.realize_delay(EventType.DISCOVERY, current_time=0, token_expiry=None, retry_count=0)
        self.assertEqual(state, random.getstate())
        self.assertEqual(upstream, tuple(source.variate_at(slot) for slot in (1, 2, 3, 4)))
        self.assertEqual(5, int(PlanningDecisionSlot.TIMING))
        self.assertEqual("4c5-fixed-planning-slots-v4", PLANNING_DECISION_SLOT_ORDER_VERSION)


class TimingEnvelopeTests(unittest.TestCase):
    def test_candidate_inside_token_bound_is_unchanged(self):
        timing = realizer(
            (EventType.TOKEN_PRESENTED,),
            configuration=baseline_override(EventType.TOKEN_PRESENTED, 0.25),
        )
        observed = timing.realize_delay(
            EventType.TOKEN_PRESENTED,
            current_time=10.0,
            token_expiry=20.0,
            retry_count=0,
        )
        self.assertEqual(0.25, observed)
        self.assertFalse(timing.provenance_record()["relations"][0]["envelope_intervention"])

    def test_beyond_and_at_exclusive_bound_are_projected_strictly_inside(self):
        for center in (10.0, 0.999999):
            with self.subTest(center=center):
                timing = realizer(
                    (EventType.TOKEN_PRESENTED,),
                    configuration=baseline_override(EventType.TOKEN_PRESENTED, center),
                )
                observed = timing.realize_delay(
                    EventType.TOKEN_PRESENTED,
                    current_time=0.0,
                    token_expiry=1.0,
                    retry_count=0,
                )
                self.assertLess(observed, 1.0)
                record = timing.provenance_record()["relations"][0]
                self.assertTrue(record["envelope_intervention"])
                self.assertEqual(
                    "preserve_existing_token_validity",
                    record["envelope_intervention_reason"],
                )

    def test_no_upper_bound_and_empty_interval(self):
        timing = realizer((EventType.DISCOVERY,))
        self.assertGreater(timing.realize_delay(
            EventType.DISCOVERY, current_time=0, token_expiry=None, retry_count=0,
        ), 0)
        with self.assertRaises(EmptyTimingEnvelopeError):
            realizer((EventType.TOKEN_PRESENTED,)).realize_delay(
                EventType.TOKEN_PRESENTED,
                current_time=1.0,
                token_expiry=1.000001,
                retry_count=0,
            )

    def test_selected_multi_access_and_renewal_structure_remains_valid(self):
        steps = compose_ordinary_trace(
            LifecycleDepth.ACCESS,
            RecoveryChoice.ONE_RETRY,
            RenewalChoice.RENEWAL,
            4,
        )
        timing = realizer(
            steps,
            profile=tendencies(
                auth=(1.25, 1.25), transition=(1.25, 1.25),
                access=(1.25, 1.25), renewal=(1.25, 1.25),
            ),
            key="extreme-validity",
        )
        engine = EventEngine(
            device_id="device-1",
            gateway_id="gateway-1",
            auth_server_id="auth-1",
            persistent_context=PersistentDeviceContext(device_id="device-1"),
            start_time=1_000.0,
            timing_realizer=timing,
        )
        events, context = engine.execute(ScenarioSpec(
            scenario_id="scenario-1",
            scenario_type="normal",
            is_anomaly=False,
            generation_profile=SYNCHRONIZED_GENERATION_PROFILE,
            normal_steps=list(steps),
        ))
        self.assertEqual(steps, tuple(event.event_type for event in events))
        self.assertEqual(4, sum(event.event_type is EventType.ACCESS_GRANTED for event in events))
        self.assertEqual(1, sum(event.event_type is EventType.RETRY for event in events))
        self.assertEqual(1, sum(event.event_type is EventType.RENEWAL_REQUEST for event in events))
        self.assertTrue(all(event.result is not EventResult.FAILURE or event.event_type is EventType.AUTHENTICATION_FAILURE for event in events))
        first_issuance = next(event for event in events if event.event_type is EventType.TOKEN_ISSUED)
        accesses = [event for event in events if event.event_type in {EventType.ACCESS_REQUEST, EventType.ACCESS_GRANTED}]
        self.assertTrue(all(event.timestamp < first_issuance.token_expiry for event in accesses))
        issuances = [event for event in events if event.event_type is EventType.TOKEN_ISSUED]
        self.assertEqual(2, len(issuances))
        self.assertTrue(all(math.isclose(
            event.token_expiry - event.timestamp,
            cfg.security.token_lifetime_s,
        ) for event in issuances))
        self.assertIsNotNone(context.renewed_token_issued_at)


class TimingProvenanceTests(unittest.TestCase):
    def test_provenance_is_reconstructable_and_contains_no_latent_profile(self):
        steps = (EventType.DISCOVERY, EventType.AUTHENTICATION_REQUEST)
        profile = tendencies(auth=(1.20, 0.80))
        timing = realizer(steps, profile=profile, key="reconstruct")
        current = 0.0
        for event_type in steps:
            current += timing.realize_delay(
                event_type, current_time=current, token_expiry=None, retry_count=0,
            )
        record = timing.provenance_record()
        self.assertEqual(2, record["realized_relation_count"])
        for relation, item in zip(timing.relations, record["relations"]):
            reference = derive_relation_randomness_reference(
                timing.timing_root_reference, relation, timing.configuration,
            )
            tendency = (
                profile.auth if relation.family is TimingFamily.AUTH
                else profile.transition
            )
            baseline = timing.configuration.baseline_for(relation.event_type)
            candidate = max(
                0.0,
                tendency.center_modifier * baseline.center_s
                + normal_variate_from_reference(reference)
                * tendency.dispersion_modifier * baseline.dispersion_s,
            )
            expected = max(candidate, timing.configuration.minimum_causal_increment_s)
            self.assertEqual(expected, item["realized_delay_s"])
        serialized = repr(record)
        for forbidden in (
            "center_modifier", "dispersion_modifier", "behavioral_profile_id",
            "population_device_key", "activity", "recovery_tendency",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_runner_publishes_timing_only_in_privileged_trace_provenance(self):
        requests = [
            TraceConstructionRequest.profile_conditioned_lifecycle(
                trace_construction_key=f"runner-{index:03d}",
                scenario_id=f"scenario-{index:03d}",
            )
            for index in range(12)
        ]
        original_devices = cfg.simulation.num_devices
        original_seed = cfg.simulation.random_seed
        try:
            cfg.simulation.num_devices = 1
            cfg.simulation.random_seed = 4103
            with patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ):
                sequences, population = run_simulation_with_population(
                    planning_requests=requests,
                    planning_root_seed=4102,
                )
        finally:
            cfg.simulation.num_devices = original_devices
            cfg.simulation.random_seed = original_seed
        bundle = build_synchronized_bundle(sequences, run_population=population)
        records = [
            item["trace_planning"] for item in bundle["provenance"]["traces"]
        ]
        self.assertEqual(12, len(records))
        self.assertTrue(all("timing_realization" in item for item in records))
        self.assertTrue(all(
            item["timing_realization"]["timing_decision_slot"] == 5
            for item in records
        ))
        timing_metadata = bundle["provenance"]["dataset"]["randomness_metadata"][
            "timing_realization"
        ]
        self.assertEqual([5], timing_metadata["decision_slots"])
        self.assertEqual(
            {"validated_normal"},
            set(bundle["validation"]["trace_ground_truth_statuses"].values()),
        )
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        for surface in ("detector_observations", "ground_truth", "debug"):
            serialized = repr(bundle[surface])
            for forbidden in (
                "timing_root_reference", "timing_family",
                "timing_configuration_fingerprint", "envelope_intervention",
                "center_modifier", "dispersion_modifier",
            ):
                self.assertNotIn(forbidden, serialized)


class TimingCharacterizationCampaignTests(unittest.TestCase):
    def test_eight_device_192_trace_structure_reconstruction_and_validity(self):
        population = build_profile_population(
            [(f"device-{index}", f"population-device-{index:03d}")
             for index in range(8)],
            PopulationConfiguration(population_seed=4101),
        )
        resolver = ApplicabilityResolver()
        planner = DeviceAwareTracePlanner()
        timing_cfg = PopulationTimingConfiguration()
        sequences = []
        records_by_device = defaultdict(list)
        structural = {"L": 0, "R": 0, "N": 0, "A": 0, "sequence": 0}
        reconstructed_relations = 0
        random.seed(4103)

        for assignment in population.assignments:
            persistent = PersistentDeviceContext(device_id=assignment.device_id)
            for trace_index in range(24):
                key = f"{assignment.population_device_key}-trace-{trace_index:03d}"
                request = TraceConstructionRequest.profile_conditioned_lifecycle(
                    trace_construction_key=key,
                    scenario_id=f"scenario-{key}",
                )
                source = derive_planning_randomness(
                    planning_root_seed=4102,
                    population_device_key=assignment.population_device_key,
                    trace_construction_key=key,
                )
                lifecycle_applicable = resolver.resolve(
                    persistent, ConstructionRole.ORDINARY,
                )
                lifecycle = reconstruct_profile_conditioned_lifecycle(
                    assignment.profile.lifecycle, lifecycle_applicable, source,
                )
                recovery_applicable = resolver.resolve_recovery(
                    persistent, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                )
                recovery = reconstruct_profile_conditioned_recovery(
                    assignment.profile.recovery, recovery_applicable, source,
                )
                base = compose_ordinary_lifecycle(
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                )
                renewal_applicable = resolver.resolve_renewal(
                    persistent, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                    base,
                )
                renewal = reconstruct_profile_conditioned_renewal(
                    assignment.profile.renewal, renewal_applicable, source,
                )
                activity_applicable = resolver.resolve_activity(
                    persistent, ConstructionRole.ORDINARY,
                    lifecycle.selected_lifecycle_depth,
                    recovery.selected_recovery_choice,
                    renewal.selected_renewal_choice,
                    base,
                )
                plan = planner.plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
                    request, lifecycle, recovery, renewal, activity_applicable,
                    assignment.profile.activity, source,
                )
                timing = TimingRealizer(
                    configuration=timing_cfg,
                    timing_profile=assignment.profile.timing,
                    planning_rng_reference=source.reference,
                    event_steps=plan.event_steps,
                )
                expected = (
                    plan.event_steps
                    if not persistent.enrolled
                    else plan.event_steps[6:]
                )
                pair = EventEngine(
                    device_id=assignment.device_id,
                    gateway_id="gateway",
                    auth_server_id="auth",
                    broker_id="broker",
                    source_ip="10.0.0.1",
                    battery_level=80,
                    firmware_version="1",
                    persistent_context=persistent,
                    start_time=1_000.0,
                    timing_realizer=timing,
                ).execute(adapt_trace_plan(plan))
                events, _context = pair
                self.assertEqual(expected, tuple(event.event_type for event in events))
                self.assertEqual(
                    plan.selected_protected_operation_count,
                    sum(event.event_type is EventType.ACCESS_GRANTED for event in events),
                )
                record = timing.provenance_record()
                self.assertEqual(len(events), record["realized_relation_count"])
                family_tendencies = {
                    "AUTH": assignment.profile.timing.auth,
                    "TRANSITION": assignment.profile.timing.transition,
                    "ACCESS": assignment.profile.timing.access,
                    "RENEWAL": assignment.profile.timing.renewal,
                }
                for relation, item in zip(timing.relations, record["relations"]):
                    reference = derive_relation_randomness_reference(
                        timing.timing_root_reference, relation, timing_cfg,
                    )
                    self.assertEqual(reference, item["relation_randomness_reference"])
                    tendency = family_tendencies[item["timing_family"]]
                    if relation.event_type is EventType.RETRY:
                        base = timing_cfg.retry_base_backoff_s * timing_cfg.retry_backoff_factor
                        candidate = max(
                            timing_cfg.retry_positive_floor_s,
                            tendency.center_modifier * base
                            + signed_uniform_from_reference(reference)
                            * tendency.dispersion_modifier
                            * timing_cfg.retry_jitter_fraction * base,
                        )
                    else:
                        baseline = timing_cfg.baseline_for(relation.event_type)
                        candidate = max(
                            0.0,
                            tendency.center_modifier * baseline.center_s
                            + normal_variate_from_reference(reference)
                            * tendency.dispersion_modifier * baseline.dispersion_s,
                        )
                    expected_delay = max(
                        candidate, timing_cfg.minimum_causal_increment_s,
                    )
                    upper = item["upper_delay_exclusive_s"]
                    if upper is not None and expected_delay >= upper:
                        expected_delay = math.nextafter(upper, 0.0)
                    self.assertAlmostEqual(candidate, item["candidate_delay_s"], places=13)
                    self.assertAlmostEqual(expected_delay, item["realized_delay_s"], places=13)
                    reconstructed_relations += 1
                records_by_device[assignment.device_id].extend(record["relations"])
                sequences.append(pair)
                structural["L"] += int(plan.lifecycle_depth is lifecycle.selected_lifecycle_depth)
                structural["R"] += int(plan.recovery_choice is recovery.selected_recovery_choice)
                structural["N"] += int(plan.renewal_choice is renewal.selected_renewal_choice)
                structural["A"] += int(
                    plan.selected_protected_operation_count
                    == sum(event.event_type is EventType.ACCESS_GRANTED for event in events)
                )
                structural["sequence"] += 1

        self.assertEqual(
            {"L": 192, "R": 192, "N": 192, "A": 192, "sequence": 192},
            structural,
        )
        self.assertEqual(
            sum(len(events) for events, _ in sequences),
            reconstructed_relations,
        )
        bundle = build_synchronized_bundle(sequences)
        self.assertEqual(
            {"validated_normal"},
            set(bundle["validation"]["trace_ground_truth_statuses"].values()),
        )
        self.assertTrue(bundle["validation"]["build_gate"]["passed"])
        all_events = [event for events, _ in sequences for event in events]
        self.assertTrue(all(
            event.observed_timestamp == event.timestamp for event in all_events
        ))
        self.assertTrue(all(
            event.timestamp < event.token_expiry
            for event in all_events
            if event.event_type in {
                EventType.TOKEN_VALIDATED,
                EventType.ACCESS_REQUEST,
                EventType.ACCESS_GRANTED,
            } and event.token_expiry is not None
        ))

        # Repeated relation types vary within at least one device, while
        # observed ranges for at least two devices overlap.
        access_delays = {
            device: [item["realized_delay_s"] for item in records
                     if item["timing_family"] == "ACCESS"]
            for device, records in records_by_device.items()
        }
        self.assertTrue(any(len(set(values)) > 1 for values in access_delays.values()))
        populated = [values for values in access_delays.values() if values]
        self.assertTrue(any(
            max(min(left), min(right)) <= min(max(left), max(right))
            for index, left in enumerate(populated)
            for right in populated[index + 1:]
        ))

        # T is relation-specific rather than a single global speed ordering.
        auth_order = sorted(
            population.assignments,
            key=lambda item: item.profile.timing.auth.center_modifier,
        )
        access_order = sorted(
            population.assignments,
            key=lambda item: item.profile.timing.access.center_modifier,
        )
        self.assertNotEqual(
            [item.population_device_key for item in auth_order],
            [item.population_device_key for item in access_order],
        )


if __name__ == "__main__":
    unittest.main()
