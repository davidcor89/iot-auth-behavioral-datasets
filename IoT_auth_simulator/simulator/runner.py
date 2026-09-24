"""
simulator/runner.py
────────────────────
Orchestrator: generates a mixed batch of normal and anomaly sessions
using ScenarioEngine + EventEngine, and returns the labelled event sequences.

Architecture (Phase 2)
──────────────────────
  ScenarioEngine.batch()  →  List[ScenarioSpec]
  EventEngine.execute()   →  (List[AuthEvent], SessionContext)
  synchronized exporter   →  observations, GT, provenance, debug, validation
"""

import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from simulator.config.settings import cfg
from simulator.config.activity_planning import ActivityPlanningConfiguration
from simulator.config.device_profile_population import PopulationConfiguration
from simulator.config.population_timing import PopulationTimingConfiguration
from simulator.config.scientific_population import ScientificPopulationConfiguration
from simulator.core.device import Device
from simulator.device_profiles.population import (
    DeviceProfilePopulation,
    build_profile_population,
    population_device_key_from_index,
)
from simulator.device_profiles.scientific_population import (
    ScientificDeviceProfilePopulation,
    build_scientific_profile_population,
)
from simulator.engines.scenario_engine import ScenarioEngine
from simulator.engines.event_engine import EventEngine, SessionContext
from simulator.event_model import AuthEvent
from simulator.security_context import PersistentDeviceContext
from simulator.planning.models import PlanningMode, TraceConstructionRequest
from simulator.planning.planner import (
    ApplicabilityResolver,
    DeviceAwareTracePlanner,
    adapt_trace_plan,
    reconstruct_profile_conditioned_lifecycle,
    reconstruct_profile_conditioned_recovery,
    reconstruct_profile_conditioned_renewal,
)
from simulator.planning.randomness import derive_planning_randomness
from simulator.planning.repertoire import compose_ordinary_lifecycle
from simulator.timing.realizer import TimingRealizer


@dataclass(frozen=True)
class RunPopulation:
    """Sibling run-scoped registries and privileged planning audit records."""

    devices: List[Device]
    devices_by_id: Dict[str, Device]
    semantic_contexts: Dict[str, PersistentDeviceContext]
    behavioral_profiles: DeviceProfilePopulation | ScientificDeviceProfilePopulation
    trace_planning_records: Dict[str, dict] = field(default_factory=dict)


def build_run_population(
    profile_configuration: Optional[
        PopulationConfiguration | ScientificPopulationConfiguration
    ] = None,
) -> RunPopulation:
    """Build the existing device/context pool plus one persistent profile/device.

    This function does not seed the legacy RNG. Callers retain control of the
    existing device-factory stream; profile sampling uses independent RNGs.
    """
    # Preserve the historical construction order and random draws.
    devices = [Device.create(index=i) for i in range(cfg.simulation.num_devices)]
    semantic_contexts = {
        device.device_id: PersistentDeviceContext(
            device_id=device.device_id,
            bootstrap_credential_reference=device.get_psk_hash(),
            legacy_device_type_id=device.device_type,
        )
        for device in devices
    }
    configuration = profile_configuration or PopulationConfiguration(
        population_seed=cfg.simulation.random_seed,
    )
    bindings = tuple(
        (device.device_id, population_device_key_from_index(index))
        for index, device in enumerate(devices)
    )
    if isinstance(configuration, ScientificPopulationConfiguration):
        behavioral_profiles = build_scientific_profile_population(
            bindings, configuration,
        )
    else:
        behavioral_profiles = build_profile_population(bindings, configuration)
    return RunPopulation(
        devices=devices,
        devices_by_id={device.device_id: device for device in devices},
        semantic_contexts=semantic_contexts,
        behavioral_profiles=behavioral_profiles,
    )


def run_simulation(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    population_callback: Optional[Callable[[RunPopulation], None]] = None,
    planning_requests: Optional[List[TraceConstructionRequest]] = None,
    planning_root_seed: Optional[int] = None,
    activity_configuration: Optional[ActivityPlanningConfiguration] = None,
    timing_configuration: Optional[PopulationTimingConfiguration] = None,
    profile_configuration: Optional[
        PopulationConfiguration | ScientificPopulationConfiguration
    ] = None,
) -> List[Tuple[List[AuthEvent], SessionContext]]:
    """
    Run the full simulation.

    Returns
    -------
    List of (events, context) pairs — one per session.
    Each pair contains:
      events  : correlated AuthEvent sequence (variable length)
      context : SessionContext with all session-level features for the feature CSV
    """
    random.seed(cfg.simulation.random_seed)
    # One coordinate anchor per run. It is used only to make exported semantic
    # times human-readable; all ordering and validity use per-device clocks.
    semantic_time_origin = time.time()

    # Keep the device pool, semantic contexts, and sibling profile registry.
    # Only the opt-in ordinary planner reads L/R/N/A; EventEngine, validators,
    # HN, and anomaly construction do not receive a profile object.
    population = build_run_population(profile_configuration)
    if population_callback is not None:
        population_callback(population)
    devices = population.devices
    persistent_contexts = population.semantic_contexts

    # Shared gateway / server / broker identifiers (round-robin by index).
    gateway_ids = [str(uuid.uuid4())[:8] for _ in range(cfg.simulation.num_gateways)]
    auth_server_id = "auth-01"
    broker_id      = "broker-01"

    # ── Phase C: optionally instantiate the real domain entities and drive them
    # through generation (flag-gated via cfg.simulation.wire_entities). ─────────
    wired = getattr(cfg.simulation, "wire_entities", False)
    entities = None
    if wired:
        from simulator.core.gateway import Gateway
        from simulator.core.auth_server import AuthServer, MQTTBroker
        entities = {
            "auth_server": AuthServer(server_id=auth_server_id),
            "broker":      MQTTBroker(broker_id=broker_id),
            "gateways":    {gid: Gateway(gateway_id=gid) for gid in gateway_ids},
            "stats":       defaultdict(int),
        }

    # Default generation remains scenario-before-device and is intentionally
    # unchanged. The opt-in 4C planning path starts from ordinary construction
    # requests, assigns a device, and only then builds a compatibility spec.
    planning_enabled = planning_requests is not None
    if planning_enabled:
        work_items = list(planning_requests)
        applicability_resolver = ApplicabilityResolver()
        trace_planner = DeviceAwareTracePlanner()
        effective_planning_seed = (
            cfg.simulation.random_seed
            if planning_root_seed is None else planning_root_seed
        )
        effective_activity_configuration = (
            activity_configuration or ActivityPlanningConfiguration()
        )
        effective_timing_configuration = (
            timing_configuration or PopulationTimingConfiguration()
        )
    else:
        scenario_engine = ScenarioEngine(seed=cfg.simulation.random_seed)
        work_items = scenario_engine.synchronized_batch(
            n_normal     = cfg.simulation.num_sessions_normal,
            n_attack     = cfg.simulation.num_sessions_attack,
            distribution = cfg.simulation.attack_distribution,
        )

    total     = len(work_items)
    sequences: List[Tuple[List[AuthEvent], SessionContext]] = []

    for i, item in enumerate(work_items):
        device     = random.choice(devices)
        gateway_id = gateway_ids[i % len(gateway_ids)]

        plan = None
        if planning_enabled:
            request = item
            semantic_context = persistent_contexts[device.device_id]
            profile_assignment = population.behavioral_profiles.assignment_for_device(
                device.device_id
            )
            applicable = applicability_resolver.resolve(
                semantic_context, request.construction_role,
            )
            planning_randomness = derive_planning_randomness(
                planning_root_seed=effective_planning_seed,
                population_device_key=profile_assignment.population_device_key,
                trace_construction_key=request.trace_construction_key,
            )
            if request.planning_mode is PlanningMode.PROFILE_CONDITIONED_LIFECYCLE:
                # Fixed decision identities: L/R/N/A own slots 1/2/3/4;
                # timing owns a derived root at conceptual slot 5.
                # Applicability is resolved before each tendency use.
                lifecycle_selection = reconstruct_profile_conditioned_lifecycle(
                    profile_assignment.profile.lifecycle,
                    applicable,
                    planning_randomness,
                )
                recovery_applicable = applicability_resolver.resolve_recovery(
                    semantic_context,
                    request.construction_role,
                    lifecycle_selection.selected_lifecycle_depth,
                )
                recovery_selection = reconstruct_profile_conditioned_recovery(
                    profile_assignment.profile.recovery,
                    recovery_applicable,
                    planning_randomness,
                )
                base_event_steps = compose_ordinary_lifecycle(
                    lifecycle_selection.selected_lifecycle_depth,
                    recovery_selection.selected_recovery_choice,
                )
                renewal_applicable = applicability_resolver.resolve_renewal(
                    semantic_context,
                    request.construction_role,
                    lifecycle_selection.selected_lifecycle_depth,
                    recovery_selection.selected_recovery_choice,
                    base_event_steps,
                )
                renewal_selection = reconstruct_profile_conditioned_renewal(
                    profile_assignment.profile.renewal,
                    renewal_applicable,
                    planning_randomness,
                )
                activity_applicable = applicability_resolver.resolve_activity(
                    semantic_context,
                    request.construction_role,
                    lifecycle_selection.selected_lifecycle_depth,
                    recovery_selection.selected_recovery_choice,
                    renewal_selection.selected_renewal_choice,
                    base_event_steps,
                    effective_activity_configuration,
                )
                plan = trace_planner.plan_profile_conditioned_lifecycle_recovery_renewal_and_activity(
                    request,
                    lifecycle_selection,
                    recovery_selection,
                    renewal_selection,
                    activity_applicable,
                    profile_assignment.profile.activity,
                    planning_randomness,
                    effective_activity_configuration,
                )
            else:
                # 4C.INF compatibility modes remain deterministic and dormant.
                planning_randomness.new_rng()
                plan = trace_planner.plan(
                    request,
                    applicable,
                    profile_assignment.profile,
                    planning_randomness,
                )
            spec = adapt_trace_plan(plan)
        else:
            spec = item

        timing_realizer = None
        if (
            plan is not None
            and request.planning_mode is PlanningMode.PROFILE_CONDITIONED_LIFECYCLE
        ):
            timing_realizer = TimingRealizer(
                configuration=effective_timing_configuration,
                timing_profile=profile_assignment.profile.timing,
                planning_rng_reference=planning_randomness.reference,
                event_steps=plan.event_steps,
            )

        driver = None
        if wired:
            from simulator.core.session_driver import SessionDriver
            driver = SessionDriver(
                device      = device,
                gateway     = entities["gateways"][gateway_id],
                auth_server = entities["auth_server"],
                broker      = entities["broker"],
                stats       = entities["stats"],
            )

        engine = EventEngine(
            device_id        = device.device_id,
            gateway_id       = gateway_id,
            auth_server_id   = auth_server_id,
            broker_id        = broker_id,
            source_ip        = device.ip_address,
            battery_level    = device.battery_level,
            firmware_version = device.firmware_version,
            driver           = driver,
            start_time       = semantic_time_origin,
            persistent_context = persistent_contexts[device.device_id],
            timing_realizer  = timing_realizer,
        )
        pair = engine.execute(spec)
        sequences.append(pair)
        if plan is not None:
            planning_record = plan.provenance_record()
            if timing_realizer is not None:
                planning_record["timing_realization"] = (
                    timing_realizer.provenance_record()
                )
            population.trace_planning_records[pair[1].trace_id] = planning_record

        if progress_callback:
            label = "normal" if not spec.is_anomaly else spec.anomaly_type
            progress_callback(i + 1, total, f"Session {i + 1}/{total} [{label}]")

    if wired:
        _print_entity_summary(entities)

    return sequences


def run_simulation_with_population(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    planning_requests: Optional[List[TraceConstructionRequest]] = None,
    planning_root_seed: Optional[int] = None,
    activity_configuration: Optional[ActivityPlanningConfiguration] = None,
    timing_configuration: Optional[PopulationTimingConfiguration] = None,
    profile_configuration: Optional[
        PopulationConfiguration | ScientificPopulationConfiguration
    ] = None,
) -> Tuple[List[Tuple[List[AuthEvent], SessionContext]], RunPopulation]:
    """Retain the run-owned registries for separate privileged DP.2 export.

    The existing ``run_simulation`` result and generation order remain intact.
    Profile-conditioned planning is opt-in. EventEngine may receive a timing
    realizer owning only T for that trace; semantic context still owns no
    profile value, and the default legacy generation path remains unchanged.
    """
    populations: List[RunPopulation] = []
    sequences = run_simulation(
        progress_callback=progress_callback,
        population_callback=populations.append,
        planning_requests=planning_requests,
        planning_root_seed=planning_root_seed,
        activity_configuration=activity_configuration,
        timing_configuration=timing_configuration,
        profile_configuration=profile_configuration,
    )
    return sequences, populations[0]


def _print_entity_summary(entities: dict) -> None:
    """Report the aggregate activity of the real domain entities after a wired run."""
    st  = entities["stats"]
    a   = entities["auth_server"]
    b   = entities["broker"]
    gws = entities["gateways"]
    total_gw_sessions = sum(len(g.sessions) for g in gws.values())
    total_nonces      = sum(len(g._nonce_cache) for g in gws.values())
    print("  ── Domain entities exercised (wire_entities=on) ──")
    print(f"    ECDH exchanges     : {st['ecdh_exchanges']:,}")
    print(f"    Enrollments        : {a.total_enrollments:,}")
    print(f"    Tokens issued      : {a.total_tokens_issued:,}")
    print(f"    Broker CONNECTs    : {b.total_connects:,}")
    print(f"    Broker PUBLISHes   : {b.total_publishes:,}")
    print(f"    Gateway sessions   : {total_gw_sessions:,}")
    print(f"    Nonces cached      : {total_nonces:,}")
