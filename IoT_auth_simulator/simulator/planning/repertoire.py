"""Versioned compositional ordinary lifecycle and recovery repertoire."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from simulator.engines.scenario_engine import (
    AUTH_PREFIX,
    CLEAN_ACCESS_CYCLE,
    NORMAL_FLOW,
    NORMAL_FLOW_WITH_RETRY,
    NORMAL_FLOW_WITH_RENEWAL,
    REGISTRATION_PHASE,
)
from simulator.event_model import EventType
from simulator.planning.models import LifecycleDepth, RecoveryChoice, RenewalChoice


LIFECYCLE_REPERTOIRE_VERSION = "4c-inf-ordinary-lifecycle-repertoire-v1"
RECOVERY_REPERTOIRE_VERSION = "4c2-authentication-recovery-repertoire-v1"
RENEWAL_REPERTOIRE_VERSION = "4c3-single-renewal-extension-v1"
ACTIVITY_REPERTOIRE_VERSION = "4c4-clean-protected-operations-1-to-4-v1"


@dataclass(frozen=True)
class LifecycleTemplate:
    """Generator mapping only; the lifecycle ID is not a BM state."""

    lifecycle_depth: LifecycleDepth
    template_reference: str
    intentional_endpoint: EventType
    event_steps: tuple[EventType, ...]


ORDINARY_LIFECYCLE_REPERTOIRE: Mapping[LifecycleDepth, LifecycleTemplate] = MappingProxyType({
    LifecycleDepth.AUTH: LifecycleTemplate(
        lifecycle_depth=LifecycleDepth.AUTH,
        template_reference="ordinary-direct-auth-success-v1",
        intentional_endpoint=EventType.AUTHENTICATION_SUCCESS,
        event_steps=tuple(NORMAL_FLOW[:11]),
    ),
    LifecycleDepth.SESSION: LifecycleTemplate(
        lifecycle_depth=LifecycleDepth.SESSION,
        template_reference="ordinary-session-open-v1",
        intentional_endpoint=EventType.SESSION_OPENED,
        event_steps=tuple(AUTH_PREFIX),
    ),
    LifecycleDepth.ACCESS: LifecycleTemplate(
        lifecycle_depth=LifecycleDepth.ACCESS,
        template_reference="ordinary-single-clean-access-v1",
        intentional_endpoint=EventType.ACCESS_GRANTED,
        event_steps=tuple([*AUTH_PREFIX, *CLEAN_ACCESS_CYCLE]),
    ),
})


@dataclass(frozen=True)
class RecoveryTemplate:
    """One supported eventually-successful authentication subtrajectory."""

    recovery_choice: RecoveryChoice
    template_reference: str
    event_steps: tuple[EventType, ...]


_ONBOARDING_LENGTH = len(REGISTRATION_PHASE)
AUTHENTICATION_RECOVERY_REPERTOIRE: Mapping[
    RecoveryChoice, RecoveryTemplate
] = MappingProxyType({
    RecoveryChoice.DIRECT: RecoveryTemplate(
        recovery_choice=RecoveryChoice.DIRECT,
        template_reference="ordinary-direct-authentication-v1",
        event_steps=tuple(NORMAL_FLOW[_ONBOARDING_LENGTH:11]),
    ),
    RecoveryChoice.ONE_RETRY: RecoveryTemplate(
        recovery_choice=RecoveryChoice.ONE_RETRY,
        template_reference="ordinary-one-failure-one-retry-success-v1",
        event_steps=tuple(NORMAL_FLOW_WITH_RETRY[_ONBOARDING_LENGTH:16]),
    ),
})


LIFECYCLE_SUFFIX_REPERTOIRE: Mapping[
    LifecycleDepth, tuple[EventType, ...]
] = MappingProxyType({
    LifecycleDepth.AUTH: (),
    LifecycleDepth.SESSION: (
        EventType.TOKEN_ISSUED,
        EventType.TOKEN_PRESENTED,
        EventType.TOKEN_VALIDATED,
        EventType.SESSION_OPENED,
    ),
    LifecycleDepth.ACCESS: (
        EventType.TOKEN_ISSUED,
        EventType.TOKEN_PRESENTED,
        EventType.TOKEN_VALIDATED,
        EventType.SESSION_OPENED,
        EventType.ACCESS_REQUEST,
        EventType.ACCESS_GRANTED,
    ),
})


# Exact existing supported continuation beginning after the one clean ACCESS.
RENEWAL_EXTENSION_REPERTOIRE: Mapping[
    RenewalChoice, tuple[EventType, ...]
] = MappingProxyType({
    RenewalChoice.NO_RENEWAL: (),
    RenewalChoice.RENEWAL: tuple(NORMAL_FLOW_WITH_RENEWAL[17:]),
})


def compose_ordinary_lifecycle(
    lifecycle_depth: LifecycleDepth,
    recovery_choice: RecoveryChoice,
) -> tuple[EventType, ...]:
    """Compose onboarding, one supported auth trajectory, and a depth suffix."""
    recovery = AUTHENTICATION_RECOVERY_REPERTOIRE.get(recovery_choice)
    suffix = LIFECYCLE_SUFFIX_REPERTOIRE.get(lifecycle_depth)
    if recovery is None or suffix is None:
        raise KeyError((lifecycle_depth, recovery_choice))
    return tuple((*REGISTRATION_PHASE, *recovery.event_steps, *suffix))


def recovery_template_reference(
    lifecycle_depth: LifecycleDepth,
    recovery_choice: RecoveryChoice,
) -> str:
    """Return a stable reference for one supported compositional realization."""
    if lifecycle_depth not in LIFECYCLE_SUFFIX_REPERTOIRE:
        raise KeyError((lifecycle_depth, recovery_choice))
    recovery = AUTHENTICATION_RECOVERY_REPERTOIRE[recovery_choice]
    return f"{recovery.template_reference}+{lifecycle_depth.value.lower()}-endpoint-v1"


def compose_ordinary_trace(
    lifecycle_depth: LifecycleDepth,
    recovery_choice: RecoveryChoice,
    renewal_choice: RenewalChoice,
    protected_operation_count: int | None = None,
) -> tuple[EventType, ...]:
    """Compose L/R/N plus the supported pre-renewal clean activity count.

    Omitting ``protected_operation_count`` preserves the 4C.3 structural base:
    zero operations for AUTH/SESSION and one for ACCESS.
    """
    base = compose_ordinary_lifecycle(lifecycle_depth, recovery_choice)
    extension = RENEWAL_EXTENSION_REPERTOIRE.get(renewal_choice)
    if extension is None or (
        renewal_choice is RenewalChoice.RENEWAL
        and lifecycle_depth is not LifecycleDepth.ACCESS
    ):
        raise KeyError((lifecycle_depth, recovery_choice, renewal_choice))
    structural_count = 1 if lifecycle_depth is LifecycleDepth.ACCESS else 0
    count = structural_count if protected_operation_count is None else protected_operation_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise KeyError((lifecycle_depth, recovery_choice, renewal_choice, count))
    if lifecycle_depth is LifecycleDepth.ACCESS:
        if not 1 <= count <= 4:
            raise KeyError((lifecycle_depth, recovery_choice, renewal_choice, count))
        additional_activity = tuple(CLEAN_ACCESS_CYCLE) * (count - 1)
    else:
        if count != 0:
            raise KeyError((lifecycle_depth, recovery_choice, renewal_choice, count))
        additional_activity = ()
    return tuple((*base, *additional_activity, *extension))


def renewal_extension_reference(renewal_choice: RenewalChoice) -> str:
    if renewal_choice not in RENEWAL_EXTENSION_REPERTOIRE:
        raise KeyError(renewal_choice)
    return (
        "ordinary-no-renewal-extension-v1"
        if renewal_choice is RenewalChoice.NO_RENEWAL
        else "ordinary-single-successful-renewal-extension-v1"
    )
