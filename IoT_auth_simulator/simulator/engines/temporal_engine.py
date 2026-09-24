"""
simulator/engines/temporal_engine.py
──────────────────────────────────────
Phase 3 — Temporal Engine

Retains historical per-trace temporal bookkeeping and helpers.

  Token lifecycle
    - Records when a token is issued / renewed
    - Checks token expiry in the current EventEngine path
    - Computes token age at any moment

  Nonce lifecycle
    - Records nonce issue time; nonce-expiry helper is not a current guard

  Retry policy
    - Tracks consecutive failure count
    - Computes exponential-backoff delay for already-selected RETRY events
    - Retains dormant generic retry-limit helpers

  Replay window
    - Retains a historical age-window helper, not synchronized replay truth

One TemporalEngine instance is created per trace by the EventEngine. Active
expiry/bookkeeping/backoff behavior is separate from dormant decision helpers:
retry backoff does not select retry occurrence; should_renew() does not select
the current renewal template or a future Device Profile renewal tendency;
is_replay_violation() does not define canonical synchronized token replay.
The helpers remain available for compatibility and are not deleted here.
"""

import random
from dataclasses import dataclass
from typing import Optional


# ── Default lifetime constants (seconds) ──────────────────────────────────────
DEFAULT_TOKEN_LIFETIME     = 3600.0   # 1 hour
DEFAULT_ATTACK_TOKEN_LIFE  = 120.0    # 2 min — short, more exploitable
DEFAULT_NONCE_LIFETIME     = 30.0     # nonce expires after 30 s
DEFAULT_RENEWAL_THRESHOLD  = 0.80     # renew when 80 % of lifetime elapsed
DEFAULT_REPLAY_WINDOW      = 60.0     # token presented > 60 s after issuance = replay

DEFAULT_MAX_RETRIES   = 3
DEFAULT_BASE_BACKOFF  = 1.0           # seconds
DEFAULT_BACKOFF_FACTOR = 2.0          # exponential multiplier


@dataclass
class TemporalConfig:
    token_lifetime:      float = DEFAULT_TOKEN_LIFETIME
    attack_token_life:   float = DEFAULT_ATTACK_TOKEN_LIFE
    nonce_lifetime:      float = DEFAULT_NONCE_LIFETIME
    renewal_threshold:   float = DEFAULT_RENEWAL_THRESHOLD
    replay_window:       float = DEFAULT_REPLAY_WINDOW
    max_retries:         int   = DEFAULT_MAX_RETRIES
    base_backoff:        float = DEFAULT_BASE_BACKOFF
    backoff_factor:      float = DEFAULT_BACKOFF_FACTOR


class TemporalEngine:
    """
    Per-session temporal state tracker.

    Parameters
    ----------
    config    : TemporalConfig — lifetimes and retry policy
    is_attack : if True, uses the shorter attack token lifetime
    """

    def __init__(
        self,
        config:    Optional[TemporalConfig] = None,
        is_attack: bool = False,
    ):
        self.cfg       = config or TemporalConfig()
        self.is_attack = is_attack

        self._token_issued_at:    Optional[float] = None
        self._token_presented_at: Optional[float] = None
        self._nonce_issued_at:    Optional[float] = None
        self._retry_count:        int             = 0
        self._renewal_count:      int             = 0

        self._effective_token_life = (
            self.cfg.attack_token_life if is_attack
            else self.cfg.token_lifetime
        )

    # ── Token lifecycle ───────────────────────────────────────────────────────

    def record_token_issued(self, now: float) -> None:
        self._token_issued_at = now

    def record_token_presented(self, now: float) -> None:
        self._token_presented_at = now

    def is_token_expired(self, now: float) -> bool:
        if self._token_issued_at is None:
            return False
        return (now - self._token_issued_at) >= self._effective_token_life

    def should_renew(self, now: float) -> bool:
        """Historical threshold helper; does not select generator renewal."""
        if self._token_issued_at is None:
            return False
        elapsed = now - self._token_issued_at
        return elapsed >= self._effective_token_life * self.cfg.renewal_threshold

    def token_age(self, now: float) -> float:
        if self._token_issued_at is None:
            return 0.0
        return max(0.0, now - self._token_issued_at)

    def time_until_expiry(self, now: float) -> float:
        if self._token_issued_at is None:
            return self._effective_token_life
        return max(0.0, self._effective_token_life - (now - self._token_issued_at))

    def record_renewal(self, now: float) -> None:
        """Reset token lifetime clock on successful renewal."""
        self._token_issued_at = now
        self._renewal_count  += 1

    @property
    def renewal_count(self) -> int:
        return self._renewal_count

    # ── Nonce lifecycle ───────────────────────────────────────────────────────

    def record_nonce_issued(self, now: float) -> None:
        self._nonce_issued_at = now

    def is_nonce_expired(self, now: float) -> bool:
        if self._nonce_issued_at is None:
            return False
        return (now - self._nonce_issued_at) >= self.cfg.nonce_lifetime

    def nonce_age(self, now: float) -> float:
        if self._nonce_issued_at is None:
            return 0.0
        return max(0.0, now - self._nonce_issued_at)

    # ── Replay window ─────────────────────────────────────────────────────────

    def is_replay_violation(self, now: float) -> bool:
        """
        Historical age-window check; not synchronized replay semantics.

        Condition: token_presented_at - token_issued_at > replay_window,
        This comparison alone does not establish actual token-material reuse.
        """
        if self._token_issued_at is None or self._token_presented_at is None:
            return False
        age = self._token_presented_at - self._token_issued_at
        return age > self.cfg.replay_window

    # ── Retry policy ──────────────────────────────────────────────────────────

    def record_failure(self) -> None:
        self._retry_count += 1

    def can_retry(self) -> bool:
        return self._retry_count < self.cfg.max_retries

    def next_retry_delay(self) -> float:
        """Backoff for an already-selected RETRY, not retry selection."""
        base   = self.cfg.base_backoff * (self.cfg.backoff_factor ** self._retry_count)
        jitter = random.uniform(-base * 0.2, base * 0.2)
        return round(max(0.05, base + jitter), 3)

    def exhausted(self) -> bool:
        return self._retry_count >= self.cfg.max_retries

    @property
    def retry_count(self) -> int:
        return self._retry_count
