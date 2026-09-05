"""Explicit policy for backend fallback decisions.

The policy is intentionally independent of the native flags object.  It gives
the registry and future dispatch layer one validated decision point before the
legacy ``flags.use_*`` routing is migrated.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FallbackMode(str, Enum):
    ERROR = "error"
    WARN = "warn"
    ALLOW = "allow"


class FallbackError(RuntimeError):
    """Raised when a backend miss is forbidden by policy."""


@dataclass(frozen=True)
class FallbackDecision:
    """A query result suitable for logging and dispatch diagnostics."""

    mode: FallbackMode
    operator: str
    backend: str
    reason: str
    fallback_backend: Optional[str]

    @property
    def allowed(self) -> bool:
        return self.mode is not FallbackMode.ERROR


class BackendFallbackPolicy:
    """Validate and evaluate backend fallback behavior.

    ``warn`` is the runtime-compatible default; test harnesses can construct
    ``error`` to make an accidental fallback fail closed.
    """

    _MODES = frozenset(FallbackMode)

    def __init__(self, mode: str = FallbackMode.WARN):
        self._mode = self._coerce(mode)

    @staticmethod
    def _coerce(mode: str) -> FallbackMode:
        if isinstance(mode, FallbackMode):
            return mode
        try:
            return FallbackMode(mode)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in FallbackMode)
            raise ValueError("backend fallback must be one of: %s" % choices) from exc

    @property
    def mode(self) -> str:
        return self._mode.value

    def set_mode(self, mode: str) -> None:
        self._mode = self._coerce(mode)

    def decide(self, operator: str, backend: str, reason: str,
               fallback_backend: Optional[str] = None) -> FallbackDecision:
        if not operator or not backend:
            raise ValueError("operator and backend are required for fallback")
        if fallback_backend == backend:
            raise ValueError("fallback backend must differ from requested backend")
        decision = FallbackDecision(
            self._mode, operator, backend, reason, fallback_backend)
        if not decision.allowed:
            target = "%s/%s" % (backend, operator)
            raise FallbackError("backend fallback denied for %s: %s" % (target, reason))
        return decision

