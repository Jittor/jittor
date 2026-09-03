"""Automatic differentiation APIs."""

from . import functional as functional
from .functional import jvp, vjp
from .policy import (
    AutogradPolicy,
    EXPLICIT_REQUIRES_GRAD,
    NATIVE,
    get_policy,
    policy_scope,
    set_policy,
)

__all__ = [
    "AutogradPolicy",
    "EXPLICIT_REQUIRES_GRAD",
    "NATIVE",
    "functional",
    "get_policy",
    "jvp",
    "policy_scope",
    "set_policy",
    "vjp",
]
