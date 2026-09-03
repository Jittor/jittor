"""Autograd policy selection.

The compiled runtime implements only generic gradient decisions. Higher-level
APIs choose a named policy rather than leaking a compatibility mode flag into
the graph core.
"""

from jittor import core as _core


class AutogradPolicy:
    __slots__ = (
        "_frozen",
        "name",
        "stop_outputs_when_inputs_stopped",
        "preserve_requires_grad_on_assignment",
    )

    def __init__(
        self,
        name,
        stop_outputs_when_inputs_stopped=False,
        preserve_requires_grad_on_assignment=False,
    ):
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "stop_outputs_when_inputs_stopped", bool(
            stop_outputs_when_inputs_stopped
        ))
        object.__setattr__(self, "preserve_requires_grad_on_assignment", bool(
            preserve_requires_grad_on_assignment
        ))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("AutogradPolicy objects are immutable")
        object.__setattr__(self, name, value)

    def __repr__(self):
        return "AutogradPolicy(%r)" % self.name


NATIVE = AutogradPolicy("native")
EXPLICIT_REQUIRES_GRAD = AutogradPolicy(
    "explicit_requires_grad",
    stop_outputs_when_inputs_stopped=True,
    preserve_requires_grad_on_assignment=True,
)


def set_policy(policy):
    """Select ``policy`` for subsequently constructed graph nodes."""
    if not isinstance(policy, AutogradPolicy):
        raise TypeError("policy must be an AutogradPolicy")
    _core._set_autograd_policy(
        policy.stop_outputs_when_inputs_stopped,
        policy.preserve_requires_grad_on_assignment,
    )


def get_policy():
    """Return the active policy, preserving canonical preset identity."""
    bits = _core._get_autograd_policy()
    if bits == 0:
        return NATIVE
    if bits == 3:
        return EXPLICIT_REQUIRES_GRAD
    return AutogradPolicy(
        "custom",
        stop_outputs_when_inputs_stopped=bool(bits & 1),
        preserve_requires_grad_on_assignment=bool(bits & 2),
    )


class policy_scope:
    """Temporarily select an autograd policy and restore it on exit."""

    def __init__(self, policy):
        if not isinstance(policy, AutogradPolicy):
            raise TypeError("policy must be an AutogradPolicy")
        self.policy = policy
        self._backups = []

    def __enter__(self):
        self._backups.append(get_policy())
        set_policy(self.policy)
        return self.policy

    def __exit__(self, *exc):
        set_policy(self._backups.pop())


__all__ = [
    "AutogradPolicy",
    "EXPLICIT_REQUIRES_GRAD",
    "NATIVE",
    "get_policy",
    "policy_scope",
    "set_policy",
]
