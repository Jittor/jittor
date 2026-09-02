"""Policy for parameters that are accepted for signature compatibility but not honoured.

A number of Jittor APIs take a parameter purely so that torch-shaped call sites
keep working, and then never read it.  ``nn.relu(x, inplace=True)`` allocates a
new var, ``sort(x, stable=True)`` runs an unstable sort, and
``nn.instance_norm(x, running_mean=m)`` neither updates nor uses ``m``.  All of
them return without a word, so the caller only finds out from the numbers.

This module puts every such parameter into one of two buckets.

Which bucket: one question
--------------------------

**Would honouring the parameter change any number the caller can observe?**

*No* -> :func:`ignored`.  The returned values already are what the parameter
asks for; only a non-semantic promise goes unkept -- a memory saving, a driver
choice, an ordering that is not part of the contract.  The caller's program is
still correct, it just does not get the resource win it asked for.  Warns once
per process per API+parameter and never raises: raising would break working
code over a performance hint.

*Yes* -> :func:`unsupported`.  The caller asked for behaviour B and is quietly
given behaviour A.  Raises ``NotImplementedError``, because a wrong answer that
arrives without a word is worse than no answer.

Worked examples of the boundary, because this is where reuse goes wrong:

``nn.relu(x, inplace=True)`` -> **ignored**.  ``inplace`` is a memory hint.  The
returned tensor is identical either way; only the peak-memory promise is broken.

``linalg.inv_ex(a)`` (``check_errors=False``) -> **reported on every default
call, because no value of the parameter would fix it**.  The docstring tells
callers to build a validity mask from ``info == 0``; ``info`` is hard-coded to
zero, so a singular matrix never gets a non-zero entry and the caller silently
treats garbage as valid.  That is a changed number, not a missed optimisation.

``sort(x, stable=True)`` -> **unsupported**.  A stable sort and an unstable one
return *different index tensors* for tied keys, and anything downstream that
depends on tie order silently diverges.

``topk(x, k, sorted=False)`` -> **neither; do not route it here at all**.
``sorted=False`` asks for a *weaker* guarantee than what is returned, and torch
leaves that ordering unspecified, so returning sorted output satisfies the
request.  Nothing is withheld, so there is nothing to report -- registering it
would teach users to ignore these warnings.

The escape hatch
----------------

A caller who knowingly wants the old (silently ignoring) behaviour opts in::

    JITTOR_ALLOW_UNSUPPORTED_ARGS=1 python train.py

or ``jittor._arg_policy.set_allow_unsupported(True)``.  With the hatch on,
:func:`unsupported` **still warns once per process per API+parameter** and then
returns.  It never degrades to silence: the point of the hatch is to unblock a
run, not to restore the state where nobody could tell.  Off by default, one
warning per API when on -- the same contract as
:mod:`jittor.compat.stub_policy`.

Relationship to :mod:`jittor.compat.stub_policy`
------------------------------------------------

These two solve the two sides of the same problem and are deliberately kept
apart.  Do not add a third; extend whichever of the two fits.

* :mod:`jittor._arg_policy` (this module) -- Jittor's own public APIs, and the
  *parameters* of those APIs that are accepted and then never read.  Env var
  ``JITTOR_ALLOW_UNSUPPORTED_ARGS``.
* :mod:`jittor.compat.stub_policy` -- the torch compatibility layer, and whole
  torch-named *APIs* installed as no-ops (``torch.autocast``, ``dcp.save``).
  Env var ``JITTOR_TORCH_ALLOW_STUB``.

Different audiences (a Jittor user versus a script written against torch) and
different default policies, so merging them would make both harder to reason
about.  They do share :func:`env_flag_enabled`: disagreeing about what ``FOO=0``
means would turn one of the two hatches on by accident.

Both buckets record into :func:`registry`, so the set of unhonoured parameters
can be enumerated by a test instead of by grepping.

This module deliberately imports nothing from Jittor, so any submodule can
import it during package initialisation.
"""

import os
import warnings

__all__ = [
    "ENV_VAR",
    "env_flag_enabled",
    "allow_unsupported",
    "set_allow_unsupported",
    "ignored",
    "unsupported",
    "registry",
    "reset_warned",
]

ENV_VAR = "JITTOR_ALLOW_UNSUPPORTED_ARGS"

#: None => consult the environment; True/False => explicit process-level override.
_override = None

#: (api, parameter) pairs that already emitted their one warning, so a training
#: loop calling a degraded API every step does not print a million lines.
_warned = set()

#: {(api, parameter): (kind, consequence)} for every parameter declared here.
_registry = {}


def env_flag_enabled(name):
    """True when environment variable ``name`` is set to something truthy.

    Shared with :mod:`jittor.compat.stub_policy`, which gates its own escape
    hatch on its own variable: the two policies stay separate, but they have to
    agree on what "off" looks like, or ``FOO=0`` turns one of them on.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def allow_unsupported():
    """True when the escape hatch has been enabled for this process."""
    if _override is not None:
        return bool(_override)
    return env_flag_enabled(ENV_VAR)


def set_allow_unsupported(value):
    """Enable/disable the escape hatch. ``None`` returns to consulting the env.

    Returns the previous override so callers can restore it.
    """
    global _override
    previous = _override
    _override = None if value is None else bool(value)
    return previous


def reset_warned():
    """Forget which APIs already warned (tests want each case to warn again)."""
    _warned.clear()


def registry():
    """``{(api, parameter): (kind, consequence)}`` for every declared parameter."""
    return dict(_registry)


def _record(api, parameter, kind, consequence):
    _registry[(api, parameter)] = (kind, consequence)


def _warn_once(api, parameter, message):
    key = (api, parameter)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, UserWarning, stacklevel=4)


def ignored(api, parameter, value, consequence):
    """``parameter`` is accepted but not honoured; the result is still correct.

    Use this only when honouring the parameter would change **no number the
    caller can observe** -- a memory hint, a driver choice, an ordering torch
    itself leaves unspecified.  If the numbers would differ it belongs in
    :func:`unsupported`, not here.

    :param api: dotted name of the API, e.g. ``"jittor.nn.relu"``.
    :param parameter: the parameter name.
    :param value: the value the caller passed (shown in the warning).
    :param consequence: what the caller does not get, in one clause.
    """
    _record(api, parameter, "ignored", consequence)
    _warn_once(api, parameter, (
        "{api}: {parameter}={value!r} is accepted for signature compatibility "
        "but not honoured -- {consequence}. The returned values are still "
        "correct. (warned once per process)"
    ).format(api=api, parameter=parameter, value=value, consequence=consequence))


def unsupported(api, parameter, value, consequence):
    """``parameter`` is not honoured and honouring it would change the result.

    Use this whenever the caller would get different numbers, different indices
    or different control flow than the parameter asks for; staying quiet about
    that is a wrong answer, not a missing optimisation.

    Raises ``NotImplementedError`` unless the escape hatch is on, in which case
    it warns once per process per API+parameter and returns.  The hatch never
    restores silence.
    """
    _record(api, parameter, "unsupported", consequence)
    message = (
        "{api}: {parameter}={value!r} is not implemented -- {consequence}. "
        "Silently ignoring it would return a different answer than requested. "
        "Set {env}=1 to keep the old (silently ignoring) behaviour."
    ).format(api=api, parameter=parameter, value=value,
             consequence=consequence, env=ENV_VAR)
    if allow_unsupported():
        _warn_once(api, parameter, message)
        return
    raise NotImplementedError(message)
