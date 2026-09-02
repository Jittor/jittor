"""Central policy for torch-compat APIs whose signature exists but whose
semantics are missing.

The compatibility layer historically installed a number of ``torch`` APIs that
accept every argument the real API accepts, return a plausible value, and do
nothing.  ``torch.autocast`` was a no-op context manager, ``load_state_dict``
returned an empty ``IncompatibleKeys`` no matter what was loaded, and
``dcp.save()`` wrote no bytes yet reported success.  Every one of those failed
*silently*: the script ran to completion and produced wrong numbers.

This module makes that class of gap loud by default::

    from jittor.compat.stub_policy import unimplemented

    unimplemented("torch.distributed.checkpoint.save",
                  "report success without writing the checkpoint")

raises ``NotImplementedError`` naming the API and stating the consequence.  A
user who knowingly wants the old behaviour opts in explicitly::

    JITTOR_TORCH_ALLOW_STUB=1 python train.py

or programmatically::

    import jittor as torch
    torch.compat_allow_stub(True)

With the escape hatch on, ``unimplemented`` warns once per API and returns the
``stub_result`` the caller supplies, reproducing the pre-2.0 behaviour.

``degraded`` is the weaker sibling: it is for APIs that do work, but not the
way torch works (worker threads instead of worker processes, for example).  It
never raises; it warns once so the difference is on the record.
"""

from __future__ import absolute_import

import os
import warnings

__all__ = [
    "allow_stub",
    "set_allow_stub",
    "unimplemented",
    "degraded",
    "unimplemented_callable",
    "reset_warned",
    "registry",
    "record_unimplemented",
    "ENV_VAR",
]

ENV_VAR = "JITTOR_TORCH_ALLOW_STUB"

# None => consult the environment; True/False => explicit process-level override.
_override = None

# APIs that already emitted their one warning, so a training loop calling a
# degraded API every step does not print a million lines.
_warned = set()

# Every API declared through this module, with its stated consequence.
# tests/compat/torch/test_torch_compat_unimplemented.py renders this into the
# generated "unimplemented API list" the refactor plan asks for.
_registry = {}


def _env_allows():
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return False
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def allow_stub():
    """True when silent-stub fallback has been explicitly enabled."""
    if _override is not None:
        return bool(_override)
    return _env_allows()


def set_allow_stub(value):
    """Enable/disable stub fallback for this process.

    ``set_allow_stub(None)`` returns to consulting ``JITTOR_TORCH_ALLOW_STUB``.
    Returns the previous override so callers can restore it.
    """
    global _override
    previous = _override
    _override = None if value is None else bool(value)
    return previous


def reset_warned():
    """Forget which APIs have already warned (used by tests)."""
    _warned.clear()


def registry():
    """{api name: {"effect", "hint"}} for every API declared here."""
    return dict(_registry)


def record_unimplemented(api, effect, hint=None):
    """Declare an unimplemented API without calling it.

    Used at install time so the generated list also covers APIs that a given
    run never happens to touch.
    """
    _registry[str(api)] = {"effect": str(effect),
                           "hint": None if hint is None else str(hint)}


def _message(api, effect, hint):
    msg = ("Jittor's torch compatibility layer does not implement %s; "
           "running it as a no-op would %s." % (api, effect))
    if hint:
        msg += " " + hint
    msg += (" Set %s=1 (or call torch.compat_allow_stub(True)) to fall back to "
            "the previous silent no-op behaviour at your own risk." % ENV_VAR)
    return msg


_MISSING = object()


def unimplemented(api, effect, hint=None, stub_result=_MISSING,
                  error_type=NotImplementedError):
    """Refuse to pretend an API works.

    :param api: dotted name as the user typed it, e.g. ``"torch.autocast"``.
    :param effect: what goes wrong when the call silently succeeds, phrased to
        complete "running it as a no-op would ..." (e.g. "train the model in
        float32 while the script believes it is in mixed precision").
    :param hint: optional actionable sentence appended to the message.
    :param stub_result: value returned when the escape hatch is on.  Omit to
        return ``None``.
    :raises NotImplementedError: unless :func:`allow_stub`.
    """
    record_unimplemented(api, effect, hint)
    if not allow_stub():
        raise error_type(_message(api, effect, hint))
    if api not in _warned:
        _warned.add(api)
        warnings.warn(
            "%s=1: %s is a no-op stub in Jittor's torch compatibility layer; "
            "it would %s." % (ENV_VAR, api, effect),
            RuntimeWarning, stacklevel=3)
    return None if stub_result is _MISSING else stub_result


def unimplemented_callable(api, effect, hint=None, stub_result=_MISSING):
    """Build a callable that :func:`unimplemented`\\ s when invoked.

    Convenient for the many ``mod.fn = lambda *a, **k: None`` install lines.
    The API is registered immediately, so it shows up in the generated list
    even when nothing calls it.
    """
    record_unimplemented(api, effect, hint)

    def _raise(*args, **kwargs):
        return unimplemented(api, effect, hint, stub_result)

    _raise.__name__ = str(api).rsplit(".", 1)[-1]
    _raise.__qualname__ = _raise.__name__
    _raise.__doc__ = _message(api, effect, hint)
    _raise._jittor_unimplemented = api
    return _raise


def degraded(api, difference, hint=None):
    """Warn once that an API works, but not the way torch works.

    Never raises: the call proceeds with the approximation.
    """
    record_approximate(api, difference, hint)
    if api in _warned:
        return
    _warned.add(api)
    msg = ("%s is approximated by Jittor's torch compatibility layer: %s."
           % (api, difference))
    if hint:
        msg += " " + hint
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


_approximate = {}


def record_approximate(api, difference, hint=None):
    """Declare an API that is implemented, but not identically to torch."""
    _approximate[str(api)] = {"effect": str(difference),
                              "hint": None if hint is None else str(hint)}


def approximate_registry():
    """{api name: {"effect", "hint"}} for every approximated API."""
    return dict(_approximate)
