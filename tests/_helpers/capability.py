"""Asking "can this run here" in a way that cannot turn a broken build green.

``jt.capability`` deliberately refuses ``bool()``: a capability has no single
truth value. These helpers are the test-side translation, and the translation
is the whole point:

    absent / disabled  ->  SkipTest, with the machine-level reason attached
    failed             ->  AssertionError

The second line is the one that matters. Every previous spelling of these
checks turned a failure into a skip:

* ``tests/_helpers/cutt.py`` used to ``SkipTest`` on a cuTT *build failure*,
  so six tests reported "skipped" for months while transposes silently fell
  back to the built-in kernel. It was fixed by hand; this makes the fix the
  default for every capability, instead of something each helper re-derives.
* ``if not jt.has_cuda: raise SkipTest("no CUDA on this machine")`` says
  something false on a machine with eight GPUs and a CPU-only build. The skip
  is arguably right, the reason is not, and the reason is what a human reads
  when deciding whether a whole class of work is verifiable here.

So the skip message always states what the machine has, separately from what
the build enabled.
"""

import unittest

import jittor as jt


def _skip(capability):
    raise unittest.SkipTest(
        "%s %s is %s: %s"
        % (capability.kind, capability.name, capability.state.value,
           capability.reason))


def _refuse(capability):
    raise AssertionError(
        "%s %s was requested and did not come up: %s\n"
        "This is a broken build, not a missing feature, so it must not skip: "
        "a skip here is indistinguishable from a pass and is how the cuTT "
        "build stayed broken while its tests reported 'skipped'."
        % (capability.kind, capability.name, capability.reason))


def check_accelerator(name):
    """Return the accelerator capability, refusing to let a failure skip.

    Use when the caller wants to branch rather than skip. ``FAILED`` still
    raises, because no caller should ever branch around a broken build.
    """
    capability = jt.capability.accelerator(name)
    if capability.failed:
        _refuse(capability)
    return capability


def require_accelerator(name):
    """Skip unless accelerator ``name`` is usable in this build right now."""
    capability = check_accelerator(name)
    if not capability.enabled:
        _skip(capability)
    return capability


def check_library(name, load=True):
    capability = jt.capability.library(name, load=load)
    if capability.failed:
        _refuse(capability)
    return capability


def require_library(name, load=True):
    """Skip unless backend library ``name`` is loaded and exposes its ops.

    ``load=True`` by default: with ``load=False`` a library that is built on
    first use answers ``UNPROBED``, and skipping on that is exactly the "a
    reason no run could disprove" bug.
    """
    capability = check_library(name, load=load)
    if capability.unprobed:  # pragma: no cover - only with load=False
        raise AssertionError(
            "library %s is still unprobed, so this skip would be based on "
            "nothing; call require_library(%r) with load=True"
            % (name, name))
    if not capability.enabled:
        _skip(capability)
    return capability


def machine_has_accelerator(name):
    """Does the *machine* have it, regardless of how this build was configured?

    The question to ask before reporting that something is unverifiable here.
    ``True`` with ``require_accelerator`` skipping means the build is the
    variable, not the hardware.
    """
    return jt.capability.accelerator(name).present


__all__ = ["check_accelerator", "require_accelerator", "check_library",
           "require_library", "machine_has_accelerator"]
