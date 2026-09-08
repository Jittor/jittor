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
import functools
import inspect

def _native_backend(backend=None):
    if backend is None:
        import jittor
        return jittor
    return backend


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


def check_accelerator(name, *, backend=None):
    """Return the accelerator capability, refusing to let a failure skip.

    Use when the caller wants to branch rather than skip. ``FAILED`` still
    raises, because no caller should ever branch around a broken build.
    """
    backend = _native_backend(backend)
    capability = backend.introspection.capabilities.backend(name)
    if capability.failed:
        _refuse(capability)
    return capability


def require_accelerator(name):
    """Skip unless accelerator ``name`` is usable in this build right now."""
    capability = check_accelerator(name)
    if not capability.enabled:
        _skip(capability)
    return capability


def any_accelerator_enabled(*, backend=None):
    """Legacy generic accelerator sweeps must also retain ACL/ROCm coverage."""
    backend = _native_backend(backend)
    queries = backend.introspection.capabilities
    names = tuple(name for name in queries.registered_backends() if name != "cpu")
    if not names:
        # No registered accelerator may mean an intentionally CPU-only build
        # or a requested backend that failed. Do not collapse the latter.
        names = tuple(name for name in queries.backends() if name != "cpu")
    enabled = False
    for name in names:
        enabled = check_accelerator(name, backend=backend).enabled or enabled
    return enabled


def check_library(name, load=True, *, backend=None):
    backend = _native_backend(backend)
    capability = backend.introspection.capabilities.library(name)
    if capability.unprobed and load:
        # Loading is an explicit test prerequisite, outside read-only
        # introspection. Never relabel an unprobed or failed loader as absent.
        capability = backend.capability.library(name, load=True)
        if capability.failed:
            _refuse(capability)
        capability = backend.introspection.capabilities.library(name)
    if capability.failed:
        _refuse(capability)
    return capability


def library_enabled(name, *, backend=None):
    capability = check_library(name, load=True, backend=backend)
    if capability.unprobed:
        raise AssertionError("library %s remained unprobed after explicit initialization" % name)
    return capability.enabled


def library_required(name, *, backend=None):
    """Defer an optional library prerequisite until execution, not collection."""
    def decorate(target):
        if inspect.isclass(target):
            setup = target.setUpClass.__func__
            @classmethod
            def checked_setup(cls):
                capability = check_library(name, load=True, backend=backend)
                if capability.unprobed:
                    raise AssertionError("library %s remained unprobed" % name)
                if not capability.enabled:
                    _skip(capability)
                setup(cls)
            target.setUpClass = checked_setup
            return target
        @functools.wraps(target)
        def checked(*args, **kwargs):
            capability = check_library(name, load=True, backend=backend)
            if capability.unprobed:
                raise AssertionError("library %s remained unprobed" % name)
            if not capability.enabled:
                _skip(capability)
            return target(*args, **kwargs)
        return checked
    return decorate


def device_count(backend_name, *, backend=None):
    backend = _native_backend(backend)
    inventory = backend.introspection.capabilities.devices(backend_name)
    if inventory.capability.failed:
        _refuse(inventory.capability)
    if inventory.count is None:
        raise AssertionError("device inventory for %s is unprobed" % backend_name)
    return inventory.count


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
    return _native_backend().capability.accelerator(name).present


__all__ = ["check_accelerator", "require_accelerator", "check_library",
           "require_library", "machine_has_accelerator"]
