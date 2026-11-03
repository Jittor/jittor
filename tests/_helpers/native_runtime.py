"""Check native execution ownership without banning definition-only imports.

Importing compatibility definitions or genuine Torch does not activate the
Jittor Torch frontend. Inspect its actual publication/install state instead;
this helper never installs, resets, or repairs the runtime being tested.
"""

from contextlib import contextmanager
import sys


def assert_native_runtime(jt):
    """Require native Var/graph policy and no active or partial Torch install."""
    state = jt.runtime.service_state("jittor.torch.activation", factory=None)
    torch = sys.modules.get("torch")
    assert torch is not jt, "native test is running through the legacy Torch alias"
    if torch is not None:
        assert vars(torch).get("_torch_owner") is not jt, "Jittor Torch frontend is published"
        assert not vars(torch).get("_jittor_torch_shim_placeholder"), "partial Torch shim publication"
    if state is not None:
        assert not state.get("installed"), "Jittor Torch frontend is installed"
        assert state.get("phase", "inactive") in ("inactive", "failed"), (
            "Jittor Torch activation is not inactive", state.get("phase"))

    targets = [jt, torch]
    if state is not None:
        targets.append(state.get("installation_target"))
    for target in targets:
        if target is None:
            continue
        attributes = vars(target)
        assert not attributes.get("_torch_compat_install_complete"), "completed Torch install"
        assert not attributes.get("_torch_compat_install_steps"), "partial Torch install steps"
        context = attributes.get("_torch_compat_install_context")
        if context is not None:
            assert not context.complete, "completed Torch install context"
            assert not context.markers, "partial Torch install context"

    policy = jt.core._get_autograd_policy()
    assert policy == 0, ("native graph policy changed", policy)
    original = sys.modules.get("jittor._core.var")
    assert original is not None, "canonical native Var owner is missing"
    assert jt.Var is jt.core.Var is vars(original).get("Var"), "native Var identity changed"
    return {"autograd_policy": policy, "native_var": True,
            "torch_imported": torch is not None, "activation_installed": False}


@contextmanager
def native_execution_scope(jt):
    """Guard one native test and finish its graph inside the fallback guard."""
    from jittor._runtime.fallback import forbid_backend_fallbacks

    assert_native_runtime(jt)
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


def assert_no_new_fallbacks(before, after):
    """Validate a protocol-local counter delta, retaining absolute diagnostics."""
    assert type(before) is int and type(after) is int, "fallback counters must be integers"
    assert 0 <= before <= after, ("invalid fallback counter interval", before, after)
    delta = after - before
    assert delta == 0, ("new backend fallback during native protocol", before, after, delta)
    return {"before": before, "after": after, "delta": delta}
