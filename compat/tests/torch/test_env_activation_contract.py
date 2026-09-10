"""``JITTOR_TORCH_SHIM=1`` is the whole switch: no PYTHONPATH, no deployment.

The environment variable is the documented way to select Torch mode, and what
it selects has changed. It used to pick an import-time path that installed the
Torch API onto the native module, so `torch is jittor`; that path was removed,
and `install(jittor)` now refuses. The variable today composes the independent
`TorchNamespace` -- the same object `shim.activate()` returns.

`docs/compatibility/torch-shim.md` still described the old meaning, which is
the kind of claim only a test can hold: prose about a runtime behaviour has
nothing checking it, so it keeps whatever it said on the day the behaviour
changed underneath it.

These run in child processes because activation is process-global and
idempotent. Asserting it in-process would only observe whatever the session's
own mode already was.
"""

from _helpers import child_process

import json


ENV_ONLY = {"JITTOR_TORCH_SHIM": "1"}

PROBE = """
import json, os, sys
import jittor as jt
import torch
print("RESULT" + json.dumps({
    "is_jittor": torch is jt,
    "tensor_is_var": torch.Tensor is jt.Var,
    "namespace_type": type(torch).__name__,
    "works": float((torch.ones(2, 2) + 1).sum().item()),
    "path_had_resources": any("shim/resources" in entry for entry in sys.path),
}))
"""


def _probe(env):
    finished = child_process.run_python_child(
        ["-c", PROBE], env=env, timeout=600, merge_stderr=True)
    assert finished.returncode == 0, finished.stdout[-2000:]
    marker = finished.stdout.rindex("RESULT")
    return json.loads(finished.stdout[marker + len("RESULT"):].splitlines()[0])


def test_the_environment_variable_alone_selects_the_independent_namespace():
    result = _probe(dict(ENV_ONLY))
    # The claim the stale documentation got wrong, in both halves: the mode is
    # on, and it is the modern one rather than the removed alias mode.
    assert result["namespace_type"] == "TorchNamespace"
    assert not result["is_jittor"]
    assert not result["tensor_is_var"]
    assert result["works"] == 8.0


def test_no_path_configuration_is_needed():
    # Source checkouts reach the shim by putting `compat/shim/resources` on
    # PYTHONPATH, which is a second, undocumented way in. If the variable only
    # worked when that directory happened to be on the path, the variable would
    # not be the switch -- so this pins that it is not on the path when the
    # probe succeeds.
    result = _probe(dict(ENV_ONLY))
    assert not result["path_had_resources"]
    assert result["namespace_type"] == "TorchNamespace"


def test_importing_torch_first_still_needs_a_deployed_package():
    """The one thing the variable cannot do, stated so nobody expects it to.

    An environment variable cannot put a module on ``sys.path``. Reaching
    ``import torch`` as a process's *first* import therefore still needs a
    deployed ``torch`` package (or that directory on PYTHONPATH); the variable
    covers everything after ``import jittor``.

    The failure a stale deployment produces used to say "use shim.activate()
    and import torch", which reads as nonsense to someone who just imported
    torch. It now names the file, and that is what is asserted -- an error
    message is only useful if it survives.
    """
    finished = child_process.run_python_child(
        ["-c", "import torch"], env=dict(ENV_ONLY), timeout=600,
        merge_stderr=True)
    if finished.returncode == 0:
        return  # a current deployment is present; nothing to diagnose
    assert "deployed shim at" in finished.stdout, finished.stdout[-1500:]
    assert "redeploy" in finished.stdout


def test_leaving_it_unset_stays_native():
    finished = child_process.run_python_child(
        ["-c", "import jittor as jt; import sys;"
               " print('TORCH_PRESENT', 'torch' in sys.modules)"],
        env={"JITTOR_TORCH_SHIM": "0"}, timeout=600, merge_stderr=True,
        without_torch_mode=True)
    assert finished.returncode == 0, finished.stdout[-2000:]
    # Native startup must not compose the Torch surface: the variable is the
    # switch in both directions, and a mode that turns itself on is not one.
    assert "TORCH_PRESENT False" in finished.stdout
