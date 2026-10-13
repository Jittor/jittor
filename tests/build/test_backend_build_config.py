"""Configured CPU bootstrap must never invoke CUDA discovery or installation."""

from unittest import mock

from _helpers.child_process import run_child_script


def test_user_nvcc_flags_reach_backend_cuda_sources(tmp_path, monkeypatch):
    """JT_BUILD_NVCC_FLAGS applies to a backend's .cu files as to the core's.

    `compile_backend_sources` passed only the common flags on, so a flag the
    user set for nvcc silently missed `nan_checker.cu` and the like.
    """
    import jittor as jt
    from jittor_utils.build_config import BuildConfig, BuildSource
    source = tmp_path / "probe.cu"
    source.write_text("")
    config = BuildConfig(cache_path=str(tmp_path), nvcc_path="nvcc",
                         backend_sources=(BuildSource(str(source), language="cuda"),
                                          BuildSource(str(source), language="cxx")))
    monkeypatch.setenv("JT_BUILD_NVCC_FLAGS", "-DUSER_NVCC_FLAG_PROBE")
    with mock.patch.object(jt.compiler.jit_utils, "run_cmds") as run_cmds:
        jt.compiler.compile_backend_sources(config, "-DCOMMON")
    cuda_command, cxx_command = run_cmds.call_args[0][0]
    assert "-DUSER_NVCC_FLAG_PROBE" in cuda_command, cuda_command
    assert "-DCOMMON" in cuda_command
    assert "-DUSER_NVCC_FLAG_PROBE" not in cxx_command, cxx_command


def test_explicit_cpu_import_skips_cuda_services(tmp_path):
    script = r'''
import atexit
import json
import os
import sys
import traceback

import jittor_utils.install_cuda as installer

CALLS = []

def forbidden(*args, **kwargs):
    # Recorded, not raised -- and that is the point of the shape.
    #
    # This function stands in for five entry points, and something reaching one
    # of them from a destructor or an atexit hook would have raised *during
    # interpreter shutdown*, which is where a teardown exception turns into
    # glibc reporting heap corruption (KI-COMPILER-007 suspects exactly that).
    # A probe must not be able to kill the interpreter it is probing: the
    # contract is unchanged -- the import path must not reach these -- and it is
    # asserted below, with the atexit hook failing the process if a call arrives
    # after the last statement instead of letting an exception escape shutdown.
    CALLS.append("".join(traceback.format_stack()[-6:]))
    print("CUDA_SERVICE_CALLED\n" + CALLS[-1], file=sys.stderr, flush=True)
    return None

installer.has_installation = forbidden
installer.install_cuda = forbidden
installer.get_cuda_driver = forbidden
installer.get_cuda_driver_win = forbidden
installer.get_cuda_wheel_stack = forbidden
import jittor as jt
assert jt.compiler.build_config.backend == "cpu"
assert jt.compiler.nvcc_path == ""
assert not jt.compiler.build_config.has_cuda
assert not jt.compiler.backend_modules
assert CALLS == [], (
    "the explicit CPU bootstrap reached a CUDA service before the assertions: "
    + "\n".join(CALLS))
print("CPU_BUILD_CONFIG=" + json.dumps({"cuda_services": len(CALLS), "backend": "cpu"}))
print("CPU_BUILD_CONFIG_END", flush=True)


def _fail_if_a_late_call_arrived():
    if not CALLS:
        return
    print("the CPU bootstrap reached a CUDA service during teardown; the stack "
          "is above", file=sys.stderr, flush=True)
    # A deliberate, diagnosable failure: raising here would be the hazard this
    # shape exists to avoid, and exiting from an atexit hook reports it as the
    # non-zero status the caller checks.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(3)


atexit.register(_fail_if_a_late_call_arrived)
'''
    result = run_child_script(
        script, directory=tmp_path, name="explicit_cpu_backend", timeout=600,
        env={"JT_BACKEND": "cpu", "nvcc_path": "/must/not/probe/nvcc",
             "JTCUDA_AUTO_INSTALL": "1"}, without_torch_mode=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert 'CPU_BUILD_CONFIG={"cuda_services": 0, "backend": "cpu"}' in result.stdout
    # The last line means the import path finished; the exit status above means
    # no call arrived after it either.
    assert "CPU_BUILD_CONFIG_END" in result.stdout, result.stdout[-2000:]
