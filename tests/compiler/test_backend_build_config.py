"""Configured CPU bootstrap must never invoke CUDA discovery or installation."""

from _helpers.child_process import run_child_script


def test_explicit_cpu_import_skips_cuda_services(tmp_path):
    script = r'''
import json
import jittor_utils.install_cuda as installer

def forbidden(*args, **kwargs):
    raise AssertionError("explicit CPU bootstrap invoked a CUDA service")

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
print("CPU_BUILD_CONFIG=" + json.dumps({"cuda_services": 0, "backend": "cpu"}))
'''
    result = run_child_script(
        script, directory=tmp_path, name="explicit_cpu_backend", timeout=600,
        env={"JT_BACKEND": "cpu", "nvcc_path": "/must/not/probe/nvcc",
             "JTCUDA_AUTO_INSTALL": "1"}, without_torch_mode=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert 'CPU_BUILD_CONFIG={"cuda_services": 0, "backend": "cpu"}' in result.stdout
