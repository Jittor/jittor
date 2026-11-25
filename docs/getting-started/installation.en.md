# Installation

[Section contents](index.en.md) · [中文](installation.md)

Run a CPU example before configuring an accelerator. Jittor compiles operators on your machine, so it needs a C++ compiler as well as Python. See [`pyproject.toml`](https://github.com/Jittor/jittor/blob/2.0-refactor/pyproject.toml) for this checkout's dependency declarations.

## Prepare a Linux environment

These commands target Debian/Ubuntu and create an environment with the local `python3`. It must meet the source package’s Python requirement; repository maintenance uses Python 3.11:

```bash
sudo apt install python3-dev python3-venv libomp-dev g++ build-essential
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/getting-started"
python3 -m venv "$STATE/venv"
. "$STATE/venv/bin/activate"
```

macOS needs clang and OpenMP (`brew install libomp`). See [platform support](../guides/hardware/platform-support.en.md) for validation coverage on other platforms. The Linux commands above do not apply directly to Windows.

## Choose an installation source

To use a published release:

```bash
python -m pip install jittor
```

To use the implementation described by this branch's documentation, run this from an existing repository checkout:

```bash
python -m pip install --config-settings editable_mode=compat -e .
```

Choose one installation method. The PyPI release may not contain every feature in the current branch.

## Verify CPU computation

```bash
python - <<'PYCPU'
import jittor as jt
jt.flags.use_cuda = 0
x = jt.array([1.0, 2.0, 3.0])
print((x * 2).numpy())
PYCPU
```

The result should be `[2. 4. 6.]`. The first run may compile code. If it fails, check the compiler, headers or dependencies named in the error. For further installation checks, run `python -m jittor.selftest`.

## Configure ROCm / HIP

Install the device driver and the complete HIP development toolchain, including `hipcc`, the HIP
runtime, the hipBLAS and rocPRIM headers and libraries, and the hipRAND device headers. The Python package does not install these
system dependencies. For Hygon DTK, set `ROCM_HOME` to the SDK root, rather than its `hip/` subdirectory.

```bash
export JT_BACKEND=rocm
export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
export JITTOR_HOME="${STATE:-../jittor-lab/_state/getting-started}/rocm"
DISABLE_MULTIPROCESSING=1 python - <<'PYHIP'
import jittor as jt
assert jt.compiler.has_rocm
with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
    result = (jt.array([1.0, 2.0, 3.0]) * 2).sum()
    result.sync()
    assert result.location() == "device"
    assert result.item() == 12.0
    print("HIP device computation passed")
PYHIP
```

`use_cuda` is the runtime's general accelerator switch; `JT_BACKEND=rocm` selects the actual backend.
This check forbids CPU fallback. Configuration reports a missing HIP runtime development library as
`libamdhip64.so`; missing hipBLAS or rocPRIM dependencies also fail explicitly. MIOpen and RCCL
providers remain unimplemented. This check does not establish convolution-library, multi-device
communication, or arbitrary-model support; see [platform support](../guides/hardware/platform-support.en.md)
for the validation scope.

## Add Torch compatibility when needed

`jittor-compat` is a separate distribution. For this checkout, install both projects from the repository root in a new dedicated environment:

```bash
python -m pip install --config-settings editable_mode=compat -e .
python -m pip install -e ./compat
```

The compatibility package provides the `torch` entry point. Keep genuine PyTorch in a separate environment because both runtimes use that module name. Matching wheels declare their core version requirements in package metadata. Continue with the [Torch compatibility guide](../compatibility/torch.en.md).

## Next

Complete the [quickstart](quickstart.en.md), then use the [device placement tutorial](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/device_placement.md) to configure a device. CUDA needs a suitable toolkit and driver; NPU and other backends need their own environments. A successful CPU example does not establish support for another chip or a downstream library.
