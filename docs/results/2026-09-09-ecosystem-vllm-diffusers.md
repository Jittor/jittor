# 2026-09-09 ecosystem smoke results

This note records the available host evidence for the Torch-compatible ecosystem
request. It does not claim vLLM or TRELLIS support without those packages.

## Host evidence

The current Python environment imports `torch`, `diffusers`, `transformers`, and
`mmcv`. A real PyTorch `mmcv.cnn.ConvModule(3, 4, 3, padding=1)` forward and
backward completed with output shape `(2, 4, 8, 8)` and a finite input gradient
norm. A real PyTorch `diffusers.DDPMScheduler.add_noise` smoke also completed
for a `(2, 4)` tensor.

The isolated PyTorch 3.12 target now contains `diffusers==0.35.1`. With the
incompatible torchvision package temporarily hidden, a real PyTorch compact
`UNet2DModel` forward/backward completed on CPU: output `(1, 4, 8, 8)` and
finite input-gradient norm `0.0134599`. This is a PyTorch smoke, not Jittor
parity; the torchvision package is restored after each run.

## Compatibility status

The current environment has no importable `vllm` or TRELLIS package, so their
compatibility and PyTorch forward/backward parity remain unverified. Running the
in-tree adapter test entry points with the isolated Torch shim currently reaches
core-runtime construction and produced no result within the short validation
window; this is recorded as an environment/build prerequisite, not as a pass.

PyPI metadata exposes a `vllm==0.11.0` cp38-abi3 wheel, but it requires
`torch==2.8.0`, matching torchaudio 2.8 and torchvision 0.23. The available
oracle is `torch==2.12.1+cu126`, so installing vLLM into it would replace the
reference runtime and invalidate the comparison; it was deliberately not
installed. PyPI has no matching `trellis` distribution. TRELLIS therefore needs
its upstream source installation on a target machine.

An isolated wheel metadata probe found `vllm==0.11.0` (CPython 3.8 ABI,
manylinux x86_64), requiring Python 3.9--3.13 and exactly
`torch==2.8.0`/`torchvision==0.23.0`/`torchaudio==2.8.0`. The available Torch
2.12.1+cu126 environment therefore was deliberately left unchanged. PyPI has
no `trellis` distribution; TRELLIS requires its upstream source installation.

No performance claim is made. A dedicated Jittor-vs-PyTorch UNet parity run
still requires running the same compact model through the Torch shim.

The compact `diffusers.UNet2DModel` probe was started with the isolated shim and
existing CPU cache. Runtime initialization reached the optional
`flash_attn` fallback, but forward/backward produced neither completion nor a
Python exception within 90 seconds and was stopped. This is an inconclusive
build/runtime prerequisite result, not a parity pass.

## Reproducible target-machine preflight

Record the target matrix before installing optional ecosystems:

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvidia-smi
python -c 'import importlib.util; print({n: bool(importlib.util.find_spec(n)) for n in ("vllm", "trellis")})'
```

Use an isolated environment for vLLM. The probed wheel requires Python 3.9--3.13
with `torch==2.8.0`, `torchvision==0.23.0`, and `torchaudio==2.8.0`; install
those matching CUDA wheels first, then `python -m pip install vllm==0.11.0`.
Do not replace the current PyTorch 2.12.1 oracle environment. TRELLIS has no
PyPI package; use the upstream repository's pinned requirements in its own
environment. Verify imports before enabling the adapter:

```bash
python -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'
python -c 'import trellis; print(trellis.__file__)'
python -m pip install ./adapters
```

The in-tree entry point is `jittor_vllm = jittor_adapters.vllm:register` and
importing the adapter package alone is inert.

## Tiny UNet shim probe

A reproducible compact `UNet2DModel` probe (sample size 4, one channel and one
block) exposed two configuration-boundary incompatibilities before a complete
forward/backward run. With diffusers defaults, `attention_head_dim=8` and four
channels produce zero attention input features and a `ZeroDivisionError` in
Jittor's `invariant_uniform` initializer. Setting `attention_head_dim=4` then
reaches default `norm_num_groups=32` with four channels and raises the native
GroupNorm divisibility assertion. With both values valid
(`attention_head_dim=4`, `norm_num_groups=1`), runtime initialization produced
no result within the 30-second bounded probe. These are compatibility/runtime
blockers; no runtime code was changed.

The model-independent adapter contract check also ran successfully:
`python adapters/tests/test_adapters.py -q` executed 4 tests in 0.004 seconds.
It covers version guards, import protection, and the NPU probe without importing
the Jittor core or requiring optional model packages.

With the optional package absent, the adapter lifecycle remains testable:
`JITTOR_VLLM_HOST_ONLY=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=adapters python -m
pytest -q adapters/tests/vllm/test_plugin_lifecycle.py` passed 5 tests in 0.10
seconds. This checks entry-point arming, no-import behavior, ownership, and
rollback only; it is not vLLM numerical or CUDA-serving evidence.
