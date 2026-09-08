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
