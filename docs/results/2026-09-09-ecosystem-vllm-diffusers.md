# 2026-09-09 ecosystem smoke results

This note records the available host evidence for the Torch-compatible ecosystem
request. It does not claim vLLM or TRELLIS support without those packages.

## Host evidence

The current Python environment imports `torch`, `diffusers`, `transformers`, and
`mmcv`. A real PyTorch `mmcv.cnn.ConvModule(3, 4, 3, padding=1)` forward and
backward completed with output shape `(2, 4, 8, 8)` and a finite input gradient
norm. A real PyTorch `diffusers.DDPMScheduler.add_noise` smoke also completed
for a `(2, 4)` tensor.

The dedicated PyTorch 2.12 environment does not contain `diffusers`; the
installed package is only visible from the environment whose `python` entry
point activates the Jittor shim. Consequently an actual PyTorch UNet
forward/backward parity run was not available on this host.

## Compatibility status

The current environment has no importable `vllm` or TRELLIS package, so their
compatibility and PyTorch forward/backward parity remain unverified. Running the
in-tree adapter test entry points with the isolated Torch shim currently reaches
core-runtime construction and produced no result within the short validation
window; this is recorded as an environment/build prerequisite, not as a pass.

No performance claim is made. No code or package was installed for this check.
