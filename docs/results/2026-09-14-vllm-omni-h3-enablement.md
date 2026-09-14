# MiniMax-H3 through jittor + torch-compat + vLLM-Omni: the shim gaps

- Status: In progress; five shim/backend defects fixed and verified individually,
  engine construction still walking forward
- Date: 2026-09-14
- Owner: Jittor compatibility maintainers
- Review when: the offload path, `torch.device` placement, or the conv/pow
  op-type tables change

## Question

Can `jittor` + `JITTOR_TORCH_SHIM=1` + `vllm-omni` run MiniMax-H3 inference
without patching vLLM-Omni? This records the jittor-side defects that blocked
it, so the next person does not re-derive each one from a stack trace.

The reference run is a single H20 (96 GiB) with the layerwise-offload profile
from `recipes/MiniMaxAI/MiniMax-H3.md` (the DiT and Qwen3-VL encoder cannot be
resident together: ~62 GiB + ~63 GiB of BF16 weights).

## 1. `TCPStore` could not rendezvous on a multi-address `localhost`

`jittor/distributed/store.py` built its listener with
`socket(AF_INET).bind((host, port))` but the client dialled with
`socket.create_connection((host, port))`. Those two resolve a hostname
differently: `bind` takes the first `AF_INET` answer, `create_connection`
walks an `AF_UNSPEC` list. On this host `/etc/hosts` maps `localhost` to `::1`
and to the machine's own IPv4 address and to **no** loopback IPv4, so the
server bound `127.0.0.1` while every client dialled `::1` first. Nothing was
ever reachable; the failure surfaced 300 s later as
`TimeoutError: timed out connecting to TCPStore ... [Errno 115] Operation now
in progress`, naming neither address.

Fix: bind the first address the client will actually dial
(`_client_dial_addresses`), skipping families whose `bind` fails.

## 2. `with torch.device("cpu"):` did not move the allocation default

vLLM-Omni builds the diffusion pipeline inside `with target_device:` and
depends on it to keep the checkpoint off the GPU. The shim's
`torch.device.__enter__` handled only `meta` and indexed CUDA devices, so a
`cpu` block was a no-op: `torch.ones(...)` still went to `cuda:0` and
construction died at `mem_info.cc:275 ... Used: 94.06GB` on a 96 GiB device.

Fix: keep a thread-local stack of the active device context (torch uses a
thread-local `DeviceContext` too) and have the factory frontend's
`_placement_request` consult it when the caller passed no device and there is
no `_like` input. Outside any block the historical jittor default is untouched;
`torch.empty_like(x)` still follows `x`, not the ambient context.

## 3. A CPU translation unit could receive a CUDA-only `pow` template

With `use_cuda=1`, `CommonOpType::expand_op` picks the CUDA op-type table from
the **runtime** flag, not from the unit's backend (`#define JIT_cpu`). The CUDA
table spells `pow` as `jittor::_signed_pow`, and `type/pow_compute.h` defined
that symbol only under `#ifdef JIT_cuda`. The include that `post_pass` adds was
therefore visible but empty, and any host-side `pow` in a CUDA-enabled process
failed to compile: `'_signed_pow' is not a member of 'jittor'`. Host-side
schedules are exactly what the CPU-offloaded pipeline builds.

Fix: give `_signed_pow` a CPU spelling (`::pow`, which already signs an
integral exponent correctly) so the symbol exists in every unit that can
receive the template. The selection-by-runtime-flag remains the underlying
design smell: any other op whose CPU and CUDA table entries diverge is still
selected by the wrong key, and `fp16_op_type.cc` has the same shape.

## 4. `torch.nn.Conv3d` rejected `padding_mode`

MiniMax-H3's video VAE subclasses `nn.Conv3d` and forwards the torch default
`padding_mode="zeros"` (its decoder uses `"reflect"`). The native signature has
no such parameter, so construction raised
`Conv3d.__init__() got an unexpected keyword argument 'padding_mode'`.

Fix: the shim's layer initializer now captures torch-only `padding_mode` and
installs an `execute` that emulates torch's semantics -- pad the input with the
requested mode using `jt.nn.pad`, then convolve with no padding. `"zeros"`
keeps the existing native path, unknown modes are refused. Verified against a
numpy reflect-pad + conv reference (max error 2e-6).

## 5. `torch.get_device_module` was missing

The same VAE stores `torch.get_device_module()` and drives its `device()` scope
and `manual_seed` through it. The name did not exist in the shim namespace, so
construction raised `AttributeError: get_device_module`.

Fix: implemented in `installers/core.py`; no argument means the current
accelerator (`torch.cuda` when `use_cuda` is on), matching torch.

## Verification

- 1: `tests/distributed/test_process_store.py::TestHostnameRendezvous` -- fails
  pre-fix with the connect timeout, passes after. Reproduced standalone before
  the test existed.
- 2: `compat/tests/torch/test_torch_compat_unimplemented.py` device-context
  tests, plus a direct probe: `torch.ones(2).device` is `cuda:0` outside a
  block, `cpu` inside `with torch.device("cpu")`, and restored after.
- 3: `compat/tests/torch/test_multi_device.py` host-resident-pow test (CUDA
  round). Standalone repro failed pre-fix with the exact compiler error and
  passes after. Note the JIT cache can mask this: the `post_pass`-added include
  is not part of the op key, so a warm cache from the fixed build will pass
  even with the header reverted.
- 4: `compat/tests/torch/test_torch_compat_conv_pool.py` `padding_mode` test.
- 5: probe only; the engine run is the integration check.

Each fix was synced into the lab venv at
`$JITTOR_LAB_ROOT/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/`,
which is a **copy** of `python/jittor` plus the installed `jittor-torch`
compat package; the adapter checkout is editable.
