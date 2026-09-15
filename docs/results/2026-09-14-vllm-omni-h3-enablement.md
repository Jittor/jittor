# MiniMax-H3 through jittor + torch-compat + vLLM-Omni: the shim gaps

- Status: Engine construction completes (`ENGINE-CONSTRUCTED`, 10.0 GiB peak
  during load); generation reaches the model forward; validation of speed /
  results / memory in progress
- Date: 2026-09-14
- Owner: Jittor compatibility maintainers
- Review when: the offload path, `torch.device` placement, the conv/pow op-type
  tables, `.data`/view handling, or `torch.as_strided`/`empty_strided` change

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

## 3. Op-type tables were chosen by the runtime flag, not the compile target

`CommonOpType::expand_op` and `FP16OpType::expand_op` picked their CUDA table
from `runtime_flag_use_cuda()` -- the *process-wide* flag -- rather than from
the translation unit's own backend. With `use_cuda=1` a host unit
(`#define JIT_cpu`, which a CUDA-enabled process compiles for every CPU-resident
Var) therefore received CUDA-only entries. Three of them do not work off
device:

- `pow`: `jittor::_signed_pow`, defined in `type/pow_compute.h` only under
  `#ifdef JIT_cuda` -> `'_signed_pow' is not a member of 'jittor'`;
- half `abs`: `::__habs`, a CUDA intrinsic -> `'::__habs' has not been
  declared`;
- half comparisons: the CUDA table's `(($2)>($4))` is a mixed comparison once
  the operands differ (e.g. `bfloat16` vs `int32`), and jittor's host half
  types convert both ways, so it is rejected as an ambiguous `operator>`.

Host-side schedules are exactly what the CPU-offloaded pipeline builds, so all
three blocked engine construction.

Fix: thread the unit's backend through `OpByType::expand_op(args, is_cuda)`.
`OpCompiler` derives `is_cuda` from the `JIT_cuda`/`JIT_cpu` define already in
`defs` at the `@expand_op(...)` expansion, falling back to the runtime flag only
for a caller of `precompile()` that declares no backend (none currently expand
`@expand_op`). The fp16 comparisons now convert through `float`, matching what
that table's `equal` entry already did. `pow_compute.h` stays CUDA-only: a CPU
unit gets the CPU table's `std::pow`, and giving the header a CPU spelling would
make the regression test below pass even if this choice broke again.

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

## 6. `tensor.data = other` copied elements instead of replacing storage

The shim's `data` setter ran `self.assign(src)`. That is the in-place
primitive behind `x.foo_()`: it writes x's values into y's storage and only
then aliases the two, so it requires equal element counts. Torch's
`x.data = y` *replaces* x's data, shape and dtype. vLLM-Omni's layerwise
offload swaps every block parameter for a zero-element placeholder and
restores it later, so the copy path failed with
`reshape shape is invalid for input of size [x_items(0) == y_items(1152)]`.

Fix: rebind with `_update` (the pure "adopt this Var" primitive), keeping the
existing `requires_grad` restoration. Equal-shape `x.data = y` now aliases
`y` rather than copying into it, which is what torch does and why torch
documents `.data` as unsafe.

## 7. A size-changing rebind aborted on lazily-recorded views

`view(-1)` (which the shim's `flatten()` reaches) records a *storage view* on
its base: a lazy reshape expression that `refresh_transpose_views` re-derives
whenever the base's data changes. vLLM-Omni's offload flattens each parameter
and then swaps it for a zero-element placeholder, so the refresh re-applied a
reshape of 1152 elements to an empty holder and aborted with
`reshape shape is invalid for input of size [x_items(0) == y_items(1152)]`.
Both `assign` and `_update` call the refresh, so neither data path could work.

Fix: a recorded step is only re-derived when it still fits the new value.
Reshape/Expand record the target shape, so their product must match the new
element count (Expand additionally needs each source axis to be that size or
1); Transpose records a permutation, so it needs a matching rank; Slices are
left alone. A view that no longer fits is dropped and keeps the data it was
taken from -- the same outcome torch's views have after `x.data = y` replaces
the storage under them.

## 8. `torch.as_strided` and `torch.empty_strided` were missing

`prefetch_layer`/`restore_next_block` rebuild each parameter from the packed
host buffer with `torch.as_strided(flat[offset:offset+numel], size=..., stride=...)`,
and `restore_tensor_storage` allocates an independent buffer with
`torch.empty_strided(value.shape, value.stride(), ...)` before `copy_`-ing into
it and installing it as the parameter's data. `Tensor.as_strided` exists in the
shim but neither module-level entry point did, so the offload died with
`AttributeError: as_strided` and then `AttributeError: empty_strided`.

Fix: add both in `installers/core.py`. `as_strided` delegates to the existing
method, which materializes the window with a gather: reads are exact, and the
result does not alias `input`, which matters only for the `copy_`-into-the-view
direction `flatten_physical_storage` uses for a *non-contiguous* tensor (the
shim reports its parameters contiguous, so that branch is not taken). Its
default offset is 0 rather than torch's `input.storage_offset()`: jittor
materializes slices, so `input`'s own data already starts at its first element
and torch's default applied the parent-relative offset twice ("index 10751 is
out of bounds for dimension 0 with size 5376" on
`gpu_weight[offset:offset+numel]`). `empty_strided` allocates `size`
contiguously -- jittor cannot honor `stride`, and a non-strided `layout` is
refused -- which is all the caller needs before its element-wise `copy_`.

## 9. Cross-device `copy_` moved the destination to the source's device

`_ip` (behind every `x.foo_()`) uses jittor's `assign`, which writes x's values
into *y's* storage and then aliases the two. For `copy_` that is the wrong
shape: torch's `copy_` is an in-place write, so the destination keeps its own
device and a cross-device copy is a transfer. A cuda tensor filled from a CPU
one reported `cpu` afterwards.

That was the encoder's `load_to_device()` guard:
`PinnedModuleStager._load_once` builds `device_storages` with
`torch.empty_like(group.master, device=cuda)` (correct -- measured `cuda:0`),
fills them with `device_storage.copy_(group.master)` (CPU master), and every
storage silently became a host tensor again. The stager then bound all
non-block parameters to host views, and `encode_ids` raised
`call load_to_device() before encode_ids()` because
`next(self.parameters()).device.type != device_target.type`.

Fix: in `_copy_`, materialize the source on the destination's device first when
they differ. Values are unchanged; same-device copies keep the existing path.

Reproduced and verified with a two-minute standalone harness (a two-layer
`Sequential` plus the real `PinnedModuleStager`, no engine): parameters were
`cpu` after `load()` before the fix and `cuda:0` after.

## 10. Core helpers allocated on the ambient placement, not the inputs'

MiniMax-H3 builds its sigma schedule with `device="cpu"` while the process is on
CUDA, and then calls `unique_consecutive` on it. `jt.ones`/`jt.zeros`/`jt.empty`
follow the *ambient* placement, so the helper tensors landed on the device:
`dispatch_context` then rejected the op with "Expected all tensor inputs on the
same backend and device" (the message now names both placements -- that is how
this was found).

Fix: a small `jittor._core.var.placement_scope_like(x)` context manager, used by
`concatenation._concat_direct` for its output and by `unique_consecutive` for its
leading `True`, its `counts` and its scatter `ones`. (In `_core/var.py` the name
`int` is shadowed by jittor's integer dtype constructor, so the helper uses
`ori_int`.)

Two more shim/core mismatches on the same path:

- `torch.backends.cuda` had the four `enable_*_sdp` setters but none of the
  matching `*_sdp_enabled` getters; the encoder saves and restores
  `cudnn_sdp_enabled()`. The four getters now report the recorded state
  (`cudnn` defaults to False -- there is no cuDNN fused SDPA here).
- `dtype.is_complex` / `is_floating_point` are torch-style *attributes* but
  jittor core calls them as *methods* (`advanced_indexing._indexing_index`).
  They now return a `_CallableBool`, so both readings work.

## 11. A mixed fp32/fp16 product fell back to the outer product

This is the one that made the request slow, and the last one needed for it to be
*normal* rather than merely correct.

The H3 video VAE's decoder blocks are explicitly an autocast workload.
`install_h3_vae_optimizations` casts the decoder `nn.Linear` weights to float16
so it does not rebuild the same cast on every tile, its comment says so
("The H3 decode path always uses FP16 CUDA autocast"), and
`pipeline_minimax_h3.decode` wraps the call in
`create_autocast_context(dtype=torch.float16)`. The activations stay float32 --
`_optimized_transformer_block` only takes its fast path when they are, and under
real torch they are float32 *as tensors* too; casting them is autocast's job.

Under real torch that is a float16 product. Under the shim, autocast sets
`jt.flags.amp_reg` and the cast never happened, so every decoder `Linear`
reached the cuBLAS row as a **mixed** pair. That row is selected on the
operands' own dtypes (`_same_floating_dtype` requires `a.dtype == b.dtype`), so
a mixed pair matched no kernel, and `matmul`/`matmul_transpose` fell through to
their generic form:

    shape = list(a.shape)[:-1] + list(b.shape)
    return (a.broadcast(shape) * b.broadcast(shape)).sum(-1)

which *materializes* `[B, out, in]`. At the decoder's real shape that is
`1797 * 6144 * 2048 = 22.6e9` elements -- 45 GiB in float16, 86 GiB the moment
anything promotes it to float32, which is how the standalone decode died
(`could not allocate 86256 MiB`, `op: fused_op(unary.cast)
in: float16[1797,6144,2048,] out: float32[1797,6144,2048,]`). In the run that
did fit, this was the whole `sync` column: the Triton bridge's per-launch
`jt.sync_all(True)` was waiting for ~1.3 s of outer product per decoder block,
and `py-spy` put the wait there because that is where the pump was.

Measured at `512x2048x6144`, on the same device:

| `linear(x, w)` | kernel chosen | time |
| --- | --- | --- |
| float32 x float32 | `cublas_matmul` | 0.57 ms |
| float16 x float16 | `cublas_matmul` | 0.49 ms |
| float32 x float16 | *none* -> outer product | **53.6 ms** |
| bfloat16 x float16 | *none* -> outer product | **51.7 ms** |

Fix: `_mixed_float_compute_dtype` resolves a mixed floating pair to a single
dtype and `_matmul_kernel_dispatch` retries the relay with both operands cast.
The resolved dtype is the one the active autocast/AMP region asks for (float16,
or bfloat16 when an operand already is -- what `amp_prefer16` means in
`src/type/nano_string.h`), so the product comes back float16 exactly as it does
under real torch, which is also what the VAE's bit-exact residual kernel
(`try_scaled_residual_exact`) requires of its `branch` operand. Outside such a
region the pair is promoted, widest first. A pair that already shares a dtype,
or that involves a non-float operand, is left on its previous route.

## Result: a full request now completes, at normal speed

With every fix above in the working tree, one `fl2va` request runs end to end
(`vllmomni-stats4.log`, single H20, `diffusion_offload_config` = layer mode over
`dit`+`text_encoder`):

    Model loading took 10.0312 GiB and 280.8 seconds
    [gen] ENGINE-CONSTRUCTED in 304.9s
    MiniMax H3 t2va Qwen presentation: 13 tokens
    100%|##########| 1/1 [00:13<00:00, 13.00s/it]        <- denoise, warm JIT cache
    [gen] frames: (124, 256, 256, 3) dtype: uint8
    [gen] audio: (1, 2, 165600)
    [gen] peak_memory_mb: 24302.0
    [gen] GENERATE-OK

124 frames at 24 FPS is the requested 5 s, and 165600 samples per channel is
5.175 s at 32 kHz -- both shapes match the request. Peak device memory 24.3 GiB
on a 96 GiB card.

**Speed is not normal yet.** The same run reports `GENERATE took 720.5s` with
`stage_0_gen_ms=720503`: the denoise is 13 s (warm) while the video VAE decode
is ~700 s. `py-spy` puts the decode in vLLM-Omni's `try_scaled_residual_exact`
Triton kernel, i.e. inside jittor's Triton bridge, where every launch calls
`device_raw_ptr` (which does `sync(true, false)`) once per pointer argument and
bounces every operand through a guarded buffer (`GUARD_ENABLE = True` by
default). That is the next thing to fix.

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
- 6: `compat/tests/torch/_torch_compat_checks.py` (`x.data =` checks) and a
  standalone probe: empty placeholder, shape restore, dtype change,
  `requires_grad` preserved.
- 7: `tests/core/test_transpose_view_staleness.py` (13 existing cases plus the
  new size-changing-rebind case, all 14 passing on CPU and CUDA). The
  same-shape refresh cases in that file are what caught an early version that
  also dropped Transpose views.
- 8: standalone probes: `as_strided` over a plain and over a slice of a larger
  flat buffer, and `empty_strided` + `copy_` matching the source.
- Integration: with every fix above in the working tree, engine construction
  reached `[try] ENGINE-CONSTRUCTED` at 10.0 GiB peak during load and shut
  down cleanly (`vllmomni-try18.log`).
- 6: `compat/tests/torch/_torch_compat_checks.py` (`x.data =` checks) and a
  standalone probe: empty placeholder, shape restore, dtype change,
  `requires_grad` preserved.

Each fix was synced into the lab venv at
`$JITTOR_LAB_ROOT/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/`,
which is a **copy** of `python/jittor` plus the installed `jittor-torch`
compat package; the adapter checkout is editable.
