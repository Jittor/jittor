# MiniMax-H3 through jittor + torch-compat + vLLM-Omni: the shim gaps

- Status: A full `fl2va` request runs end to end at normal speed
  (`GENERATE` 39.6 s, down from 710.7 s); frames, audio and peak memory match the
  earlier correct-but-slow run
- Date: 2026-09-14, speed resolved 2026-09-15
- Owner: Jittor compatibility maintainers
- Review when: the offload path, `torch.device` placement, the conv/pow op-type
  tables, `.data`/view handling, `torch.as_strided`/`empty_strided`, or the
  matmul dtype routing changes

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

## 12. Serving it: the OpenAI server needed one more stub fix

The offline `Omni(...)` path and the `vllm-omni serve ... --omni` path do not
import the same code. The server pulls in `vllm.entrypoints.serve` and
vLLM-Omni's `entrypoints/openai/`, which the offline path never touches, so the
server needs (a) the deps only the serving stack declares and (b) one more shim
stub fix.

The stub fix: `vllm_omni/utils/audio.py` does

    from torchaudio.functional import melscale_fbanks

at module level, and `serving_chat` imports that module, so the whole server
import chain died on it. The packaged `torchaudio` stub
(`compat/shim/resources/stubs/torchaudio/__init__.py`) answered
`from torchaudio import functional` -- its `__getattr__` fabricates a module for
any name -- but not the submodule spelling: a module with no `__path__` offers
the import system no submodule to find, so the import failed before it reached
the name lookup `__getattr__` would have answered. The stub now registers a
`_AnyFinder` for anything under `torchaudio.`, and the fabricated modules carry
`__path__` so a deeper name resolves too. Attributes are still the same
loudly-failing classes -- the shim ships no audio DSP, and `melscale_fbanks(...)`
raises rather than returning a plausible number. The H3 video path never calls
it; the import only has to succeed.

One trap worth knowing about the deployment: the shim materialises its runtime
site-packages under `$XDG_CACHE_HOME/jittor/torch-shim/<project>-<hash>/`, keyed
by the *project path*, and re-deploys the stubs into it on every activation from
the **installed** `jittor/compat/shim/resources/stubs/`. Editing only
`site-packages/torchaudio/__init__.py` therefore gets overwritten on the next
run; the resource copy has to be updated too.

Verified with `vllm-omni serve ... --omni --num-gpus 1` on one H20
(`serve1.log`): `GET /v1/models` 200, and `POST /v1/videos/sync` returned a
256x256 `video/mp4` with both a video and an audio stream, **39.5 s** against
39.6 s for the same request offline. Serving is at parity with the offline path.

## 13. Multi-GPU: three jittor defects, then a TP2 server that runs

`--tensor-parallel-size` / `--text-encoder-tp-size` / `--vae-patch-parallel-size`
above 1 all need multi-rank collectives, and that path had never been run here.
Four things stood between the recipe and a TP2 server, in the order they surface.
The first three are fixed; the fourth is new work, described at the end:

1. **The launcher does not enable the shim's multi-rank bootstrap.** The shim
   implements a dynamic NCCL bootstrap for `init_process_group(world_size>1)`,
   but it is gated behind `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT`, which nothing
   sets. Unset, every diffusion worker dies with "multi-rank torch.distributed
   requires launching Jittor with jittor.distributed.launch or explicit dynamic
   bootstrap". Fixed by exporting it -- an undocumented prerequisite.

2. **`make_cache_dir` raced, and lost.** With the bootstrap on, both ranks call
   `setup_nccl`, both create `.cache/jittor/nccl` through

       if not os.path.isdir(cache_path): os.mkdir(cache_path)

   and the loser died with `FileExistsError: .../.cache/jittor/nccl`. Sharing a
   JITTOR_HOME across ranks is *the* case where two processes create this
   directory at once, so the race was the ordinary path. Now
   `os.makedirs(..., exist_ok=True)` (which also creates a missing parent, as
   the bare `mkdir` did not).

3. **NCCL is not found without mpirun.** `setup_nccl` will use a system NCCL if
   given `JT_BUILD_NCCL_INCLUDE_PATH` / `JT_BUILD_NCCL_LIB_PATH`, otherwise it
   goes to `install_nccl`, whose `if not inside_mpi(): return` still sits under
   the comment saying the mpirun-free path is meant to build too -- so on this
   box (system NCCL 2.27.3 in `/lib64`, `nccl.h` in `/usr/include`) the
   no-mpirun path can never find it. With the two variables set, the NCCL ops
   compile and publish (the earlier "did not publish collective ops" goes away).

4. **The shim does not derive `JT_NCCL_ROOTINFO_FILE` when a store is in hand,
   so `new_group` cannot work at all.** This was the real blocker, and it is the
   one that looked like a rendezvous problem:

       RuntimeError: NCCL store rendezvous timeout: rank 1 waited 120 s and
       timed out using the provided Store

   The store path itself is fine. `nccl_create_process_group` exchanges every
   *sub-group's* unique id through a file named after `JT_NCCL_ROOTINFO_FILE`
   (`<rootinfo>.pg<group_id>`, `backends/comm/nccl/src/nccl_wrapper.cc:531`),
   and the shim only derived that path in its store-less branch
   (`compat/torch/installers/distributed.py:97`, `if not rootinfo and store is
   None:`). A store carries the *world* communicator's id only, so with a store
   present the file path stayed unset and the next `new_group` died with

       RuntimeError: nccl_wrapper.cc:531: NCCL process groups require
       JT_NCCL_ROOTINFO_FILE in MPI-free mode

   vLLM-Omni's `GroupCoordinator` calls `new_group` to build the world group
   (`group_coordinator.py:110`), so this fires immediately on every multi-rank
   run. Deriving the path is independent of whether a store arrived; dropping
   `and store is None` fixes it. Single-GPU never enters this function.

   Two operational traps, both of which look like code defects and are not.

   The first TP2 attempt died even earlier, on `Orchestrator initialization
   failed: ipc path "..." is longer than 107 characters` -- the shim sets
   `TMPDIR` to `<runtime>/tmp`, which is 90 characters under this box's
   `XDG_CACHE_HOME`, and vLLM-Omni appends an `ipc://` socket name.
   `JITTOR_TORCH_KEEP_TMPDIR=1` (the shim's own escape hatch, otherwise
   undocumented here) is the fix.

   The rendezvous timeout above is **orphaned workers, not the store** -- at
   least for the first occurrences. Killing a server by matching the port string
   reaches only the parent -- the `DiffusionWorker` children do not carry
   `--port` in their argv -- so they survive holding `MASTER_PORT`, and the next
   run's rank 1 rendezvouses with the orphan's store and waits for a `set` that
   will never come. `stop-vllmomni.sh` (walk the process tree from the parent,
   then clear `/tmp/jittor-nccl-*`) is the fix; the per-group files are named
   after `MASTER_ADDR`-`MASTER_PORT` alone, so a rerun on the same port must not
   inherit them either. It kept happening after that fix, though, and the
   thread-level evidence below is what the *remaining* cases look like -- a
   different mechanism with the same message, so do not stop at this paragraph.

**Where TP2 stands.** With all four addressed, a `--num-gpus 2 --tensor-parallel-size 2`
server **starts and serves**: both ranks create their groups with no NCCL error,
both load the model with layer-wise offload (`DiffusionWorker_TP0` / `TP1`) at
**10.07 GiB per rank** (304.6 s to load, against 62 GB for the whole DiT
single-GPU), and a request gets as far as the denoise loop. It then dies on the
**first** denoise step, on rank 1 (`device=1`):

    cudaErrorIllegalAddress (sticky: it surfaces at the next cudaMemGetInfo)

with this stack, which is about as precise as the error gets:

    minimax_h3_denoise_loop            denoise_loop.py:335
    -> _forward_varlen_packed          minimax_h3/flash_attn.py:233
    -> flash_attn_varlen_func          flash_attn/__init__.py:199
    -> packed_low_level.varlen_fwd     adapter.py:370

The trace also rules out the obvious suspects. At the failing call, on both
ranks, with `H3_FA_SHAPE_TRACE=1`:

    TP0: q.dev=0 cu.dev=0 jt_dev=0 torch_cur=0 q_contig=True q_stride=[3584, 128, 1]
         q=[289, 28, 128] k=[289, 28, 128] v=[289, 28, 128]
         cu_q=[0, 289, 289] cu_k=[0, 289, 289] max_q=289 max_k=289
    TP1: q.dev=1 cu.dev=1 jt_dev=1 torch_cur=1 q_contig=True q_stride=[3584, 128, 1]
         (identical shapes and cu_seqlens)

So the packed plan matches the tensor (`cu_q[-1] == q.shape[0] == 289`), the
devices agree on each rank, and the tensors are contiguous with the expected
stride. Ruled out, in order of how much they were suspected:

- a sharded-token / `cu_seqlens` mismatch;
- a device-selection mistake (each rank's current device equals its tensors');
- a non-contiguous view reaching the kernel.

**A host-resident argument at that call is ruled out too, and by experiment
rather than by argument.** A helper that moves any `device_id >= 0` Var whose
`location()` is `cpu` back to its device, installed immediately before
`packed_low_level.varlen_fwd` and enabled for the run, **never fired** -- and the
run still died with `cudaErrorIllegalAddress`. That also disposes of the
"one root cause, two symptoms" reading this section previously carried: the
`is_cuda` symptom appeared only in the run whose trace called
`cu_q.data.tolist()`, which is itself the host-read path that parks a Var. The
instrument changed the system it was measuring. The correct statement is that
`cu_seqlens_q must be on CUDA` is what section 16 fixed, and the TP2 illegal
address is a *different* fault with well-formed inputs.

What is left is something below the arguments: the illegal access is raised on
rank 1 with `device=1` and surfaces device-wide (sticky, at the next
`cudaMemGetInfo`), so the recorded stack is a candidate list, not proof. The
next tool for it is `compute-sanitizer` on a TP2 run, which is decisive about
*which* launch faults.

**The illegal address is not the attention bridge.** `--tensor-parallel-size 2`
with `TORCH_SDPA` -- no flash-attn involved at all -- fails the same way
(`cudaErrorIllegalAddress`, rank 1, first denoise step). The stack moves with the
backend, so both stacks are downstream of something shared; with SDPA it reads

    denoise_loop.py:335 -> minimax_h3_transformer.py:1542 -> :899
    -> attention/ops/minimax_h3_modulation.py:258 rms_norm_indexed_scale_shift
    -> :21 _launch_row_chunks -> triton -> compat/triton/backend.py:1102

and `backend.py:1102` is `jt.sync_all(True)`, i.e. where jittor *notices* the
fault, not where it happened. Worth noting for whoever picks this up: the kernel
that line launches, `_indexed_scale_shift_kernel`, masks columns but **not
rows** (`row = tl.program_id(0)`, unmasked `indices_ptr + row * stride`), so any
disagreement between the grid's `rows` (taken from `x.shape`) and the row count
of `indices`/`shift`/`scale` is an unmasked out-of-bounds access. That is a
hypothesis to test, not a finding -- the same caveat as above applies.

**The rendezvous flake, from the threads rather than from the message.** A hung
TP2 start (`py-spy dump`, both ranks) shows it is *not* an orphaned store and
*not* a slow rank:

    rank 0 (pid 1145000) MainThread:  _init_nccl_from_store (compile_extern.py:684)
                                      -> nccl_init_with_unique_id(...)
                        Thread-1: _accept_loop (accept)
                        Thread-2: _serve_connection (socket write)
                        Thread-3: _serve_connection (readline, idle)
    rank 1 (pid 1145002) MainThread:  get (store.py:342) -> request (store.py:271)
                                      -> readinto (socket.py:720)

Rank 0 is already **past** its `store.get` and building the communicator, so the
key *is* in the server's store; rank 1 has only a client thread and is waiting
for the response. `ss` shows both connections ESTABLISHED with Send-Q and
Recv-Q zero on both sides, and no server thread sits in `_dispatch`. So the
bytes for that `get` are not in flight and no handler is running, which rules
out "rank 0 never set the key" -- the earlier orphan-port story explained the
*first* occurrences and does not explain this one. The remaining suspect is the
request/response framing between `_TCPStoreClient.request` and
`_serve_connection` (statically each pair looked balanced), and the way to settle
it is a per-request trace in those two functions on a hung run. The lab has that
ready: `instrument_store_and_triton.py` applies both traces to the deployed shim
(store request/response logging, and a triton launch trace printing each kernel's
grid next to its operands' shapes, which is the one-line check for the unmasked
row walk above). Relaunch with `H3_STORE_TRACE=1 H3_TRITON_SHAPES=1`; both are
no-ops without their env var, and the repo copies are the clean ones to restore
from afterwards.

**The cause is jittor's own allocator, and it is reached through the offload's
storage swap.** With `compute-sanitizer` on a TP2 run the failure is not a bad
kernel launch at all -- the sanitizer reports no invalid memory access -- but
this, raised out of the request:

    sfrl_allocator.cc:305: mem_ptr does not belong to allocation: 2167
    [check failed: (char*)mem_ptr >= (char*)block->memory_ptr
                   && (char*)mem_ptr <= (char*)block->memory_ptr + block->size]

    method_api.py:506 in _data_set
        self._update(src)
    "... This is an internal Jittor invariant, not an error in your program."

`_data_set` is the shim's `tensor.data = value`, and the caller is vLLM-Omni's
layer offload: `tensor_utils.py:111 set_tensor_storage` does `target.data = value`
to swap a parameter for a zero-element placeholder
(`clear_tensor_storage`), and later installs a fresh weight buffer with
`torch.as_strided` (`layerwise_backend.py:196`). So TP2 dies inside a jittor
memory-management invariant, which is also why the reported location kept moving
with the instrumentation: every candidate stack was just the first CUDA call
after the corruption.

**Reduced to three lines** (`probe_split_alias.py`, lab):

    param = torch.zeros(4096, 512, dtype=torch.bfloat16, device="cuda")
    alias = param[3072:]                                    # non-zero offset
    param.data = torch.empty((0,), dtype=..., device="cuda")  # offload's swap
    small = torch.ones(8, 512, ...) + 1.0                   # reuse -> split
    del alias    # -> sfrl_allocator.cc:305, same assert

Bisected (`probe_split_alias_cases.py`): a **tail slice with a non-zero offset** is
the trigger; `.data`, `.detach()`, a head slice and a zero-offset `as_strided` are
all fine, and so is the swap with no reuse.

The two code facts that meet:

- `Var::share_with(x, offset)` only *records* the request
  (`var.h:113`: `share_src = x; share_offset = offset;`). The alias is registered
  with the allocator later, in the child's own `Var::alloc`
  (`var.cc:242-265`), which does
  `x->allocator->share_with(storage_span_bytes(), x->allocation)` and then
  `mem_ptr = (char*)x->mem_ptr + share_offset`. If the source's storage was
  freed in between, that call registers against *whatever now owns that
  allocation id* and the child points into reused memory.
- `should_split` (`sfrl_allocator.cc:172`) is `block->size - size >= ALIGN_SIZE`
  -- it does not consider shares -- and on a split the tail keeps the *same*
  allocation id (`:278 rest->allocation = block->allocation`). So one allocation
  spans two adjacent blocks (the code leans on that: `try_merge_two_blocks`
  asserts neighbours share an allocation), while `free` validates the pointer
  against only the single block `get_occupied(allocation)` returns. If the
  split-off head is what the id maps to, a tail pointer is "outside" it.

An eager `jt.sync_all(True)` after the slice (to register the share before the
swap) does **not** help, so `share_times` is not protecting the block in this
ordering.

What the alias then *reads* is worth recording, because it bounds the claim:
after the swap the alias reports shape `(0,)`, i.e. it re-evaluated through the
rebound variable rather than reading the small tensor's values
(`probe_alias_uaf.py`). So this repro shows a **stale deferred share being freed**
-- a bookkeeping defect -- and the downstream that turns it into the observed
illegal address (a freed block handed to a live tensor) is the next thing to pin,
not something this repro demonstrates directly.

Two candidate fixes, both in jittor, neither applied yet:

1. **Make the share eager and pinned.** Register the share at request time when
   the source is already allocated, and hold a liveness reference to the source
   (`share_src` is a raw `Var*` today, so the request can outlive the variable it
   names). Then `share_times` keeps the block from being freed or split for as
   long as the alias exists, which is the property the allocator's own comment
   assumes.
2. **Invalidate a deferred request whose source storage changed.** Record the
   source's allocation/pointer at request time and refuse to alias at `alloc`
   time when it no longer matches, falling back to the fresh-allocation path that
   already exists at `var.cc:266-269`.

**The failure is jittor's multi-device resource path, and the last run names it
directly in jittor's own executor.** With the flash-attn bridge off, no offload,
the text encoder sharded (`JITTOR_FLASH_ATTN_JITTOR=0`, `OFFLOAD=`,
`TEXT_ENCODER_TP=2`, `ATTN=TORCH_SDPA`) and `NCCL_DEBUG=INFO`, the server starts
and loads 66.4 GiB per rank, NCCL logs the *correct* device for every communicator
(`rank 1 ... Setting affinity for GPU 5 ... localRank 1`), and the request then
dies on rank 1 with

    exec_runner.cc:402: Execute fused operator(0/35) failed
    cudaEventRecord((cudaEvent_t)event.handle, (cudaStream_t)stream.handle)
      -> cudaErrorInvalidResourceHandle(400)   [driver.cc record_event]

`record_event` opens with `CHECK(event.device.index == stream.device.index)`, and
that check passes -- so the two agree about the device while CUDA still rejects
the event handle on that stream. The event comes from the side-stream pipelining
machinery (`src/runtime/backend_streams.cc`), whose per-device `ready`/`done`
events are created by `get_resources(device)` and destroyed by
`cleanup_streams()` (registered as a cleanup callback, which also clears the
`resources` vector). So the handle is stale or belongs to another context; which
of the two is the thing to settle next, with a print of the handle and its
creating device at both `event_create` and `record_event` as the instrument.

Three jittor-side defects are now identified on this path, in the order they are
reached. None is fixed yet:

1. **A sub-group communicator is created without selecting the rank's device.**
   `nccl_create_process_group` (`backends/comm/nccl/src/nccl_wrapper.cc:536-560`)
   calls `init_nccl_comm` -> `ncclCommInitRank` with no `set_current_device` /
   `cudaSetDevice`, while the world path in `nccl_init` does both explicitly
   (`:592-597`). Rank 0 is right by accident (its device is already 0); rank 1
   creates the group communicator against whatever device is current. This is one
   observed failure on its own -- `ncclCommInitRank failed: unhandled cuda error`,
   reached from `_build_text_encoder_group` -> `init_world_group` -> `new_group`
   -- and jittor then reports it through the wrong checker, `checkCudaErrors` on
   an **ncclResult_t** (`:247`), which is why the text reads like a CUDA error and
   never mentions NCCL.
2. **The deferred share in `Var::share_with` is resolved too late** (see above):
   the alias is registered in the child's own `alloc`, against whatever owns the
   source's allocation id by then.
3. **A stale or foreign event handle reaches `cudaEventRecord`**, which is what
   the run above actually trips.

Two startup failures are separate from those and are *intermittent*, not
deterministic: the store rendezvous (above) and the sub-group comm init
(identical `JITTOR_FLASH_ATTN_JITTOR=0` runs: one died at
`_build_text_encoder_group`, the next came up). That intermittency is why the same
TP2 configuration has produced illegal address, unhandled CUDA error, invalid
resource handle and allocator-invariant failures across attempts, and why every
stack recorded before the last one was a candidate rather than the cause.

**Fix 1 landed, and what it did and did not change.** The sub-group device
selection is fixed in `backends/comm/nccl/src/nccl_wrapper.cc`: the path now does
`set_current_device(nccl_device_id)` + `cudaSetDevice(nccl_device_id)` before
`ncclCommInitRank`, mirroring `nccl_init`, and a failed `ncclCommInitRank` now
reports `ncclGetErrorString(result)` instead of handing an `ncclResult_t` to
`checkCudaErrors`. Verified: the NCCL extension rebuilds
(`Compiling jittor_nccl_core(9/9)`), the server starts, both ranks load 66.4 GiB —
and the request **still fails** with the same `cudaErrorInvalidResourceHandle` on
rank 1. So that fix is correct on its own terms and is *not* this symptom's cause;
the event-handle defect (item 3) is what this configuration trips.

**Two traps hit while landing it, both worth knowing.** First: the deployed tree
(`site-packages/jittor/`) is an older, self-consistent snapshot, so copying a
single repo file into it can break the build — the repo's `driver.cc` assigns
`ops.graph_capture_end`/`graph_launch`/`graph_release`, which the deployed
`backend.h` does not declare. The matching variant is the previous successful
build's snapshot in the repo's `build/lib/jittor/…`; restoring from there and
re-running a two-device probe confirmed the core rebuilds and runs. Second: the
NCCL extension compiles from the *deployed* sources and is cached under
`JITTOR_HOME/.cache/.../custom_ops/`, so a fix must be copied there and that
cache invalidated before a rebuild picks it up.

**Fix 2 landed and verified: the event-handle fault is gone.** The handle trace
settled it: the failing `cudaEventRecord` had a *valid* event (created on device 1,
recorded successfully ten times before, destroyed only after the fault), so the
invalid resource was not a stale handle; and substituting the device's compute
stream for the null one still failed, so it was not only the stream either. The
cause is that `on_device_void` drives jittor's own device switch
(`ops.set_device = accelerator_set`) and not a raw `cudaSetDevice`, so a thread
whose CUDA context is still the process default records a device-1 event on a
device-0 context -- which CUDA rejects while jittor's own
`CHECK(event.device.index == stream.device.index)` passes. `record_event` now
resolves a null stream to the device's compute stream and calls `cudaSetDevice`
explicitly, as `nccl_init` already did. Two independent verifications: the
no-flash/no-offload config went from `fused operator(0/35)` to `(24/34)` with no
`InvalidResourceHandle` (237 successful rebinds), and the flash+offload config
reached the denoise loop. Committed as its own change.

**What is still open, after that fix** (so the next session starts from the real
front line, not from this section's earlier lists):

   Confirmed as the *sole* blocker across every memory-adequate configuration:
   flash-attn + DiT/text-encoder offload (`acc-2`) and DiT-resident +
   text-encoder offload (`best-2`, `RESULT failed in 25.1 s`) both now reach the
   denoise loop after fix 2 and both die here; the no-flash/no-offload config
   instead runs out of memory. The fix-2 verification and this verdict come from
   the same runs, so the front line is unambiguous.
   **Not tied to the offload after all** -- a no-offload TP2 run at 512x512
   (text encoder TP-sharded, `OFFLOAD=`) fails with it too, earlier in the
   pipeline (~10 s, before the denoise loop, i.e. in prompt encoding). With the
   offload on it was reached only later, in the denoise loop, which is what made
   the offload look implicated; the earlier reasoning that the layer offload's
   `param.data =` swap is the trigger is therefore withdrawn.
1. `cudaErrorIllegalAddress` on rank 1 in the flash-attn + layer-offload config,
   now that the run gets into the denoise loop. This is the class the earliest
   TP2 attempts hit; with the event fault gone it is the first thing the request
   trips.
2. Out of memory in the no-offload + text-encoder-TP config at 832x480 (66.4 GiB
   resident per rank of 97 GiB, so the DiT activations have no headroom). A
   memory-budget question, not a correctness one.
3. The two intermittent startup failures: the TCPStore rendezvous (~1 in 3 runs)
   and the sub-group communicator init. `run_tp2_probe.sh` retries past them so
   they do not block diagnosis, but they are unfixed.

**The remaining fix, spelled out so it is one step next time.** The range-checked
variant of `share_with` is the one that should be landed, and the only reason it
was not is mechanical: widening the virtual to

    bool share_with(size_t size, size_t allocation, size_t offset = 0)

requires **every** override to move with it in the same commit. There are five,
across four headers, plus one call site:

    src/mem/allocator.h:55                            the virtual (add offset)
    src/mem/allocator/sfrl_allocator.h:150            SFRLAllocator (+ .cc def)
    src/mem/allocator/cuda_dual_allocator.h:67        CudaDualAllocator
    src/mem/allocator/cuda_dual_allocator.h:97        DelayFree
    src/mem/allocator/foreign_allocator.h:17          ForeignAllocator (+ .cc)
    src/mem/allocator/shared_allocator.h:50           SharedAllocator
    src/core/var.cc                                   Var::alloc passes share_offset Two more mechanical traps, both hit while trying to land it: the base virtual
must gain the default (`size_t offset = 0`) or every override is "marked
`override`, but does not override" -- the trailing `;` after the inline body makes
the obvious text replacement miss; and `cuda_dual_allocator.h:98` *calls*
`share_with(size_t, size_t)` internally, so it needs the extra argument too.
Widening only part of the set leaves the rest marked `override` without a
matching base, and the build stops with

    error: 'bool jittor::CudaDualAllocator::share_with(size_t, size_t)'
           marked 'override', but does not override

which is what happened. With the whole set changed, the check
`if (offset + size > block->size) return false;` makes a request whose range no
longer fits the block the allocation id names fall through to the
independent-allocation path instead of aliasing it -- which is the use-after-free
behind the illegal address. Verify with `probe_split_alias.py` (seconds, no
server) and then one `run_tp2_probe.sh` cycle.

**Fix 3 landed: the allocator invariant, verified at the repro.** `share_with`
now takes the request's `offset` and `SFRLAllocator` refuses a request whose range
no longer fits the block the allocation id names (`offset + size > block->size`),
falling through to the independent-allocation path `Var::alloc` already has.
Landing it meant widening the virtual *and* all five overrides across four
headers, the `DelayFree` internal forwarding call, the `Var::alloc` call site and
the C++ tests that call the two-argument form directly -- three failed builds
along the way, each stopping on one of those. Verified where it matters: 
`probe_split_alias.py` went from tripping
`sfrl_allocator.cc:305 mem_ptr does not belong to allocation` twice per cycle to
passing silently (`ALIAS RELEASE OK`).

**But it is not the only source of the illegal address.** The same TP2 request
that used to die at 20-25 s now runs **145 s** before failing with
`cudaErrorIllegalAddress` again -- much further in, but not through. So the
allocator defect was real and is fixed; at least one more cause of the same
CUDA error remains, and the run's new length is the evidence that the two are
distinct. The next instrument is the same handle/allocator trace on this longer
run, now that the run survives far enough to make the trace readable.

**Where the remaining fault now sits** (read off the 145 s run, so this is the
front line, not the older lists): the innermost frames are

    flash_attn/__init__.py:199 flash_attn_varlen_func
      -> _call_native :123
        -> adapter.py:355 flash_attn_varlen_func
          -> packed_low_level.varlen_fwd(...)
            -> sticky cudaErrorIllegalAddress, device=1

i.e. the packed varlen call in the DiT denoise loop, on **rank 1 only** -- the same
"second device only" signature as the record_event fault, and now reached after
145 s of work (text encoder and earlier steps all succeed) instead of at 20 s.
Two candidates follow from the shape of it, in order: the officially-built
flash-attn extension's per-device state on a non-default device (the sanitizer
already caught exactly that class inside NCCL, `cudaErrorNoKernelImageForDevice`
in `ncclInitKernelsForDevice`), or the packed path's own `cu_seqlens`/output
buffers on that device. `instrument_store_and_triton.py` plus the `H3_FA_*`
traces are the instruments; the run is long enough now to make them readable.

Two candidates for it are now *disproved by experiment*, both of which made the
run fail earlier than the 145 s baseline: binding the device from the shim's
packed branch (`torch.cuda.set_device(rank)` before `varlen_fwd`) -> 25 s, and
routing the packed path through the non-packed entry
(`JITTOR_FLASH_ATTN_DIRECT_PACKED=0`) -> 25 s. Do not retry either; the next
session should start from a fresh hypothesis about the packed call on rank 1.

**Five hypotheses measured away** (do not repeat these):

- *offload-related* -- `OFFLOAD=` reproduces it, earlier.
- *wrong ambient device at the extension boundary* -- at the packed call on
  rank 1: `jt_current=1`, `shim_current=1`, and every argument is either
  `loc=none` (lazy q/k/v) or `loc=device` (`cu_seqlens`). Nothing host-resident,
  and `torch::empty_like` therefore allocates on the right device.
- *invalid NCCL device index* -- measured at bootstrap: rank 1 has
  `CVD='4,5'`, `visible_n=2`, `jt_device_count=2`, `jt_current=1`, and
  `nccl_device_id = 1 % 2 = 1`, all valid.
- *device binding in the packed branch* (`torch.cuda.set_device(rank)`) -- 25 s,
  worse than the 145 s baseline.
- *the non-packed entry* (`JITTOR_FLASH_ATTN_DIRECT_PACKED=0`) -- 25 s, worse.

And `compute-sanitizer --tool memcheck` on a run that reaches the failure reports
**no invalid memory access at all** -- only three `cudaErrorNoKernelImageForDevice`
inside NCCL's `ncclInitKernelsForDevice`, which persists even though the device
index is valid. So this illegal address is not a data access memcheck can see.
That, plus the "second device only" signature, is what narrowed the next step to
synchronous launches (`CUDA_LAUNCH_BLOCKING=1`), which attribute a fault to the
launch immediately before it.

**The fault is now reproducible in seconds, without a server** --
`probe_packed_devices.py` calls the shim's bridged `flash_attn_varlen_func` on a
289x28x128 fp16 tensor with `cu_seqlens=[0,289,289]`:

    CUDA_VISIBLE_DEVICES=4,5  ... python probe_packed_devices.py
        --- device 0
            device 0 OK, out mean=1.000000
        --- device 1
            device 1 FAILED: cudaErrorIllegalAddress

and the two things that pin it down:

    CUDA_VISIBLE_DEVICES=5  (that card alone -> index 0)   -> OK
    CUDA_VISIBLE_DEVICES=4  (the other card alone -> index 0) -> OK
    CUDA_VISIBLE_DEVICES=4,5, call device 1 FIRST, then 0  -> device 1 fails

So it is **not the card and not the order**: the bridged flash-attn entry faults
whenever the CUDA device *index* is non-zero, in a plain script with no H3, no TP,
no NCCL and no offload. jittor's own ops on device 1 in that same script
(`jt.ones`, `.to_device(1)`, `device_copy`, the reshapes) all succeed, so this is
the extension/its bridge, not core device handling. That is the mechanism behind
every rank-1-only TP2 failure in this section.

Everything above about offload, ambient device, NCCL device index, the non-packed
entry and Python-side device binding is superseded as an explanation: those were
different views of this one defect, and it is now isolated. The next step is a
seconds-long bisection on this probe -- dense entry vs packed, and shim bridge vs
extension -- which is what the "five hypotheses measured away" list above could
not offer at 7 minutes per cycle.

One more observation from that probe, and it points *back into jittor* rather
than at the extension: with the tensors materialised first, the failure's launch
list names a **`device_copy`** -- `probe_packed_devices.py:35`, the `.to_device(1)`
line -- and the recent candidates are all `device_copy` (id=16) ops. So the
seconds-long repro suggests the illegal address may arise in jittor's own
cross-device copy to a non-zero index, before or around the extension call, which
would make it fixable in `src/ops/composite/device_copy_op.cc` /
`src/runtime/backend_streams.cc` rather than in the flash-attn bridge. Worth
establishing before touching the extension: bisect the probe to the copy alone
(`jt.ones(...).to_device(1)` in a fresh process) and see whether that faults by
itself. `probe_side_streams.py` copied across devices without faulting, so the
difference between the two scripts is the thing to pin down.

**Two seconds-long bisections that finish the isolation.** First, the copy path
is innocent -- `probe_dev1_copy.py` does only jittor's own work on device 1 and
every step passes:

    to_device(1) OK | op on device 1 OK | device 1 -> 0 OK | big fp16 to_device(1) OK

so the launch list naming `device_copy` was just naming the most recent op.
Second, it is not the packed path either -- the **dense** entry fails the same way:

    DENSE dev=0 OK mean=1.0000
    DENSE dev=1 FAILED cudaErrorIllegalAddress

So the characterization is now: *jittor's own device-1 work is fine, and the
bridged flash-attn boundary faults for both entries whenever the CUDA device index
is non-zero, on either card.* The remaining bisection is which side of that
boundary: the shim's generic extension machinery (in which case a triton kernel or
the shim's own cpp_extension on device 1 is the comparison, and the fix is central
and probably small) versus something the flash-attn extension caches at load. That
is a few seconds per run now, not seven minutes.

**A first candidate, tried, that turned out to be a red herring.** The obvious
suspect was `compat/shim/cpp_extension/include/c10/cuda/CUDAStream.h`, which ignored
the device and returned the legacy default stream:

    inline CUDAStream getCurrentCUDAStream(int = -1) { return CUDAStream((cudaStream_t)0); }

Handle 0 is the *current context's* legacy stream, so an extension running on
device 1 would launch on a device-0-context stream while its pointers belong to
device 1. It was changed to return `cudaStreamPerThread` for the device the caller
names, binding that device first, and the flash-attn extension was force-rebuilt
(`JITTOR_FLASH_ATTN_FORCE_BUILD=1`, ~30 min). **It did not fix it**: device 0 still
passes and device 1 still fails with the same illegal address. The change is
reverted in both trees rather than left in unverified.

That accessor was in fact already correct. Handle 0 is not a device-0 stream, it
is "the current device's default stream", so once the *current device* is the
tensor's device the launch lands on that device. The stream was never the
mechanism; the current device was, and the fix belongs one level up (below).

**Both device-binding candidates were right, and each was only half the fix.**
Neither was really disproved -- they were each measured in isolation, and the
fault needs both.

`torch::Tensor::device()` and `get_device()` hardcoded index 0 for CUDA tensors
(`compat/shim/cpp_extension/include/torch/extension.h`), so
`at::cuda::CUDAGuard device_guard{q.device()}` -- which every flash-attn entry
guards with at its top -- bound device 0 for a device-1 tensor. Reporting the
Var's real index is necessary, and was measured alone: still
`device 1 FAILED cudaErrorIllegalAddress`.

It is not sufficient, because of the second half. The shim's tensor factories --
the `torch::empty` calls an extension makes for `out`, `softmax_lse`,
`rng_state`, `softmax_lse_accum` and `out_accum` -- build their Var from
*jittor's* current device, and `cudaSetDevice` does not move that one:

    jt.current_device() == 0        # also after jt.ones(...).to_device(1)
    jt.flags.device_id = 1  ->  jt.current_device() == 1

So the index fix alone leaves those buffers on device 0 while the kernel runs on
device 1, and moving the ambient device alone leaves the kernel on device 0 while
its inputs and buffers are on device 1. Either way a kernel reads or writes memory
that does not belong to the device it runs on: `cudaErrorIllegalAddress`, for any
non-zero index, with index 0 unaffected because there both notions of "current
device" already agree.

**The fix, and why it is the shim's to make.** Keep torch's two notions of
"current device" in step:

- `compat/shim/cpp_extension/include/torch/extension.h`: `device()`/`get_device()`
  report the Var's real index, via `vh_device_index` (defined in
  `src/jtorch_aten.cu`).
- `compat/shim/cpp_extension/include/c10/cuda/CUDAGuard.h`: `CUDAGuard` and
  `OptionalCUDAGuard` also switch jittor's current device, through the out-of-line
  helpers `accelerator_current_device`/`accelerator_set_current_device`. They stay
  out-of-line because the public ABI header must not pull jittor headers in.

No downstream library changed: `CUDAGuard` still means what libtorch means by it,
it just moves both devices, and it restores both when it goes out of scope.

**Verified on two visible devices.** `probe_packed_devices.py`, dense and packed:

    device 0 packed OK | device 1 packed OK | device 0 dense OK | device 1 dense OK

plus `probe_hdim_dev.py` (head dim 64 and 128) on both indices, `cuda:1` called
*before* `cuda:0`, and both ambient-device variants of `probe_ambient_dev.py`.
`probe_dev_parity.py` checks values rather than just absence of a fault: the two
indices agree with each other and with a plain-jittor
`softmax(q k^T * scale) v` reference to `max|diff| = 2.7e-4` (fp16 noise).
`compat/tests/torch/test_cpp_extension_device_index.py` builds a probe extension
that reports both halves and passes 6/6.

**Next instrument, prepared but not yet run.** The event-handle defect needs the
handle's provenance: log device + handle at `create_event`, `destroy_event` and
`record_event` (`backends/cuda/runtime/driver.cc`, `record_event` is where the
400 comes from) plus every `get_resources`/`cleanup_streams` in
`src/runtime/backend_streams.cc`. Apply that trace to the *deployed-matching*
`driver.cc` variant, not the repo's newer one, or the build breaks as above. That
one run distinguishes "the handle was destroyed by `cleanup_streams` while an op
still referenced it" from "the handle belongs to another device's context", which
are different fixes.

**Do not reuse a per-step number for TP2.** The loop's progress bar reached
`0/7` before the fault, so there is no measured TP2 step time. An earlier
version of this section recorded 6.4 s/it as the TP2 denoise rate; that figure
came from a *VAE shard-loading* bar in the same log (denominators `/13`, `/14`,
which the single-GPU log carries too) and is withdrawn here rather than left for
someone to quote. The single-GPU 10.14 s/it at 832x480 is from the real denoise
bar (`/49`).

So: **single-GPU serving is done and verified; TP2 boots, shards and starts
denoising, but is not yet correct.** `--num-gpus 1` remains the profile the
frontend is served on.

## 14. `linspace` did not land on `end`, and every >= 4-step request died

Found by serving the model and asking why the picture looked wrong. Two separate
things were wrong, and only one of them was jittor.

**The jittor defect.** `linspace` built its series as
`i * (end - start) / (steps - 1) + start`, so the last point was a rounding
result rather than `end`. On the **CPU placement**:

| n | last value of `linspace(1.0, 0.0, n)` |
| --- | --- |
| 2 | `0` |
| 4 | `-2.98e-08` |
| 6 | `-1.49e-08` |
| 8 | `-4.47e-08` |
| 50 | `+2.05e-08` |

numpy and torch both guarantee the endpoint exactly, and callers compare it.
MiniMax-H3 builds its sigma schedule from `linspace(1.0, 0.0, num_inference_steps)`
and validates `sigma_next >= 0`, so **every request with 4 or more denoise steps
failed** with `ValueError: sigma_next must be non-negative`; 2 steps happened to
round to exactly 0 and escaped. The CUDA placement also rounded to 0, which is
why only the CPU one showed it. Fixed by pinning the endpoint
(`jt.cat([res[:-1], jt.array([end], dtype=res.dtype)])`), with tests in
`tests/ops/test_random_op.py`.

**The configuration defect, and it was ours.** The lab workflow set
`num_inference_steps = 2`. The sanctioned value is **50** -- every curl example
and the benchmark table in `recipes/MiniMaxAI/MiniMax-H3.md`, and
`tests/e2e/accuracy/minimax_h3` (`NUM_INFERENCE_STEPS = 50`, `FLOW_SHIFT = 12.0`,
`AUDIO_FLOW_SHIFT = 3.0`, SSIM >= 0.97 against a reference). 2/4/8 steps produce
undenoised output that looks like texture; 50 steps produce a real scene, and the
audio tells the same story (rms 0.57 with peaks above 1.0 at 2 steps -- clipped
noise -- versus rms 0.06 at 50).

**Two corrections to earlier sections of this page.** The sub-second diagnosis
that followed the "wrong picture" report was wrong twice over, and both are worth
recording because the reasoning was tempting:

* Section 11's table and the `_same_floating_dtype` analysis stand, but the
  "671 s of outer product" attribution was measured before the sigma bug was
  fixed, so the slow run and the crash were **different** faults, not one.
* More importantly: several intermediate conclusions drawn while investigating
  the wrong picture -- "the remote VAE decode is broken", "it is the tiling",
  "it is the two Triton kernels" -- **were void**. They came from decoding
  `real512.latents.npy` standalone, and that array is not in the space
  `decode_latent` expects. Decoding the same array on **real PyTorch** produces
  the *same* 16-px mosaic (mean 85.7 / std 63.7 / hf 13.41 against the shim's
  85.7 / 63.7 / 13.40). A probe whose input has not been validated cannot be
  evidence; the control has to come first.

The decode was then checked the right way -- on the latent the pipeline itself
produced -- by capturing `decode_latent`'s input in-process and decoding it under
real PyTorch. The shim and torch agree to **0.30/255 at 256x256 and 0.31/255 at
512x512**, so **jittor's VAE decode is correct** and nothing in the decode path
is implicated.

## 15. Speed: what 180 s is made of, and the two levers

256x256, 50 steps, one H20:

| profile | denoise | total |
| --- | --- | --- |
| offload `dit` + `text_encoder` (layer mode) | 3.35 s/step | 179 s |
| offload `text_encoder` only, DiT resident | **1.68 s/step** | **145 s** |
| 512x512, offload `dit` (resident OOMs: 91.65 of 95 GiB) | ~12 s/step | 627 s |

The reference in the recipe is **0.73 s/step at 832x480** -- six times these
pixels -- with `FLASH_ATTN`. So per pixel this is roughly 10-15x slow, and the
first lever is the one already taken: with the DiT offloaded, every denoise step
streams all 50 blocks over PCIe, which costs about the same as the step itself.
Keeping the DiT resident halved it. The second lever is the attention backend:
`TORCH_SDPA` is what this shim has been run with, because `FLASH_ATTN`

    RuntimeError: TORCH_CHECK failed: x.is_cuda() cu_seqlens_q must be on CUDA

from `flash-attention/csrc/flash_attn/flash_api.cpp:20` -- the **real torch CUDA
extension** is being called with tensors it does not accept. The mask/varlen
helper that builds `cu_seqlens` keeps the CUDA placement at every step
(`sum(dtype=int32)` -> `nonzero` -> `cumsum(dtype=int32)` -> `F.pad` all stay on
`cuda:0`, verified), and so do `torch.arange(device=...)`, `.to(cuda)`,
`.cuda()` and the bool factories, so the placement is not lost in Python --
which points at the extension resolving to the un-bridged module.

That leads to the real blocker, and it is a **missing shim module**:

    [w] integrations.py:10 external runtime patch vllm skipped:
        cannot import name 'vllm' from 'jittor.compat'

`compat/integrations.py` tries `from jittor.compat import vllm` and calls
`_vllm_compat.register()`, and `compat/torch/__init__.py` lists an optional
`jittor_vllm` step with the same expectation. **`compat/vllm.py` does not
exist**, so the integration is skipped on every run. A vLLM-Omni deployment is
exactly what that step is for, and the flash-attn routing above is the kind of
thing it would own. Implementing it is the next real piece of work in this area
-- it is not a one-line change, and it should not be guessed at.

## 16. Every resolution above 256x256 lost the scene: a device Var parked on the host

Fixed. The cause was the attention backend *and* the reason it could not be
selected, and the two were one defect in the shim.

**It is not a resolution problem.** 256x256 at 50 steps denoises to a real scene
(hands pouring beans onto a white surface, coherent motion). 512x512 and the
recipe's own **832x480** at the same 50 steps both come out as the 16-px mosaic
(mean 82.5 / std 54.5 for 832x480, against 209.3 / 80.4 for the working 256).
So the model fails as the answer gets *bigger*, which is where the packed
sequence gets longer -- 2816 tokens at 256x256, 9920 at 512x512.

The loop itself is fine at 512: `update_mask` is all-True, `cu_seqlens` is on
device, the initial latent has std 1.0008 -- but the latent moves **0.30** over
two steps where the 256 run moves **1.21**, and leaves with std 1.0620 (still
noise) against 0.9433. The denoiser is producing a much weaker update, not none.

**The cause is the attention backend.** `FlashAttentionBackend` declares

    supports_multi_doc_packed_varlen(): True for CUDA
    supports_packed_mask_free():        True for CUDA

with a comment spelling out why: the CUDA row dispatches
`_forward_varlen_packed -> flash_attn_varlen_func` **over the caller's
`cu_seqlens` without a mask**, "so an arbitrary N-document packing keeps its
boundaries", and the NPU row only accepts a `[real, pad]` two-document layout
and otherwise "silently attends across request boundaries". Running with
`TORCH_SDPA` gives up exactly that guarantee: the packed row holds separate
text, audio and video documents, and the fallback mask does not keep them
apart. The more tokens, the more cross-document mixing, which is why 256x256
survives and 512x512 / 832x480 do not.

That is also the whole speed story: without a varlen kernel, `TORCH_SDPA`
materialises the score matrix, and 832x480 runs at **25 s/step** (1236 s for 50
steps) against the recipe's 0.73 s/step. The two open items were one item, and
selecting FLASH_ATTN closed both -- see the numbers at the end of this section.

**Why `FLASH_ATTN` could not be selected.** It failed with

    TORCH_CHECK failed: x.is_cuda() cu_seqlens_q must be on CUDA

from `flash-attention/csrc/flash_attn/flash_api.cpp:20` -- a check in the real
libtorch extension. Three things were wrong, and all three are now fixed or
accounted for.

The import *was* wrong and *is* fixed in the lab:
`vllm_omni/diffusion/attention/backends/utils/fa.py` does

    from flash_attn_interface import flash_attn_varlen_func

-- the **top-level** name, which is how the upstream package spells it -- while
the shim ships the bridged implementation only as
`flash_attn.flash_attn_interface`. With the flash-attention checkout on
`PYTHONPATH` the top-level import therefore resolved to the unbridged extension.
A two-line top-level alias that re-exports the bridged functions fixes the
resolution (verified: `flash_attn_varlen_func.__module__ == "flash_attn"`,
`is_flashattn_jittor_available() == True`, backend
`flashattn_jittor_official:/root/jittor-lab/flash-attention`). It is not landed
in the repo yet because it is a deploy-surface change: the stub lists in
`compat/tests/structure/test_torch_shim_structure.py` and
`test_torch_shim_deploy.py` would have to carry it.

**The real cause is a parked Var.** jittor distinguishes *belonging* from
*being*: `device_id` names the device a Var belongs to, `Var.location()` says
where its bytes are, and the executor parks a device Var in host memory whenever
a CPU op consumes it (`src/core/exec_runner.cc:319-322` migrates a CPU op's
inputs to the host and deliberately keeps `device_id`), moving it back when a
device op consumes it again (`:334-336`). A tensor handed straight to an
extension never passes through that per-op migration.

The pipeline parks `cu_seqlens` on its own, with one host read:

    cu = packed["cu_seqlens"].to(torch.int32)   # denoise_loop.py:87
    used = int(cu[1])                           # denoise_loop.py:92 -- parks it
    cu.to(device)                               # denoise_loop.py:115

`int(cu[1])` takes the `item()` path, which ends in `migrate_to_cpu` when
`save_mem || _HAS_ACCELERATOR` (`src/core/var_holder.cc:709`), and views migrate
with their group, so the base `cu` is parked too. The next line asks for the
device and got the *metadata* answer instead: `_make_cuda_resident` returned
early on `v.placement_backend == backend and v.device_id == index`
(`compat/torch/types.py:564`) and `_move_to_cuda_index` on `current == idx`
(`:450`), both of which are true for a parked Var because parking keeps both
fields. The shim reported the tensor as already being on `cuda:0` while its
bytes were on the host, so the extension correctly refused a host pointer.

Reproduced standalone in six lines (`probe_cu_parking.py`), which is also the
regression test: after `int(cu[1])` the Var reads `device=cuda:0, loc=cpu`, and
`.to("cuda:0")` used to leave it that way.

**The fix** is for both shortcuts to require real residence, which is what the
native move already does -- `Var.to_device` refuses to skip unless
`location() == "device"` (`python/jittor/_core/var.py:194`, with a comment
describing this exact failure mode for `x.cpu().cuda()`). A parked Var now takes
the device copy instead of the shortcut.

**What it bought.** FLASH_ATTN runs, and the packed CUDA varlen path is what the
higher resolutions needed:

- 512x512, 50 steps -> a real, sharp scene. Previously the same request was the
  16-px mosaic, and so was 832x480.
- 832x480, 50 steps -> a photorealistic, coherent scene; 560 s end to end, and
  **10.09 s/it** in the denoise loop against TORCH_SDPA's 25.2 s/it on the same
  resolution (2.5x). The mosaic this resolution used to produce is gone.
- 512x512 per-step cost: **6.69 s/it** against TORCH_SDPA's **11.76 s/it**
  (same request, same seed, layer-offloaded `dit`+`text_encoder`), and 380 s for
  the whole 50-step request. TORCH_SDPA's number is not a correct-output
  baseline: it still loses the packed multi-document boundaries, so at 512 it
  is faster-per-step than nothing only in the sense of finishing.
- 256x256 still works on both backends, and at 2 steps the two agree
  pixel-for-pixel in character, which is the control that says the FLASH_ATTN
  path did not change the small case.
- The control that decides attribution: at 256x256 / 50 steps / seed 11223 the
  long structured pottery prompt produces the same dotted-tile texture on
  **both** backends -- the TORCH_SDPA run and the FLASH_ATTN run are the same
  picture. That texture is therefore a prompt-and-seed property of the model at
  256, not a backend defect, and it is why the conclusion above rests on 512,
  where the two backends genuinely differ.

**Not chased here.** `TORCH_SDPA` above 256x256 remains wrong by design: the
CUDA row's guarantee is what FLASH_ATTN provides, so the serving profile now
selects FLASH_ATTN.

One trap worth fixing while in there: `_official_import_identity` keys the built
extension on the **build directory name and a generation counter, not on the
source**. A changed `.cu` or `.h` is therefore silently ignored and the old
kernel keeps running -- `JITTOR_FLASH_ATTN_FORCE_BUILD=1` is the only way to get
a rebuild.

**A pre-existing failure found on the way, left alone.**
`compat/tests/torch/test_multi_device.py::TestMultiDeviceFacade::test_to_and_cuda_with_an_index`
asserts that a bare `.to("cuda")` leaves a `cuda:1` tensor on `cuda:1`, and
fails identically before and after this change. The bare-name path resolves the
index from `torch.cuda.current_device()` (`compat/torch/frontend.py:55`), so it
only holds when the ambient device happens to be 1 -- which a full file-order
run arranges and an isolated run does not. It is a separate bug from this one
and is not touched here.

## 17. The extension build cache ignored header edits, and served a stale `.so`

Chasing section 13 cost two things that were *not* the bug, and both came from
one defect in the shim's build cache.

While diagnosing, the deployed flash-attn extension stopped importing at all:

    compile official flash-attn backend failed: .../flash_attn_2_cuda_jittor...so:
    undefined symbol: _ZN6jtorch6detail15vh_device_indexEPN6jittor9VarHolderE

`vh_device_index` had only ever existed in an uncommitted experiment of mine, and
had been reverted. The `.so` still referenced it, and the build had reported every
object `up-to-date` while it did.

The up-to-date checks compare an object's **source mtime** and its **compile
command** (`_object_matches_command`, `_output_matches_build` in
`compat/shim/cpp_extension/__init__.py`). Neither looks at headers. Editing
`torch/extension.h` or `c10/cuda/CUDAGuard.h` therefore invalidated nothing: the
objects compiled against the old text were reused and relinked, and the first
symptom was an `undefined symbol` at import -- or, worse, a silently stale
extension that re-measures the *old* behavior. The build directory is keyed by the
flash-attention checkout's path, git HEAD, head dims and dtypes, so a shim-side
header change does not even move the directory.

Two practical consequences worth knowing:

- The old workaround is to force the build (`JITTOR_FLASH_ATTN_FORCE_BUILD=1`,
  ~30 min) or to `touch` each affected source so its mtime wins. `touch` only
  covers the sources that include the header, which is the trap.
- Every measurement in this document that follows a shim header edit has to say
  how the extension was rebuilt. The two device-binding candidates in section 13
  were each force-rebuilt, so those results stand; the stale `.so` above came from
  a third edit that was *not* rebuilt.

**Fix.** `build()` now digests the shim's own ABI headers (14 files, 76 KB) and
passes the digest as `-DJTORCH_SHIM_ABI=<digest>`, and records it in the link
stamp. A header edit changes the command, the command is part of the object's
identity and its stamp, so what depends on the header recompiles and the object
files are not reused behind a "up-to-date" line.

Two notes on landing it: it invalidates every existing extension cache exactly
once (the command changed for all of them), and `compat/tests/torch/
test_torch_cpp_extension.py::TestShimHeadersInvalidateTheBuildCache` locks the
behavior with a faked compiler -- unchanged tree compiles nothing, edited header
recompiles and re-links.

## 18. Stale lab processes are a first-class failure mode

Two different stale processes each produced a wrong-looking result during this
work, and neither was a jittor defect:

- **Orphaned workers hold the rendezvous.** `serve-vllmomni.sh`'s
  `DiffusionWorker` children do not carry `--port` in their argv, so a kill by
  port string leaves them alive holding `MASTER_PORT`; the next run's rank 1 then
  rendezvouses with the orphan's store and dies with an intermittent "NCCL store
  rendezvous timeout". Three such orphans from earlier TP2 attempts were still
  resident (18-19 h old, each spinning at 100% CPU) when this section was written;
  `stop-vllmomni.sh PORT` walks the process tree from the parent and clears
  `/tmp/jittor-nccl-*` for exactly this reason. Check `ps -eo pid,ppid,pcpu,args`
  before blaming a jittor startup failure.
- **A leftover server answers the health check.** A 20 h old single-GPU server
  still listening on a port makes `curl /v1/models` succeed, so a retry loop can
  "come up" instantly and send its request to the wrong process, with the wrong
  config and the wrong build.

## 19. The intermittent TP2 startup deadlock, from the store's own trace

The startup flake of section 13 -- "NCCL store rendezvous timeout: rank 1 waited
120 s", about one run in three -- is a GIL deadlock, and the store trace settles
it. `compile_extern.py`'s rendezvous is

    if world_rank == 0: store.set(unique_id_key, ...)
    unique_id = store.get(unique_id_key)
    nccl_module.nccl_init_with_unique_id(list(unique_id))

and `nccl_init_with_unique_id` is a *collective*: it parks until every rank
arrives, and the pyjt wrapper holds the GIL for the whole of it. Rank 0's process
is also the one running the store's server threads, so while rank 0 sits in
`ncclCommInitRank` no Python thread in that process can run -- including the one
that would answer rank 1's `get`.

With `H3_STORE_TRACE=1` the wire shows exactly that, on a hung run:

    pid=692276 (rank 0, server)  server -> 29 bytes for 'set'
    pid=692276 (rank 0)          client -> 'get'
    pid=692276 (rank 0, server)  server -> 199 bytes for 'get'
    pid=692276 (rank 0)          client <- 199 bytes for 'get'
    pid=692278 (rank 1)          client -> 'get'
                                 ... and no server line for it, ever

so the request reaches a process whose server thread never dispatches it, while
`ps` shows rank 0 at 99% CPU (parked in the collective) and rank 1 at 6% (blocked
in `readline`). The earlier readings in section 13 -- "rank 0 past its own `get`",
"no server thread in `_dispatch`", "bytes not in flight" -- are all this one fact:
the server thread cannot get the GIL. It is intermittent because it only happens
when rank 0 reaches the collective before rank 1's `get` has been served.

**Fix.** Rendezvous before the collective as well as after it: every rank records
that it has *read* the unique id, and nobody enters the collective until every
rank has. `store.wait` is a socket read, which releases the GIL, so the server
keeps serving while a rank waits there. jittor already has the other fix this
shape of bug wants -- `GILReleaseScope` in `src/bindings/pyjt/gil.h`, used by the
parallel compiler and the executor's device waits -- but it cannot be used here:
the NCCL core is compiled by `compile_custom_ops` without Python headers on the
include path and dlopened, so the release has to be at the collective's call site
in Python.

**What that leaves.** With the deadlock fixed the server starts, loads and reaches
the request, which then fails in 10 s with `cudaErrorIllegalAddress` on rank 1.
The traceback finally names the call -- the launch list is stale and says
`encoder.py`, which is where the rotary runs, not where the fault is:

    out = fn(
    return packed_low_level.fwd(q, k, v, float(softmax_scale), bool(causal), wl, wr)
    RuntimeError: ... cudaErrorIllegalAddress

i.e. the text encoder's `F.scaled_dot_product_attention` reaching the shim's
*packed* flash-attn entry. That entry is exercised in isolation by
`probe_encoder_sdpa.py` -- the transposed views, the GQA `repeat_interleave`, the
rotary `cat`, `is_causal=True`, head dim 128, 32 query and 4 KV heads per rank,
fp16 and bf16, sequence 489/512/1000/1023 -- and it passes on both devices. So the
remaining difference is not the call itself but the state the server is in when it
makes it: two ranks holding tens of GB each, after a multi-threaded sharded load,
on device 1 only. That is the next thing to reproduce.

## 20. The generated direct attention entries pinned the launch to device 0

The request-path fault of section 13 has one more layer, and it was in jittor's own
code generator rather than in the extension.

Official flash-attn's dense entry guards its launch with the input's device
(`at::cuda::CUDAGuard device_guard{q.device()}` in `csrc/flash_attn/flash_api.cpp`).
The shim generates six direct entries of its own
(`compat/shim/backends/flash_attention/official_codegen.py`: `fwd`, `varlen_fwd`,
`varlen_qkvpacked_fwd`, `varlen_kvpacked_fwd`, `qkvpacked_fwd`, `kvpacked_fwd`) and
every one of them emitted

    at::cuda::CUDAGuard device_guard{0};

so a call with device-1 tensors launched device 0's kernel against device 1's
pointers. Index 0 worked; every non-zero index died with
`cudaErrorIllegalAddress` -- rank 1's text encoder, ten seconds into the request.

**Why the earlier probes missed it.** The direct entries are only used when
`_grad_enabled()` is false, i.e. in inference. Every probe so far ran with grad
enabled, so `--triton or native` attention took a *different* implementation and
the packed entries were never called. Turning on `jt.flags.no_grad = 1` in
`probe_encoder_sdpa.py` (same shapes, transposed views, GQA expansion, rotary,
causal, head dim 128, 32/4 heads, fp16 and bf16) reproduced the server's fault on
device 1 immediately, and the same command passes on devices 0 and 1 after the
six guards were changed. `H3_FA_TRACE=1` in the adapter prints the tensors each
direct call receives, which is how the two paths were told apart.

**Verified.** `probe_encoder_sdpa.py` with `no_grad`: device 0 and device 1 both
pass (before: device 0 passed, device 1 raised `cudaErrorIllegalAddress`).
`compat/tests/torch/test_flash_attn_compat.py::TestPackedEntryDeviceGuard` asserts
the generated source contains no `device_guard{0}` and that each of the six
entries derives its guard from an input tensor.

**Still intermittent, and separate.** Weight loading on rank 1 sometimes dies with
the same `cudaErrorIllegalAddress` before the request: `safetensors.get_tensor` ->
`jt.array` is where it surfaces, the launch list names the TP weight loaders, and
it happens with and without `CUDA_LAUNCH_BLOCKING=1` (so it is a race, not a
serialization artifact) and on cards with ~95 GB free (so it is not memory
pressure). A single-threaded and an 8-thread replay of every loader pattern at the
real sizes passes, so it needs the server's own multi-threaded load context.

## 21. The mirror image: rank 1's index tensor cannot be read at all

Section 13 left one hypothesis untested -- "*the `_indexed_scale_shift_kernel`
masks columns but not rows, so a grid whose row count exceeds any operand's is an
unmasked out-of-bounds read*". With `H3_TRITON_SHAPES=1` (plus a min/max print for
integer operands, added to the same trace) the first direct-kernel launch of the
denoise loop now reads:

    [trishape] _rms_norm_indexed_scale_shift_kernel grid=(3072, 1, 1)
      output_ptr[3072, 5376]@dev1  x_ptr[3072, 5376]@dev1  weight_ptr[5376]@dev1
      shift_ptr[3, 5376]@dev1  scale_ptr[3, 5376]@dev1  indices_ptr<RuntimeError>
    [trishape] _rms_norm_indexed_scale_shift_kernel grid=(3072, 1, 1)
      output_ptr[3072, 5376]@dev0  x_ptr[3072, 5376]@dev0  weight_ptr[5376]@dev0
      shift_ptr[3, 5376]@dev0  scale_ptr[3, 5376]@dev0  indices_ptr[3072]@dev0{min=0,max=2}

Two things fall out of that.

*The shapes are not the problem.* The grid is the row count of `x` (3072), the
kernel masks columns to `hidden_size`, and the index values on the rank that
works are `0..2` -- exactly the three rows of `shift`/`scale`. Nothing walks out of
bounds *if* the index buffer holds what it should.

*The index buffer is the problem, and only on device 1.* Reading its values there
raises a `RuntimeError` (`.numpy()` on the Var fails) before the kernel is even
launched, while the same tensor on device 0 reads `min=0,max=2`. So the kernel is
handed a pointer to a buffer whose contents were never valid, `index` comes back
as garbage, and `shift_ptr + index * stride_shift_row` lands wherever that garbage
points -- an unmapped page on device 1, which is the `cudaErrorIllegalAddress`. It
also explains why the fault is invisible to a *shape* check, why it is
intermittent (it depends on what the unreadable buffer happens to contain) and why
device 0 is immune: there the tensor materialises.

The trace prints only the exception's type; widening it to the message is the next
one-line measurement. That message names the jittor path that leaves a device-1
Var unreadable.

**Interleaved, and separately annoying:** the *startup* now stalls about one run in
two at `diffusion_worker.py:327` (both workers log the final IR-op-priority line,
then rank 0 spins at 100% CPU and rank 1 idles; no store trace at all, so it is the
`NCCL(env)` init path and not the TCPStore that section 19 fixed). The runner
retries past it.

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
- 16: `compat/tests/torch/test_native_tensor_placement.py::test_a_device_var_parked_on_the_host_is_moved_back_by_to_device`
  -- the new case, which fails pre-fix on `assert back.device_id == 0 and
  back.location() == "device"` and passes after. The whole file passes (8/8),
  as does `test_multi_device.py` (15/16, the one failure pre-existing and
  isolated-run-only, see section 16). End-to-end: the 512x512 50-step request
  that used to be the 16-px mosaic now renders a real scene, on a server whose
  log carries zero `cu_seqlens_q must be on CUDA`.
- Integration: with every fix above in the working tree, engine construction
  reached `[try] ENGINE-CONSTRUCTED` at 10.0 GiB peak during load and shut
  down cleanly (`vllmomni-try18.log`).
- 6: `compat/tests/torch/_torch_compat_checks.py` (`x.data =` checks) and a
  standalone probe: empty placeholder, shape restore, dtype change,
  `requires_grad` preserved.

- 13: `compat/tests/torch/test_cpp_extension_device_index.py` (6/6) builds a probe
  extension and checks both halves on every visible index -- `device().index()`,
  the CUDA device a guard actually binds, where a fresh tensor lands inside the
  guard, and that the guard restores the device it found. Behavior: the flash-attn
  bridged entry passes on devices 0 and 1, dense and packed, head dim 64 and 128,
  in either call order, and agrees with a jittor `softmax(q k^T * scale) v`
  reference to fp16 noise.
- 17: `compat/tests/torch/test_torch_cpp_extension.py::TestShimHeadersInvalidateTheBuildCache`,
  which fakes the compiler and asserts an unchanged tree compiles nothing while an
  edited shim header recompiles and re-links.

- 19: the deadlock is removed by construction (nobody enters the collective before
  every rank has read the id) and a TP2 start that previously failed ~1 run in 3
  now came up, loaded the sharded weights and served a request. The request fault
  above is still open.

- 20: `probe_encoder_sdpa.py` under `no_grad` -- the condition that reaches the
  generated entries -- fails on device 1 before the six guards are fixed and
  passes on devices 0 and 1 after; the generator's output is asserted directly by
  `TestPackedEntryDeviceGuard`.

- 21: the shape trace above, from a 4-step 256x256 request on two cards.

Each fix was synced into the lab venv at
`$JITTOR_LAB_ROOT/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/`,
which is a **copy** of `python/jittor` plus the installed `jittor-torch`
compat package; the adapter checkout is editable.
