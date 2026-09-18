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

**Follow-up: one phase was not enough.** The read barrier above still hung about
one start in two, and py-spy named both halves at once:

    rank 0 (server, 101% CPU, active+gil)
      _init_nccl_from_store (compile_extern.py:703)   <- nccl_init_with_unique_id
    rank 1 (idle)
      readinto (socket.py:720) <- request <- wait
      _init_nccl_from_store (compile_extern.py:698)   <- the read barrier's wait

Rank 1's own barrier `wait` is answered by rank 0's store server threads, and rank 0
had already entered the collective -- which holds the GIL for its whole duration --
so the answer could never be produced. "Everyone has read the id" does not imply
"nobody still has a request in flight": the marker is set *before* the waiting rank
gets its reply. The barrier now has a second phase -- each rank records that it has
*completed* the barrier, and nobody enters the collective until every rank has -- so
when the first rank parks in the collective no peer has an unanswered request.
Verified by three consecutive starts (two passed the rendezvous in 30 s, where
previously roughly every other attempt hung; the third was killed by the test
script's own back-to-back servers, not by the barrier).
`tests/distributed/test_nccl_store_rendezvous.py` now pins both phases.

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

Widening that trace to the message settles which of the two it is, and the answer
is neither of the above:

    indices_ptr<RuntimeError: helper_cuda.h:135: CUDA error at .../driver.cc:176
      code=700( cudaErrorIllegalAddress ) cudaMemGetInfo(&free, &total)>

Reading the values raises the *same* illegal address, at the allocator's
`cudaMemGetInfo` -- the first CUDA call made after an async fault. The trace runs
*before* the launch, on the first denoise kernel, so the device context was already
dead when the denoise loop started: the index read is a victim that reports the
error, not its cause, and the "garbage index" reading above is withdrawn. The fault
is in the pre-attention work of the first DiT block on rank 1, and the launch lists
of those runs name `nccl_all_gather` (TP communication) alongside `cublas_matmul`,
`reshape` and `fused_op` -- that is where the next instrument belongs: log device,
stream, counts and pointers on both ranks for the first all-gather of the denoise
loop. Nothing in the attention path is implicated any more; every fault seen after
the three fixes above happens *before* the first kernel of the loop.

**A trap worth naming, because it cost three runs:** that stall -- both workers
logging the final IR-op-priority line and then no progress, rank 0 spinning at 100%
CPU and rank 1 idle -- looks exactly like the section 19 deadlock at a different
call site, and it is not. It appeared only while `instrument_store_and_triton.py`'s
**store** trace was applied, and vanished the moment `distributed/store.py` was
restored from the repo. `_strace` writes a line per request and response on both
sides inside the store's own request/response path, and the rendezvous it is there
to observe is sensitive enough that the extra work deadlocks it. So: instrument the
store to *diagnose* the rendezvous, never while measuring anything else, and
restore `store.py` before the next run. The triton half of that script is harmless
(it only fires on kernel launches, which happen after startup).

## 22. The weight-load crash was a null allocator in `ArrayOp::run`

The intermittent `cudaErrorIllegalAddress` during rank 1's sharded weight load had
a second, deeper cause, and it is a plain null dereference in jittor's own core.

`jt.array()` is what the shim's `torch.tensor(numpy, dtype=..., device="cpu")` path
becomes -- i.e. what a safetensors load does for every tensor, from four threads at
once. `ArrayOp`'s output is created by the op itself (`create_output` gives it shape
and dtype; the `_force_fuse`/scalar shapes take the element path and never get an
allocation), and `run()` then freed "the previous allocation" unconditionally:

    if (save_mem) free_with_swap(o);
    else o->allocator->free(o->mem_ptr, o->size, o->allocation);   // o->allocator can be null

jittor's own crash handler caught it, with `addr2line` resolving the fault PC to
`jittor::ArrayOp::run()` and the fault address 0x0. Reproduced in seconds by
`probe_loader_migrate.py`, which runs the loader's real pattern -- a host tensor,
`narrow` of it, `copy_` into a device parameter -- and it reproduced **on one
thread** (twice), so this was never a threading race on its own. On a TP2 rank the
same null storage is what a later copy reads, which is the device-1 illegal address.

Fixed by skipping the free when there is no allocator (there is nothing to free).
`tests/core/test_array.py::TestArrayOpDoesNotAssumeAnOutputAllocator` covers the
`_force_fuse` path. **Verified:** the probe's single-threaded phase segfaulted
before and passes after, and a TP2 run then loaded *both* ranks with zero illegal
addresses -- previously roughly two starts in three died in this phase.

**Still there, and separate: the threaded case.** With the null allocator fixed, the
probe's four-thread phase fails with

    fused_op.cc:89: [check failed: outputs().size()]     op: fused_op(contiguous)

i.e. a **thread-safety** problem in jittor's op-building/fusion layer: several
loader threads build and fuse ops while the main thread applies weights. jittor has
the right mechanism already -- `ExecutorEntryScope` in `src/runtime/executor_entry.h`,
a per-thread recursive lock that drops the GIL before blocking on the lock -- so the
next step is to find which op-construction path the shim takes that does not enter
it (`jtorch_aten.cu`'s `make_empty_vh`/`make_copy_vh` call the op constructors
directly, from whatever thread the caller is on).

**One hypothesis measured away.** The index inputs of the first DiT block are *not*
host-resident, on either rank:

    rank 0: inverse_indices int64[3072] dev=0 loc=device   token_tags int64[3072] dev=0 loc=device
    rank 1: inverse_indices int64[3072] dev=1 loc=device   token_tags int64[3072] dev=1 loc=device

(`H3_IDX_TRACE=1`, metadata only, so a poisoned context cannot hide the answer.)
They are identical and device-resident, so the "garbage/stale index buffer" reading
is withdrawn for good -- the request-path fault is elsewhere in that block, after
the all-gather whose arguments section 21 already verified.

## 23. The threaded op-construction race is confirmed, and it is around `contiguous`

Section 22 ended with the four-thread probe failing at
`fused_op.cc:89: [check failed: outputs().size()]`. That is now pinned:

    probe_loader_migrate.py, 4 threads, no serialisation
      -> 4 threads FAILED: fused_op.cc:89: [check failed: outputs().size()]
    probe_loader_migrate.py, 4 threads, SERIALISE=1 (an RLock around each worker)
      -> 4 threads OK

So concurrent op construction from several Python threads corrupts jittor's fused
batch -- the graph is single-threaded by design and the shim is the layer that lets
a multi-threaded host library (vLLM's four-thread safetensors loader) drive it.
`fused_op.cc` builds the fused op's outputs from `var_stays_in_memory(...)`, a
verdict that comes from executor/batch state, so another thread's construction
inside that window leaves the batch with no outputs at all.

**And the request-path fault is bracketed by the same op.** In the run right after
the `ArrayOp` fix (loading clean, request reaching the denoise loop) the last
launches before the device-1 fault are

    cublas_matmul   vllm/model_executor/layers/utils.py:98      (a linear)
    getitem x2      vllm/distributed/utils.py:119               (torch.split)
    reshape         group_coordinator.py:244

i.e. vLLM's TP `split_tensor_along_last_dim` -- `torch.split(...)` then
`tuple(chunk.contiguous() ...)` -- feeding a linear. `contiguous` is exactly the op
the four-thread probe corrupts, so the leading explanation for the remaining
request-path fault is the same race, surfacing as a null/invalid storage that the
next cublas read dereferences instead of as the clean `outputs().size()` check.

**Next step, then:** serialise op construction at the shim's boundary. jittor
already has the per-thread recursive lock that does this for executor entry
(`ExecutorEntryScope`, `src/runtime/executor_entry.h`), but it is C++-only, so the
shim needs its own lock on the entry points a weight load goes through --
`torch.tensor`/the factories and `Tensor.copy_`/`narrow`/indexing -- and then the
same probe plus a TP2 request to verify. Section 22's measurement stands: this is
not the index buffers, whose metadata is identical on both ranks.

## 24. The request phase is single-threaded, so the race is *not* its cause

Section 23 ended by proposing that the request-path fault was the same
op-construction race, on the strength of `contiguous` appearing in both. That is
now measured away, and the measurement is cheap enough to have been done first:
`py-spy dump` on the rank-1 worker across the request window shows **three
threads** --

    Thread 575656 (idle): "MainThread"
    Thread 583468 (idle): "Thread-1 (_accept_loop)"      (the store server)
    Thread 583469 (idle): "Thread-2 (_serve_connection)"  (a store connection)

-- and no thread but the main one ever builds or runs ops. With one thread in the
graph there is nothing to serialise, so a shim-level lock would have been a
change that fixed a real defect (section 23 stands) while doing nothing for this
fault. Withdrawn: the request-path fault is not the concurrent-construction race.

What is left, all of it measured rather than assumed: the fault is a device-1 MMU
fault in the first DiT block, on one thread, in the sequence
`torch.split` -> `contiguous` -> linear; the all-gather's arguments are correct
(`ynum == group_size * num`, device-1 operands on device 1's stream); the index
metadata is identical on both ranks; every attention entry binds the input device;
and the shim's own documented racy-illegal-address window (the second jittor
barrier is skipped when `JITTOR_TORCH_SHIM=1`) does not apply, because no triton
kernel has launched yet when the context dies -- the `[trishape]` trace prints
exactly once per rank, at the first denoise kernel, and that kernel already
reports the poisoned context.

## 25. `transpose`'s strides are *not* a lie, and "fixing" them breaks it

The next hypothesis was that the shim misreports strides for views -- `x.transpose(0,1)`
on a `(2,4,16)` tensor reports `stride=(32,16,1)`, `is_contiguous() == True`, where a
lazy view would have `(16,64,1)` -- and that a consumer reading `.stride()` (a cublas
leading dimension, an extension laying out a kernel) would walk out of bounds. The
reasoning was wrong, and two cheap measurements show it:

    x ptr=0x7ef989624200   t ptr=0x7ef989624400   same=False
    t=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]        # a (2,3) tensor transposed: correct
    t stride=(2, 1) contig=True                   # correct for that materialized copy

CUDA's transpose is **eager**: `cutt_transpose` (registered as `OpCapability::Transpose`)
computes a fresh row-major buffer, so the output's contiguous strides are exactly right.
No lying strides, no out-of-bounds read from this, hypothesis withdrawn.

**And the attempted fix was reverted, because it broke the op.** Adding
`y->set_storage_strides(<permuted strides>)` to `TransposeOp::infer_shape` (mirroring
`reshape`/`getitem`) made the op declare a *transposed* layout for a buffer the
execution path had written *contiguously*, and the values came back wrong:

    before fix: t=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    with fix:   t=[[1.0, 5.0], [4.0, 3.0], [2.0, 6.0]]

It is reverted in both trees and the core rebuilt; the roundtrip is exact again. The
generalisable lesson, since this is easy to repeat: `set_storage_strides` describes how
the *producer writes the bytes*, so it may only be used by an op whose execution really
leaves the data strided. A materializing op that declares strided metadata hands every
later consumer a wrong layout.

## 26. The device jittor *reports* was not the device the thread ran on

Sections 19-25 chased the same symptom from several directions -- rank 1's device-1
CUDA context poisons itself during the four-thread shard load, and every later call on
it fails -- and each time the evidence pointed at whatever call happened to notice the
context was gone. The actual cause was one process-wide cache standing in for state the
CUDA runtime keeps per host thread.

`accelerator_current()` (`backends/cuda/runtime/driver.cc`) returned
`runtime_device_state().current_device`, a single `int` behind the process-global
`runtime_device_state()` (`src/runtime/device_state.h`). **CUDA's** current device is
per-host-thread and starts at 0 on every new one. So a thread-pool worker -- every
`multi_thread_safetensors_weights_iterator` worker -- sat on device 0 while jittor
reported device 1:

    # probe_thread_device.py, before the fix
    worker0  jt.current_device()=1  jt.flags.device_id=1  cudaGetDevice=0
    worker1  jt.current_device()=1  jt.flags.device_id=1  cudaGetDevice=0

That alone would only mislabel things. What made it fatal is `on_device()`:

    int previous = accelerator_current();          // process value: 1
    if (device != previous) accelerator_set(device);   // 1 != 1 -> skipped
    auto result = func();                          // runs on THIS thread's device: 0

`previous` is the process value, so on a fresh thread `device != previous` is false and
**`cudaSetDevice` never happens**. `cudaMalloc`, `cudaMemcpy`, `cudaMemGetInfo`, event
recording and the cuBLAS/cuDNN handle selection then ran against that thread's device-0
context while jittor booked them against device 1 -- a cross-device mismatch. Its
signature is exactly what was observed: an Xid 31 MMU fault, and a context-sticky
`cudaErrorIllegalAddress` that surfaces later on a call that only *reports* the context
is gone, which is why `cudaMemGetInfo` inside `jt.array` was the call holding the error.
It is invariant for rank 1 and can never happen on rank 0, whose worker threads default
to the device it uses anyway.

The file already knew: `record_event` (`driver.cc:326-339`) documents this precise
hazard -- "a thread whose CUDA context is still the process default" -- and works around
it with its own `cudaSetDevice`. The fix moves that to the one place that can state the
invariant: `accelerator_current()` binds the calling thread to the device it is about to
report (a thread-local cache, one `cudaSetDevice` plus the switch hooks on first use per
thread), and `accelerator_set` records the binding. Every call site that resolves a
device through jittor is then consistent with the thread's own context.

    # after the fix
    worker0  jt.current_device()=1  cudaGetDevice=1
    worker1  jt.current_device()=1  cudaGetDevice=1

Two things this does not cover. The deployed tree is an older, self-consistent snapshot
whose `BackendOps` has no `graph_capture_*` members, so the lab build is the repo file
with that surface removed -- copying the repo's `driver.cc` in whole does not compile
there (`struct jittor::BackendOps has no member named graph_capture_begin`), which is
also recorded in the lab notes. And ROCm has the identical pattern
(`backends/rocm/runtime/driver.cc` caches `hipGetDevice` in the same process-global);
there is no ROCm hardware here to verify a fix on, so it is deliberately left alone.

## 27. Two more device-boundary defects, and the request fault is flash-attn's

With the loading illegal address gone (section 26) the TP2 request still dies on
rank 1, and the evidence now says the remaining fault is **not** the one the
launch lists kept pointing at. Three negative results, each from a run:

* the bridge's trailing `jt.sync_all(True)` is not it. `_fast_sync_enabled()`
  returns True whenever `JITTOR_TORCH_SHIM` is set -- which `env-jittor.sh` does
  -- so the barrier that `compat/triton/backend.py` says exists to close "a racy
  `cudaErrorIllegalAddress`" is skipped in every lab run. Setting
  `JITTOR_TRITON_FAST_SYNC=0` to re-enable it does not change the outcome.
* the stream the bridge launches on is not it either, though it was wrong. The
  bridge handed `cuLaunchKernel` a NULL stream -- the legacy default stream --
  while jittor puts everything on `cudaStreamPerThread`, and `driver.cc` says
  those two "do NOT synchronise with each other ... an unordered race that
  raises no error and produces no message". Fixed (`_launch_stream()`, with
  `JITTOR_TRITON_LEGACY_STREAM=1` as the escape hatch, and a test that
  intercepts `cuLaunchKernel` and asserts the argument); the fault survives it.
* it is not an async-attribution problem: `CUDA_LAUNCH_BLOCKING=1` reproduces it
  at the same place, so the fault is deterministic and not a kernel-vs-kernel
  race.

What does discriminate is the attention backend. The same request with
`ATTN=TORCH_SDPA` runs the two denoise steps to completion (`100%| 1/1`) with
zero illegal addresses, while every `ATTN=FLASH_ATTN` run dies before finishing
the first step -- and the logs confirm the runs really did resolve different
backends ('SDPA' vs 'FLASH_ATTN' for `role='self'`). So the request-phase
device-1 fault lives in the flash-attn path, not in the shared pre-attention
work. Section 20's device guards are all still in place (`{q.device()}` /
`{qkv.device()}` in all six generated entries), so it is somewhere else in that
extension.

Section 21 was right that the index read is a victim and wrong about where to
look: `combined_indices = inverse_indices * 3 + token_tags.clamp(min=0)` is in
bounds *by construction* -- `adaln_proj` maps `[M, t_dim]` to `[M*modality_num, H]`
(`minimax_h3_transformer.py:743`), so an index in `[0, M*3)` cannot walk off it,
and the `[3, 5376]` table in the launch trace just means M = 1 on that rank.

**A third defect, found while checking that arithmetic.** `jt.unique` built its
prepended element with `jt.concat([Var([False]), diff], 0)`, and `Var([False])`
is created on the **ambient** device, so any input on another device made that
concat raise

    dispatch_context.cc:52: Expected all tensor inputs on the same backend and
    device ..., first input backend=1 index=0 but another input has backend=1
    index=1

`probe_unique_scatter.py` reproduces it in one process (device 1, ambient 0:
raises; ambient 1: matches numpy), and it is exactly what the denoise loop asks
for -- `torch.unique(timesteps, sorted=True, return_inverse=True)` at
`denoise_loop.py:207`, whose inverse becomes the AdaLN row index. It is the same
defect family as the `concat` placement fix, and the same asymmetry: a
multi-process run only hits it on the ranks whose device is not the process
default. The element is now built out of `input_sorted`, so it carries that
tensor's device, and `TestUniqueOffTheAmbientDevice` pins it.

**A fourth, same family, and this one is on the denoise path.** `ones_like`,
`zeros_like`, `full_like`, `rand_like`, `randn_like`, `randint_like` and the
`x.new_*` methods all built their result from the *ambient* device -- `ones_like`
is `ones(x.shape, x.dtype)`, and `zeros_like`/`full_like` are the same shape --
while torch's contract for the whole family preserves the reference tensor's
**device** as well as its shape and dtype:

    probe_ambient_ops.py:  idx.device_id=1  ->  jt.ones_like(idx).device_id=0

`jt.current_device()` is the ambient one and `to_device` does not move it (it
belongs to the caller, and `run_sync` restores it), so on a rank whose device is
not the process default every `*_like` in a kernel's neighbourhood produced a
device-0 tensor beside device-1 ones. `dispatch_context` rejects that for
jittor's own ops; the flash-attn extension does not check, and there it is a
launch on one device with another device's pointers. Fixed with a
`device_scope_like` context manager in `_core/var.py` (covering both an explicit
placement and a `.to_device(n)` tensor, whose `placement_backend` stays -1), and
`TestLikeConstructorsKeepTheDevice` pins the whole family.

The same asymmetry has a core half: `device_raw_ptr` / `device_ptr_ready`
(`src/core/var_holder.h`) migrate a host-resident Var with `get_allocator()`,
whose no-device overload is `current_device()` (`allocator.cc:105`), i.e. the
ambient device -- so the operand lands on one device and the caller hands that
pointer to a kernel launched on another. They now migrate to the Var's own
device instead.

## 28. What the `fused_op.cc:89` assert actually is, and where the flash-attn path really runs

**The assert is a recycled node, not a fusion verdict.** Section 27's diagnostic
(`H3_FUSE_DUMP=1`, a dump in `FusedOp::update_ops` just before the assert) fires as

    fused segment with no in-memory output: ops=1 batch_stamp_wanted=67
      batch_var_fused=set stamp_count=68 active_epochs=1
      op unary tflag=68 batch_stamp=67 outputs=0

which rules the classification out on both counts: the segment holds **one** op,
that op has **no outputs at all** (`outputs=0` -- its Var was destroyed), and its
`tflag` is a **newer** stamp than the batch's (`68` vs `67`, `78` vs `71`), which
is what a freed-and-reused node slot looks like. The batch holds raw `Op*`, so the
executor is walking an object that no longer exists. That is why 1-2 threads pass
where 4 fail and why `SERIALISE=1` passes (`probe_loader_migrate.py`).

Serialising the shim's *Python* entry points does not remove it: with
`JITTOR_SHIM_SERIALISE=1` (a lock around `torch.tensor`, `Tensor.copy_`,
`Tensor.narrow`, `Tensor.__setitem__`) the probe turns into a segfault in
`jit_utils_core.so` instead. That lock is also unsound as written -- it blocks on
an `RLock` while holding the GIL, in paths that release the GIL inside jittor,
which is the inversion `ExecutorEntryScope` documents -- so it is not a fix and
should not be kept. The sharing that matters is inside core jittor, where the
executor releases the GIL during device waits and a second thread can be inside
the same graph machinery: the exclusion has to be process-wide and GIL-aware, like
`ExecutorEntryScope`, applied to the loader's core-jittor entry points.

**`flash_attn` does not come from the shim's stub.** The shim *deploys* its stub
to a top-level `site-packages/flash_attn/`, and that installed copy is what
imports - so editing
`jittor/compat/shim/resources/stubs/flash_attn/__init__.py` changes nothing at
run time (the two files were byte-identical apart from the edit). Instrument
`site-packages/flash_attn/__init__.py`, and keep the stub in step so a redeploy
does not drop the instrument.

The jittor flash-attn *extension* is not installed either
(`import flash_attn_jittor_cuda failed: No module named 'flash_attn_jittor_cuda'`),
so `ATTN=FLASH_ATTN` runs this math fallback, not the extension. The shim's
`c10/cuda/CUDAStream.h` was still wrong -- it handed torch extensions the **legacy
default stream** while jittor runs on `cudaStreamPerThread`, the same defect as the
triton bridge in section 27 -- but that is not this fault's cause, because nothing
reaches the extension here.

**Correction to the paragraph above, and the reason the stream fix did nothing.**
`serve-vllmomni.sh` exports `JITTOR_FLASH_ATTN_JITTOR_SRC=/root/jittor-lab/flash-attention`
and `JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1`, so the real run *does* build and load
the official extension: it lands in
`$XDG_CACHE_HOME/jittor/torch-shim/<tag>/torch_extensions/flashattn_jittor/official_flash_attn*/<digest>/flash_attn_2_cuda_jittor.cpython-312-x86_64-linux-gnu.so`
(several digests, newest 8.7 MB), and `adapter.py`'s trace line sits immediately
before `packed_low_level.fwd(q, k, v, scale, causal, wl, wr)` -- the extension,
not the math fallback. `flash_attn.flashattn_jittor_backend()` says "math" because
it looks for a *different* module name (`flash_attn_jittor_cuda`), so that string
is not evidence about this path.

The build identity, in `official_build.py`, is

    <source path> | <git HEAD> | head_dims | dtypes | native_forward_backward_dropout=1
    <source path> | <git HEAD> | head_dims | dtypes | direct_packed_forward=6

-- it covers what gets *compiled* but not the headers the extension compiles
**against**. `c10/cuda/CUDAStream.h` is one of those, so fixing the stream left
every existing build in place and the stale `.so` kept launching flash-attn's
kernels on the legacy stream, unordered with jittor's allocator. Both digests now
include `|shim_hdrs=<sha256 of cpp_extension/include>`, which both states the
missing input and forces a rebuild that picks the fix up.


## 29. The `fused_op.cc` assert is a batch node freed under the planner

Section 28 located the assert as "a segment whose single op has no outputs and a
`tflag` newer than the batch's". One half of that was a red herring and the other
half was the clue:

* `tflag` is *supposed* to differ. `load_fused_op` opens a `TraversalEpoch` for
  itself and `mark()`s every op it loads, so the segment's ops carry the fused
  epoch's stamp rather than the batch's. That is the loader working as designed.
* `outputs == 0` was the real anomaly, and the dump was extended to attribute it.

### What the dump says once it prints the op's address and its holder

```
op unary addr=0x7f6728004d50 tflag=66 batch_stamp=65 outputs=0 holder=1 inputs=0
```

`holder=1` is the op's own `vector<VarPtr> outputs_holder` (`core/op.h`,
filled by `Op::create_output`): its output var is **alive and still owned by the
op**. So no var was destroyed and no slot was recycled. `inputs=0` next to it is
the tell -- these are empty *edge lists*, and `Node::free()` clears them
**directly**:

    _inputs.clear();            // node.cc, Node::free
    ...
    _outputs.clear();           // node.cc, Node::free

It does not route its own edges through `Node::erase_output`. That is what the
first instrument did: an `erase_output` hook (the path a *var* takes to detach
itself from its producer) fired **zero times** in the runs that reproduced the
assert, while the same runs printed the dump. The reading that fits every field
is that `Node::free()` ran on the `unary.cast` op itself, mid-batch: edges
cleared, `_queued_for_free` set, deletion still pending in `free_buffer()`, so
the object stayed readable and `update_ops()` -- which classifies a segment by
walking `op->outputs()` -- saw zero of them.

Who calls it: `Op::free()` is reached from `release_backward_liveness` when an
op's counters hit zero, i.e. from whichever thread drops the last reference to
the cast's result. Why nothing stops it: the batch holds raw `Op*`/`Var*` in
`plan.ops` and takes no liveness on them, and `Node::free()`'s only lifetime
guard is

    if (is_var() && _inputs.size() && (liveness.forward.active() || !is_finished())) return;

which is about **vars** and has no counterpart for an **op**. `TraversalEpoch`'s
own comment states the contract -- "Traversals are synchronous and stack-nested.
They must not destroy a node they marked before their epoch ends" -- and the
planner and a freeing thread only have to overlap for the loader's four threads
to reach it. The first reading of this section (a process-global `free_buffer()`
handing one thread's nodes to another's delete round) is **withdrawn**: it
predicts a `tflag` mismatch, which is not evidence, and it does not explain a
live `outputs_holder`.

### The early return is not the fix -- measured, not argued

An op is reachable only through the vars it produces, so "an op with a live or
unfinished output is not garbage" looks like the missing dual of the var guard.
It was tried, and it does suppress the assert: **12 loader-race runs, 0 dumps**
(against 2-3 dumps in 7 runs of the instrumented-only core). It also replaces
the assert with a different failure in 3 of those 12:

```
node.h:279: backward liveness release without a matching owner [check failed: value_ > 0]
```

`free()` is part of the liveness protocol, not just a destructor: it is where a
node releases the liveness it holds on its inputs. Returning early skips those
releases and the counters underflow later. So the node must **not** be kept
alive; the planner and the freeing thread must not **overlap**. That is
`graph_mutation_mutex()`, which `Node::free()` already takes, and which the
executor's planning path never takes -- the guard is one-sided.

### The one-sided guard, enforced: also measured, also not the fix

That made the next candidate obvious -- hold `graph_mutation_mutex()` for the
batch, in `Executor::run_sync`, so no other thread can clear a batch node's edges
while the planner reads them. It is five lines (`free()` takes the same recursive
lock, so the batch's own frees still pass through). Measured:

```
#### ex run 1
exit=124          # timeout 600
```

Both of the first two runs had to be killed; neither reached a verdict. The
exclusion **deadlocks** the four-thread loader. The reason is a lock-order
inversion, not the device wait: `run_sync` already holds `ExecutorEntryScope`
(call it L1) and would now hold `graph_mutation_mutex()` (L2) inside it, while a
second thread sitting in `Node::free()` holds L2 and can need L1 -- that drain can
re-enter the executor. Reverted in both trees.

So both cheap shapes are now excluded by measurement rather than by argument:

| candidate | result |
| --- | --- |
| keep the node alive (early return in `free()`) | assert gone, 3/12 runs fail `node.h:279` counter underflow |
| hold the graph-mutation lock for the whole batch | loader deadlocks (`exit=124`) |

What is left is to separate the two halves of `free()`: the **accounting** (the
`release_*_liveness` it enqueues on its neighbours) has to happen immediately and
exactly once, while the **graph surgery** (`erase_output` on its producers,
`erase_input` on its consumers, `_inputs.clear()`, `_outputs.clear()`, the
recursive free of its output vars, `free_var`) is what the plan can trip over and
is therefore what has to wait until the batch is done. A deferred-surgery list
drained at the end of the batch preserves the counters, which is why it is the
remaining design -- and the reason it is written down rather than landed here is
that `free()` is re-entered from several liveness paths and the erasures are
index-based (`back_index`), so the drain has to erase by identity and run exactly
once per node.

### The design that avoids both problems: snapshot the edges in the plan

Both rejected candidates are trying to keep the *live* graph alive for the
planner. But the planner is the *only* reader that needs those edges, and it
already walks them once, during collection. `build_exec_plan` BFSes from the
batch's roots over `_inputs` and numbers every node it reaches in
`Node::batch_index`; if it also recorded, per collected op, the input and output
var pointers it saw, then `load_fused_op` and `FusedOp::update_ops` would read
that snapshot instead of `op->_inputs` / `op->outputs()`.

That is the same move this code already made once, for the same reason: the
comment above `set_batch_index` says the numbering used to live in
`Node::custom_data`, "so a traversal starting while these were live renumbered
the graph under the executor" -- the fix was to give the batch its own
immutable-per-batch record instead of reading shared state. The edges are the
last thing the planner still reads live. A snapshot would make the batch
independent of any concurrent `free()`, with no lock and no deferral, and it
removes the whole class rather than one instance.

It is a larger change than the two that were measured -- `load_fused_op`'s
`edges` construction and the codegen numbering are delicate, and the vars'
producers (`v->_inputs.front()`) have to be captured too -- which is why the two
five-line candidates were tried first.

### Why the cheap family is exhausted, measured four ways

Two more shapes were built and measured before concluding that, and the reason
they fail is one property of `free()` that no amount of deferral can work around:

| candidate | 20 runs |
| --- | --- |
| deferred surgery, accounting (incl. the recursive output-var free) left out | 12 ok, 8 fail, 0 dumps, **8 underflows** |
| deferred surgery, accounting kept, including the recursion | 7 ok, 10 fail, 0 dumps, **10 underflows** |
| hold one *pending* liveness per planned node, released after execution | never built: it cannot work (below) |

`Node::free()`'s surgery is what *removes the node from its neighbours' reach*.
A node that keeps its edges keeps being the target of further counter releases --
`release_forward_liveness`/`release_backward_liveness` propagate along exactly
those edges -- so every scheme that keeps the node alive ends in
`node.h:279: backward liveness release without a matching owner`. Splitting
`free()` cannot be made sound by choosing which half to defer, because the two
halves are the same choice seen from two sides.

And the pin does not work either, for a reason worth writing down: **nothing
gates `free()` on a counter.** `release_backward_liveness` calls it
unconditionally the moment its counter reaches zero (`node.cc`, "Free
backward_liveness=0" then `free();`), so holding one liveness does not stop the
next release from reaching zero and firing it. `NodeLiveness::need_free()` is
consulted by callers, not by `free()`. To pin a node you would have to floor its
counters inside the release handlers while a batch is live -- i.e. add exactly
the state the snapshot design makes unnecessary.

So the two halves of the conclusion are: the *plan* must stop reading live edges
(snapshot), or the *release handlers* must learn about a pinned batch (floor).
The first is smaller and matches what this code did for `custom_data`.

Why "floor" is not the ten-line alternative it sounds like, so nobody starts it
blind:

* it needs a new shared `NodeFlags` bit, and that region carries an explicit
  warning that its layout is derived from two `_end`s precisely because a hand
  picked bit had already broken it once;
* it needs the floor in all three release handlers *and* on the direct
  `out.node->free()` recursion inside `free()`, which is the one free call that
  does not go through a release handler;
* and making `need_free()` false for a pinned var is observable:
  `exec_runner.cc` uses `!var->need_free()` to decide whether an output var goes
  into `outputs_bk` (kept) or is merely marked finished, so flooring changes what
  a batch retains.

A snapshot of the *outputs* alone is not enough either, which is worth writing
down: `update_ops` reads `op->outputs()`, but `load_fused_op`'s `edges` loop reads
`op->_inputs` and `v->_inputs.front()` as well. Snapshot the first and not the
rest and a freed op yields *incomplete* edges -- wrong kernels rather than an
assert, which is worse. It has to be the whole tuple.

### Resolved: the batch has to hold what it uses

The design above was built, and it is not sufficient on its own -- but building it
is what found the answer. Snapshot only, 20 runs: **0 dumps, 0 underflows, and 7
segfaults**, and the backtrace names the reader:

    FusedOp::execute_fused_prepared -> Op::execute_prepared -> run_exec_plan
      -> VarRelayManager::get_op_relay_info   (var_relay.cc)

i.e. a *codegen* walk of the graph, at execution time, over a node that had been
destroyed. So the plan does not merely need the *edges*; it needs the **nodes**,
and no amount of snapshotting or deferral changes that.

What the batch was missing is ownership, and the mechanism was already in the
tree: `VarPtr`. `Executor::run_sync` now holds a `VarPtr` for every var in
`plan.all_vars` for the batch's duration. That pins the vars, and it pins the ops
too -- an op's liveness comes from its outputs, so an op whose output var is held
cannot be freed either. It touches no counter arithmetic and takes no lock, which
is why it does not have either failure mode of the candidates above.

Measured, `H3_FUSE_DUMP=1 probe_loader_race.py threads 4 6`, 20 runs each:

| variant | result |
| --- | --- |
| no hold (baseline) | 4 ok, 1 failed, **2 dumps** in 6 runs |
| keep the node alive (early return from `free()`) | 12 ok, **3 underflow** |
| hold the graph mutation lock for the batch | **deadlock** (`exit=124`) |
| lock released around the compile phase | still **deadlock** (9/12): this lab runs with `use_parallel_op_compiler=0`, so that phase never happens and the window never opens |
| edge snapshot only | 13 ok, **7 segfault** |
| edge snapshot + this hold | **20 ok, 0 dumps, 0 segfaults, 0 underflows, 0 hangs** |

The probe's own checksum is unchanged (`sum=-26415.4` in every run), so the fix
does not alter the numbers. The edge snapshot stays in the change as well: a
*live* var can still have its edges released underneath the planner
(`release_inputs`, which the shim uses to park tensors), so the plan should not
read them live even with the vars pinned.

Landed in `src/core/{exec_plan.h,exec_plan.cc,fused_op.h,fused_op.cc,executor.cc}`;
the four `exec_plan`/`fused_op` files had no drift between the repo and the lab's
deployed snapshot, so they were copied, and `executor.cc` (43 lines of drift) got
the same three hunks by hand. Verified by building the repo's own core
(`REPO_CORE_OK`) and by 20 loader-race runs on the deployed build.

The probe for any candidate is
`H3_FUSE_DUMP=1 probe_loader_race.py threads 4 6` -- 15-20 runs, because the rate
is roughly one in four -- and the per-run timeout should be short (~120 s), since
a wrong candidate hangs rather than fails. Count `dumps` **and**
`without a matching owner`: the assertion is easy to trade for the underflow.

Baseline on the core as it stands (stream fixes + the `H3_FUSE_DUMP`
diagnostics): **6 runs, 4 `OK`, 1 failed, 2 dumps** -- so the assert is still
there at about one run in three and the diagnostics are in the binary. Any
candidate should be read against that, not against zero.

## 30. The deployed core was missing the whole stream-consistency set

The FLASH_ATTN request was the first run with the rebuilt extension, and it
failed with a *different* error than every previous attempt:

```
exec_runner.cc:402: Execute fused operator(11/102) failed.
setitem_op.cc:295  code=1( cudaErrorInvalidValue )
  cudaMemcpyAsync(op, ip, out->size, cudaMemcpyDeviceToDevice, 0)
```

That trailing `0` is a stream, and it is the legacy default stream. The launch
candidates around it are all `stream=2` -- jittor's own `cudaStreamPerThread` --
including the `setitem` at `encoder.py:259` that is the failing op. The two
streams do not synchronise (see `compute_stream` in `backends/cuda/runtime/driver.cc`),
so the copy races whatever produced `ip`.

The macro is real and the divergence is one line:

    deployed  indexing_backend_copy() ... cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, 0)
    repo      indexing_backend_copy() ... cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, cudaStreamPerThread)

Auditing the whole tree rather than that one site turned up the full set, all of
it from one repo commit (`28e8e1a7`, the CUDA-Graph work) that the deployed
snapshot predates:

| deployed snapshot | repo |
| --- | --- |
| `kernels/core/setitem_prefix.cc` -- stream `0` | `cudaStreamPerThread` |
| `kernels/cutt/cutt_transpose_op.cc` -- stream `0` | `cudaStreamPerThread` |
| `kernels/curand/curand_random_op.cc` -- stream `0` | `cudaStreamPerThread` |
| `cublas_wrapper.cc` -- handle left on the default | `cublasSetStream(..., cudaStreamPerThread)` |
| `cudnn_wrapper.cc` -- handle left on the default | `cudnnSetStream(..., cudaStreamPerThread)` |
| `cusparse_wrapper.cc` -- handle left on the default | `cusparseSetStream(..., cudaStreamPerThread)` |
| `curand_wrapper.cc` -- generator left on the default | `curandSetStream(..., cudaStreamPerThread)` |

In the deployed tree `cudaStreamPerThread` appeared in exactly one file --
`runtime/driver.cc` -- and that one is this session's own device-binding fix.
Every cuBLAS/cuDNN/cuSparse/cuRAND call the model makes was therefore issued on
a stream that does not order against the kernels feeding it. That is a
process-wide source of intermittent wrong results and illegal addresses, and it
is a better candidate than anything else on the table for the remaining
`ATTN=FLASH_ATTN` faults -- the failures were always intermittent, and the
`SDPA` profile that passed is the same code with different timing.

### The library handles are only half of it: the kernels themselves

The same commit also splits the fix in two, and its own comment in
`driver.cc` says so:

    //   - jittor's own kernels: `--default-stream per-thread` in the nvcc flags,

The deployed `runtime/driver.cc` carries that comment (it has the whole
CUDA-Graph `compute_stream`/capture machinery) while the deployed
`build/compiler.py` **does not carry either flag**. The repo has both:

    cc_flags += " -D__CUDA_API_PER_THREAD_DEFAULT_STREAM=1 "      # host side
    nvcc_flags += " --default-stream per-thread "                 # device side

So the deployed tree is a *partial* copy of that commit: `compute_stream()` says
`cudaStreamPerThread`, the library handles were left on the legacy stream, and --
the part that matters most -- every kernel the tree compiled was compiled with
the **legacy** default stream. This is not indirect: jittor's generated kernels
launch with a bare `<<<grid, block>>>` and no stream argument. From the run's own
jit cache, `..._hash_df1ef54aa29ad859_op.cc`:

    kernel<<<1,1>>>(op0_outputp, op0_outputv);

With `--default-stream` unset that is the legacy default stream, not
`cudaStreamPerThread`. The runtime then reasons about -- and records in
`[Recent launch candidates]` -- a stream its kernels are not on, which is exactly
why those traces show `stream=2` for ops that are in fact racing on stream 0.
The macro (`__CUDA_API_PER_THREAD_DEFAULT_STREAM`) is the same fix for the
runtime-API calls that take no stream; the flag is the one that moves the
generated kernels.

Because it changes how *every* translation unit is compiled, the two flags only
take effect after a core rebuild **and** a jit-cache clear -- the jit key does not
cover the compile flags either, so a warm cache keeps the old kernels (see the
`*_prefix.cc` trap below).

Both halves are checkable at the artifact level, because jittor records the full
command line it ran next to every output: `.../<hash>.o.key` for a core object,
`<jit-key>.so.key` for a jit kernel. After the rebuild the recorded command for
`backends/nan_checker.cu` contains

    -D__CUDA_API_PER_THREAD_DEFAULT_STREAM=1
    --default-stream per-thread

and the `jit.pre-streamfix/…_op.so.key` from before the change contains neither.
That is the same kind of evidence as the `strings` check on the flash-attn
extension, and it is why the flags were applied before the run rather than
inferred from it.

Two traps in applying it:

* `backends/cuda/kernels/cublas/lt_linear_cuda.py` also carries the fix, but the
  deployed snapshot has no such file (it predates the file's introduction). It
  was **not** copied: adding one codegen module to an older tree that never had
  that path is not a stream fix.
* The two changed `*_prefix.cc` files are codegen prefixes, and section 17
  already established that the prefix/include is **not** part of an op's jit key.
  A warm kernel cache would therefore keep serving kernels built with the old
  prefix. The run's `jit/` cache was moved aside so every affected kernel
  recompiles.

### What the stream fix changed, and what is left

Three `ATTN=FLASH_ATTN` TP2 requests, 2 steps at 256x256, one after the other:

| core | where it died | how long it ran |
| --- | --- | --- |
| rebuilt flash-attn extension, nothing else | `setitem` D2D copy in request-input prep (`encoder.py:259`), `cudaErrorInvalidValue` | 10.1 s |
| + the seven library/prefix sites | same place, same error | ~10 s |
| + the two per-thread default stream flags | rank-1 `cudaErrorIllegalAddress`, surfacing at the emulated modulation kernel's sync in the *denoise* loop | 475.3 s |

So the stream consistency is real and load-bearing -- the third run gets through
request preparation and essentially the whole two-step denoise, where the first
two never reached the sampler -- but it is not sufficient. What is left has the
same shape as everything that came before it: rank 1 only, device 1 only, and
asynchronous (the error surfaces at a sync, `overwrite`d counts in the tens of
thousands, and the recorded candidates are the last sixteen launches rather than
the faulting one).

The last two things rank 1 did before the sync that reported it are worth
recording, because they narrow it:

* an `all_gather` of 2688 elements on the communication **side** stream
  (`stream=0x1d01e200`), whose device-side output `y` is then reshaped and read
  by `getitem` on the compute stream;
* the emulated triton kernel of `rms_norm_indexed_scale_shift`, which is where the
  sticky error becomes visible -- the same op the pre-fix run died in.

The side-stream join itself is *not* the defect: `backend_default_stream_wait_side`
records on the side stream and waits on `BackendStreamKind::Compute`, which
resolves to `ops.compute_stream(device)` == `cudaStreamPerThread`, so the
ordering target is already the per-thread stream and the per-thread compile
agrees with it.

Two controls, both on that same rebuilt core, and together they say where the
residual is **not**:

* **`ATTN=TORCH_SDPA`, same request: completes, 620.4 s.** The model, the TP
  collectives, the modulation kernels, the scheduler and the VAE are therefore
  all sound in this binary -- and the residual is specific to the flash-attn
  path, exactly as section 27 concluded for the original fault.
* **`ATTN=FLASH_ATTN`, single GPU: completes, 121.3 s.** Same extension, same
  shim, same core, one rank. So the flash-attn path is sound in the one
  configuration where jittor's device *is* the process default device (0).

Read together with the failure being rank-1-only, that is a narrower statement
than "flash-attn is broken": the residual needs **both** tensor parallelism and
the extension -- i.e. the case where jittor's device is not the process default.
That is the same shape as sections 20, 21, 26 and 27, and it is why the next
instrument prints the attention *result*'s device and the ambient device, not
just the inputs': correct inputs on cuda:1 with a result buffer labelled cuda:0
would send the next jittor op to device 0 with device-1 pointers.

**Correction, later the same session: that pair of controls was under-powered and
the "needs the extension" half is wrong.** Every one of those runs is n=1, and the
fault is intermittent, so a single pass says very little. The counter-example
arrived when the same `TORCH_SDPA` request was re-run on the recovered core: it
failed after **465.3 s** with the *identical* rank-1 `cudaErrorIllegalAddress` at
`cudaMemGetInfo`, and the run is confirmed to be a real SDPA run (the log says
`Resolved diffusion attention backend 'SDPA' for role='self'` twice, and
`flashattn_jittor` never appears). The launch candidates at the failure are
`denoise_loop.py:189/190/207` (`copy`, `getitem`, `setitem`) rather than the
encoder, and the sticky error again surfaces inside the emulated modulation
kernel's sync.

So the residual is: a **late, intermittent, rank-1/device-1 illegal address in a
TP2 request, independent of the attention backend**. What it is *not* is
established: the request-preparation failure (sections 27, 30) *was* flash-attn
specific and is fixed; this one is downstream of attention. The next step is the
denoise loop's own op stream on device 1 -- the `copy`/`setitem`/`getitem` burst
around `denoise_loop.py:189-207`, which is where the candidates now point -- not
the attention path.

**A third thing it is not: the batch-node lifetime hole of section 29.** That fix
landed (see the end of section 29) and the same `TORCH_SDPA` request was re-run on
the fixed core: it fails after 265.2 s with the identical rank-1
`cudaErrorIllegalAddress`. So the plan-vs-`free()` race -- which is real, and is
now fixed for what it *does* cause: the "no in-memory output" assert and the
loader's segfaults -- is not what the server trips over.

Given `CUDA_LAUNCH_BLOCKING=1` already showed the faulting operation is not one of
jittor's kernel launches (jittor wraps those in `LaunchErrorScope`, and the error
still surfaced at a memory query), what remains are the operations that
`LAUNCH_BLOCKING` does not cover: `cudaMemcpyAsync` on the copy side stream and
NCCL's driver-API work. `compute-sanitizer` cannot separate them here (NCCL fails
under it, above), and single-run controls cannot either. That is the state bug 1
is left in.

Three more components were excluded after that paragraph was written, each by
measurement on the failing request:

* **NCCL's P2P transport**: `NCCL_P2P_DISABLE=1` fails identically (20.1 s); so
  does `NCCL_PROTO=Simple` (the low-latency protocols are the ones that write
  directly into a peer's buffer, so that was the sharpest NCCL knob).
* **jittor's copy path metadata**: an instrumented `copy_async` (gated on
  `JITTOR_COPY_CHECK`) compares each copy's declared `Device` pair against what
  `cudaPointerGetAttributes` says about the two pointers -- residency *and*
  device index -- and shouts on a disagreement. Across the whole failing request
  it never fired. So `copy_kind` is not choosing a wrong direction and `copy()`
  is not picking a wrong device or peer mapping; if the fault is in the copy
  engine, the pointers really were as declared and the memory behind them had
  been recycled.
* **a disagreement between the ranks about a collective's size**: `ncclAllGather`
  takes the count from `x->num` on each rank, so a mismatch would make one rank
  mis-transfer; the instrumented trace of a failing run has every `num=` value an
  even number of times (113 collective lines, 2 per line counted once), i.e. both
  ranks called the same sequence of counts.
* **the allocator's managed-memory fallback**: `raw_malloc` logs `Unable to alloc
  cuda device memory for size ... falling back to cudaMallocManaged` before it
  takes that path, and that line appears in none of the failing runs -- nor in the
  passing ones. `cudaMemGetInfo`, the call the error is always reported at, is
  `memory_info(device, ...)`/`raw_memory_info`, so it is whoever asks for free
  memory next (the allocator, or the shim's `mem_get_info` via vLLM), not a
  component of the fault.

What that leaves is a shape rather than a component: **something recycled a
buffer that an operation still had in flight**, which needs only the ordering
guarantees to have failed -- and the condition that separates the failing server
from every control that passes is **memory pressure**. Each TP2 rank holds ~66
GiB with the probe's `OFFLOAD=` empty, on a card where another tenant's usage
moves; the single-GPU run that passes uses the shipped default offload, and so
does the real serve command. `run_tp2_probe.sh` is what sets `OFFLOAD=` empty --
that is a property of the probe, not of the configuration the lab ships.

### The shipped config fails *differently*, in 30 s, and jittor reports it

Running the same request with the shipped default (layerwise offload of
`text_encoder`, i.e. what `serve-vllmomni.sh` does when `OFFLOAD` is not
overridden) fails the *same* request much faster and with a *reported* error
instead of a sticky one:

```
exec_runner.cc:402: Execute fused operator(0/295) failed.
getitem_op.cc:455: index 395033601 is out of bounds for dimension 0 with size 388956160
op: getitem
  in:  bfloat16[388956160,], int64[388956160,]
  out: bfloat16[388956160,]
```

Three different indices were reported in that one run -- 737396059, 395033601,
439601509, all against the same 388956160-row buffer -- which is what garbage
looks like, not what a systematic offset looks like. A 389M-entry int64 index
tensor gathering from a 778 MB bf16 buffer is the offload path moving a layer,
and the single-GPU run (which passes) uses the same offload.

That matters for two reasons, and it is the state bug 1 is actually left in:

* it is a **jittor-reported out-of-bounds driven by an index tensor**, i.e. the
  same shape as the sticky `cudaErrorIllegalAddress` of the no-offload runs --
  there the access went to device memory and no `IndexFault`-instrumented op
  happened to be the first reader, so nothing named it;
* it is a **30-second repro on rank 1 instead of a 265-475 s one**, with the
  shapes and the op in hand. Any fix for the garbage index should be re-checked
  against the no-offload configuration, because that is the one the probe used
  and the one this document's earlier measurements are all stated in.

### Root cause: the residency manager's shared-storage idiom, and the shim's gather

Following that lead into the shim found the reader. `_as_strided` is not a real
strided view: it builds a full-length int64 index (`arange(size[d]) * stride[d]`
summed over the dimensions, plus `storage_offset`) and gathers with it out of
`self.reshape(-1)`. torch validates that request against the *storage*; the shim
did not check anything, so an oversized view became an out-of-bounds gather.

Adding that check (there is now a test for it) made the failing request name
itself in 5.1 s instead of after 30-475 s:

```
as_strided: sizes (75968, 5120), strides (5120, 1), storage_offset 388956160
  are too large for the 388956160 element(s) this tensor can address
  (the view would span [388956160, 777912319])
```

`75968 x 5120 = 388956160` is exactly how many elements the tensor *has*, and the
storage offset asked for is that same number: the caller wants a view of the
**second half** of a `777912320`-element buffer. That is torch's shared-storage
idiom -- `t.set_(storage, storage_offset=N)` to re-point a layer tensor at another
shard, which is what the residency manager does when it moves a layer between the
host and a rank -- and it is the one thing the shim cannot express:

* `set_`'s own docstring says so: "jittor has no user-visible byte storage, so ...
  the two tensors do not share memory";
* `_Storage.nbytes()` returns `numel x dsize` -- the *tensor's* elements, not the
  storage's, so the manager's arithmetic is working with half the buffer it
  believes in.

Both faces of bug 1 are this one gather: when the out-of-range index landed in
unmapped memory it was the sticky `cudaErrorIllegalAddress` of the no-offload runs
(reported later, at a memory query, with no site); when it landed inside the
larger offload allocation, jittor's `IndexFault` caught it first and named the
`getitem`. Same call, same offset, different landing.

So the jittor defect -- an out-of-bounds gather that manifested as an illegal
address -- is fixed at the root: the request now fails immediately and says
exactly which view it could not serve, and the device read is gone. Making the
*request* succeed needs one of two things, and they are not small: the shim would
have to track the tensor a view's storage came from and gather from *that*
allocation (real shared storage, which the materialising view model does not
have), or the residency manager would have to stop using the idiom. Neither is a
fix that can be assumed without the shape of the manager's decision to change.

`CUDA_LAUNCH_BLOCKING=1` is the other half of the picture. It fails *earlier*
(30.1 s) with the same rank-1 `cudaErrorIllegalAddress` at `cudaMemGetInfo`.
Making kernel launches synchronous did not move the error to a launch, and
jittor wraps its own launches in `LaunchErrorScope`, so the faulting operation is
**not** one of the launches jittor issues in that window. That leaves the paths
`CUDA_LAUNCH_BLOCKING` does not cover: `cudaMemcpyAsync` (the copy stream and the
H2D path) and NCCL's own driver-API launches -- for which a rank-1-only fault and
a sticky error surfacing at the next memory query are both expected.

### The fault is inside the extension call, at an allocation

The trace was extended to print the attention **result** and the ambient device
after each `packed_low_level.fwd(...)`, and the run fails in 20.1 s with the same
error every time now. What it shows on rank 1:

* every q/k/v is `dev=1 contig=True bf16 shape=(1,289,32,128)`, with pointers that
  repeat across calls (the same buffers are reused);
* every *returned* output is `dev=1` with a device-1 pointer, and
  `current_device=1` after each call -- the device-plumbing on the extension
  boundary is right, so the section-20 `CUDAGuard` and section-26/27 device work
  hold;
* the final call prints its q/k/v and `causal=... scale=...` and then **no result
  line at all**. The process dies inside that `packed_low_level.fwd(...)`.

That also explains why the error surfaces at `cudaMemGetInfo` in
`backends/cuda/runtime/driver.cc`: the query is jittor's allocator asking for free
memory, and the allocation that asks for it is one the **extension** makes for
its own buffer (`torch::empty` for out / `softmax_lse_accum` / `out_accum` goes
through the shim's tensor factory). So the sticky error was set by an
*asynchronous* operation before that allocation, and the allocation is merely the
first CUDA call after it.

Put together with the two controls, the residual is: an asynchronous read/write
on device 1 by something in the extension path, after ~100 identical calls that
worked, in the one configuration where jittor's device is not the process default
device. The next audit is the lifetime of the buffers the extension itself
allocates through the shim -- those are the ones no Python owner keeps alive, and
the only ones (unlike q/k/v, which the caller holds) whose recycling depends on
allocator timing rather than on a reference.

### A plain attention call is fine, so it is the context around it

`probe_encoder_sdpa.py` is the seconds-scale repro for the encoder's attention
(the SDPA call that `no_grad` routes to the dense flash-attn entry). On the same
core, on device 1 with device 0 as the ambient one -- the failing rank's
configuration -- it passes for S in {289, 489, 512, 1000, 1023} in both fp16 and
bf16, all ten through the official extension (`backend:
flashattn_jittor_official:/root/jittor-lab/flash-attention`, `hits: 10`), with
every tensor reporting device 1.

So there is no geometry-, dtype- or device-level defect in the attention call
itself. What the full request adds is *context*: a hundred calls deep, an
allocator that has been recycling for minutes, TP collectives in the same stream
graph, and a running executor. Combined with the trace (same geometry as the
calls that worked, correct devices, the process dying inside the call) and with
`CUDA_LAUNCH_BLOCKING` not moving the error to a launch, that is the signature of
a lifetime or ordering hole that only the long-running context can open -- not of
a wrong index.

### Components checked and cleared, so they are not re-audited

Each of these was a plausible seat for a device-1-only asynchronous fault, and
each is correct in the tree as it now stands:

* `compat/shim/cpp_extension/include/c10/cuda/CUDAStream.h` -- the extension
  launches on `cudaStreamPerThread`, and the freshly built `.so` is confirmed to
  contain it (the `strings` check above).
* `c10/cuda/CUDAGuard.h` -- `CUDAGuard` calls `cudaSetDevice` *and* moves jittor's
  own current device, which is what stops the extension's `torch::empty` buffers
  from being allocated on device 0. Its comment already describes this exact
  failure mode.
* The generated packed entry -- every entry constructs
  `at::cuda::CUDAGuard device_guard{q.device()}`, and the generated source in the
  build directory confirms it.
* `flash_api.cpp` -- every launch site takes
  `at::cuda::getCurrentCUDAStream().stream()`, and there is **no** runtime-API call
  that omits the stream. That is also why rebuilding the extension with
  `--default-stream per-thread` would be thirty minutes for nothing.
* The shim's `jtorch::Tensor` -- it holds `shared_ptr<VarHolder>`, so the buffers
  the extension allocates for itself (`out`, `softmax_lse`, `out_accum`,
  `softmax_lse_accum`) are owned and cannot be recycled while the C++ object
  lives. The Python-side `_mark_readonly_borrow` is skipped on the packed path,
  but that is not a hole: the packed path's inputs go through the generated
  entry's `jt_readonly_tensor`.
* `nccl_stream_begin` / `nccl_stream_end` -- the side-stream join records on the
  communication stream and waits on `BackendStreamKind::Compute`, i.e.
  `cudaStreamPerThread`, so the ordering target agrees with the per-thread
  compile.

What is left is the long-running context itself. The two things to point
`compute-sanitizer` at, when a run can be afforded, are the TP collectives'
driver-API launches (which `CUDA_LAUNCH_BLOCKING` does not serialise) and the
`cudaMemcpyAsync` on the copy side stream. NCCL is the weaker of the two now:
`NCCL_DEBUG=WARN,NCCL_DEBUG_SUBSYS=INIT,COLL` on a failing run prints nothing at
all -- NCCL initialises on the right device per rank and never reports an error.

### The generated packed entry is not the cause

`JITTOR_FLASH_ATTN_DIRECT_PACKED=0` turns off the direct/packed adapter, so the
model's attention goes through the *classic* dense low-level entry
(`flash_attn_2_cuda_jittor.so`'s `mha_fwd`) instead of the five-hundred-line
generated `flashattn_jittor_packed_fwd.cu`. That entry was worth suspecting: it
is generated, it was the seat of the section-20 device-guard bug, and its
`jt_fill_params` is a hand-port of upstream's `set_params_fprop` (it checks out
against upstream, including `seqlen_q_rounded = round_multiple(seqlen_q, 128)`
with `softmax_lse` sized `[batch, heads, seqlen_q]`).

The request fails **identically** with it disabled: 20.1 s, rank-1
`cudaErrorIllegalAddress` at `cudaMemGetInfo`, same signature. So both entry
paths fail, which is what they have in common -- `run_mha_fwd` and the extension's
kernels -- and the generated adapter is out.

That leaves the extension's kernels themselves against buffers whose contents
come from a long-running request, or an unchecked extension launch reading a
buffer jittor recycled. The measurement that would separate them is
`compute-sanitizer --tool memcheck` over one request; at roughly 10-50x, on a load
that already takes six minutes, that is a hours-long run and it was not
attempted here.

**Correction, later the same session: it was attempted, and it does not work.**
`compute-sanitizer --tool memcheck --target-processes all` on a TP2 `TORCH_SDPA`
serve (`LAUNCHER=... ./serve-vllmomni.sh`, which `serve-vllmomni.sh` supports for
exactly this) reaches the request and reports `ERROR SUMMARY: 202 errors`, but
**every one of them is NCCL**: 100 ×

    Program hit cudaErrorNoKernelImageForDevice (error 209) ...
    ncclInitKernelsForDevice -> ncclCommInitRankFunc -> ... -> jittor::nccl_init

and the application's failure is `exec_runner.cc:402` (the collective), not the
usual `cudaMemGetInfo` report. NCCL cannot initialise its own kernels under the
sanitizer, so the run never reaches the fault. There is also a structural reason
not to expect this tool to be decisive even if NCCL worked: memcheck reports
accesses outside an allocation, while a node or block that has been freed and
recycled *inside the same mapping* is still a valid address -- which is the shape
this fault has had all along (`Node::free()` clearing edges, the recycled-buffer
family of section 29).

The extension itself was re-checked as a suspect and cleared: of the four
`getCurrentCUDAStream()` launch sites and every `flash_api.cpp` runtime call,
none omits the stream, so there is nothing for a per-thread compile of the
extension to change and no reason to spend thirty minutes rebuilding it.

The next measurement is the attention call itself: `H3_FA_TRACE=1` prints q/k/v
device, contiguity and pointers immediately before each `packed_low_level.fwd(...)`,
and the last one before the fault is where a device-1 geometry or pointer
mismatch would show up.

## 31. `Tensor.to(1)` dropped the device index

Found while chasing the corrected residual of section 30 into the denoise loop,
whose `Recent launch candidates` are `copy`/`getitem`/`setitem` bursts around
`denoise_loop.py:189-207`. That code builds its per-step kwargs with in-place
scattered writes driven by precomputed tensors, with `x[0].index_copy_(0,
self.img_pos_dev, video_rows)` and `timesteps[self.img_pos_dev[mask]] = t` in the
middle of it. A repro of the same shape was written
(`probe_denoise_scatter.py`) and it failed immediately -- but on the *index*
tensor's device, not on the write:

```
tensors: img_pos=cuda:0 mask=cuda:1 x_base=cuda:1
dispatch_context.cc:52: Expected all tensor inputs on the same backend and device
```

`img_pos` was built as `torch.arange(ROWS, dtype=torch.int64).to(dev)` with
`dev=1`. The shim's `_to` classifies each argument as a dtype, a `torch.device`,
another tensor or a string, and a bare **int matches none of those branches**, so
it was dropped: `dev` stayed `None`, `dev = self.device` fell back to the tensor's
own device, and the tensor never moved. Isolated:

    .to(1)                        device=cuda:0   <- asked for 1
    .to("cuda:1")                 device=cuda:1
    .to(torch.device("cuda", 1))  device=cuda:1
    torch.arange(8, device=1)     device=cuda:1

torch itself raises on an int here (`TypeError: to() received an invalid
combination of arguments`), so the silent-drop is the one behaviour that cannot
be right: on a rank whose ambient device is not the tensor's, a caller that asks
for cuda:1 gets whatever it already had. Fixed by reading a bare int (excluding
`bool`, which is an `int`) as a device index.

**This is not the cause of the section-30 residual**, and saying so matters:
the model passes `device=self.device` and `self.device` comes from
`get_local_device()`, which returns a `torch.device`, and every `torch.device`
form was already correct. So the fix closes a real silent-misplacement hole in
the same family as sections 26/27, but the fault that survives is still open.

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

- 19: the deadlock is removed by construction -- in two phases, because one was not
  enough (see the follow-up in section 19) -- and repeated TP2 starts now pass the
  rendezvous in ~30 s instead of hanging about every other attempt. The request
  fault of section 21 is still open.

- 20: `probe_encoder_sdpa.py` under `no_grad` -- the condition that reaches the
  generated entries -- fails on device 1 before the six guards are fixed and
  passes on devices 0 and 1 after; the generator's output is asserted directly by
  `TestPackedEntryDeviceGuard`.

- 21: the shape trace above, from a 4-step 256x256 request on two cards.
- 25: pointer comparison + values + strides for `transpose` on CUDA (materialized
  copy, correct contiguous strides), and the same check after the reverted attempt,
  which is how the wrong-values regression was caught.
- 24: `py-spy dump` on the rank-1 worker across the request window: three
  threads, only the main one in the graph.
- 23: `probe_loader_migrate.py` with and without `SERIALISE=1` (fails / passes with
  four threads), and the launch list of the post-`ArrayOp` TP2 run, which names
  `torch.split` + `contiguous` immediately before the fault.
- 22: `probe_loader_migrate.py` -- the single-threaded phase segfaults in
  `jittor::ArrayOp::run` before the fix and passes after; `tests/core/test_array.py`
  covers the guarded path; a TP2 run loaded both ranks with zero illegal addresses.
  The four-thread phase still trips `fused_op.cc:89`, which is the next thing.
- 26: `probe_thread_device.py` asks jittor and the CUDA runtime for the current device
  from the main thread and from fresh threads -- `jt.current_device()=1` beside
  `cudaGetDevice=0` before the fix, `1`/`1` after, which is the assertion
  `tests/runtime/test_runtime_device_state.py::test_current_device_binds_the_calling_thread`
  now makes (it fails on the old core, passes on the new one; `libcudart` is loaded
  through `ctypes` and the test skips where it cannot be). The rank-1-only asymmetry is
  the point: worker threads on rank 0 default to the device rank 0 uses.
- 27: `JITTOR_TRITON_FAST_SYNC=0` (still fails), the `cuLaunchKernel` stream
  interception test (the argument is `0x2` by default, NULL with
  `JITTOR_TRITON_LEGACY_STREAM=1`) plus the run that survives it, a
  `CUDA_LAUNCH_BLOCKING=1` run that fails identically, and the
  `ATTN=TORCH_SDPA` run that completes both denoise steps with zero illegal
  addresses against `ATTN=FLASH_ATTN` runs that never finish the first -- with
  the resolved-backend lines from each log showing the backends really differ.
  `probe_unique_scatter.py` is the `unique` repro (device 1 with ambient 0
  raises; with ambient 1 numpy agrees) and `TestUniqueOffTheAmbientDevice` is
  its regression. `probe_ambient_ops.py` prints `device_id` beside the device
  jittor would really dispatch on for every step of the encoder's rotary chain
  (`jt.ones_like(idx)` was the one that read 0), and
  `TestLikeConstructorsKeepTheDevice` covers the family. The migration-target
  change in `var_holder.h` is a core edit -- it needs the 218-file rebuild, so it
  is verified by the rebuilt core plus a TP2 run, not by a Python test.
- 28: `H3_FUSE_DUMP=1` on `probe_loader_race.py threads 4 6` reproduces the assert
  once in three runs and prints the dump twice in that run (`ops=1`, `outputs=0`,
  `tflag` 68 vs a batch stamp of 67, `active_epochs=1`); the same probe with
  `JITTOR_SHIM_SERIALISE=1` is the negative result -- it segfaults in
  `jit_utils_core.so` instead of asserting, so serialising the shim's Python entry
  points is neither a fix nor a valid test of one. `flash_attn.__file__` is
  `site-packages/flash_attn/__init__.py`, byte-identical to the shim's stub before
  the instrument, and `flashattn_jittor_last_error()` is `import
  flash_attn_jittor_cuda failed: No module named 'flash_attn_jittor_cuda'`.

- 31: standalone, before/after, on a rank whose ambient device is not the
  tensor's (`CUDA_VISIBLE_DEVICES=1,2`, device 1 requested): `.to(1)` reports
  `cuda:0` before the fix and `cuda:1` after, while `.to(0)`, `.to("cuda:1")`,
  `.to(torch.long)` and `torch.arange(..., device=1)` are unchanged.
  `compat/tests/torch/test_multi_device.py::TestMultiDeviceFacade::test_to_and_cuda_with_an_index`
  gained the int cases, placed *first* in that test: the file's bare-`"cuda"`
  case at the end fails when the file runs alone (that is the pre-existing
  "15/16, isolated-run-only" failure of section 16 -- a bare `"cuda"` resolves to
  the ambient device, which in isolation is still 0), and an early abort would
  hide anything added after it. Whole file, through the installed package:
  `1 failed, 15 passed`, failing at the bare-`"cuda"` line; a sentinel print on
  either side of the new block confirmed it executes and passes (it is what
  located the failure at that later line rather than at the new assertions).
  Note the shim's test files cannot be collected from the repo root
  (`compat/__init__.py` does `from .._runtime import ...`, so pytest importing
  `compat` as a top-level package dies with "attempted relative import beyond
  top-level package" -- the untouched sibling `test_device_contexts` fails the
  same way there); use
  `PYTHONPATH=$REPO/python $VENV/bin/python -m pytest --pyargs jittor.compat.tests...`
  from the lab's `env-jittor.sh` environment.

Each fix was synced into the lab venv at
`$JITTOR_LAB_ROOT/_state/h3/venv-jittor/lib/python3.12/site-packages/jittor/`,
which is a **copy** of `python/jittor` plus the installed `jittor-torch`
compat package; the adapter checkout is editable.

## 32. The storage view model, and the triton bridge's CUDA device

Two independent defects on the H3 TP2 path, one closed and one open, plus the
repro that separates them.

### 32.1 `_Storage` / `set_` described a tensor, not its allocation (closed)

`PinnedModuleStager` (text_encoder offload) snapshots a module group as one byte
image of its storage and re-creates each member with
`set_(storage, offset, shape, stride)`. The shim reported
`_Storage.nbytes()` as the tensor's *own* `numel * dsize` while
`storage_offset()` is an absolute offset inside the shared allocation, so a
weight sliced out of a fused one -- a shard rank 1 has and rank 0 does not --
was restored past the end of the buffer it was given: the rank-1
`as_strided`/`getitem` out-of-bounds seen at 5-30 s.

Fixed (`74bfe44c`): `_Storage.nbytes()` is the byte extent from the allocation
origin to the tensor's last element (`_storage_offset()` +
`_storage_strides()`, both real exports), and `set_(storage, ...)` is
byte-addressed. A physically non-contiguous shard has no byte range to
materialise and is refused rather than packed into the wrong positions.

Verified: minimal repro `probe_storage_model.py` (cpu and cuda:3, contiguous
shard at a nonzero offset round-trips exactly; a stride-2 shard raises);
`test_contiguous_storage`, `test_torch_compat_load_strided` (10 passed),
`test_multi_device`, `tests/core/test_storage_strides` (10 passed). On the real
TP2 `ATTN=FLASH_ATTN` + text_encoder-offload request the log now contains zero
`as_strided` and zero out-of-bounds hits: the request runs `encode_prompt` and
enters the DiT denoise loop.

### 32.2 The triton bridge's driver was pinned to CUDA device 0 (closed)

`compat/triton/backend.py` retained **one** primary context
(`cuDeviceGet(&dev, 0)`, `cuDevicePrimaryCtxRetain`) and cached every
`CUmodule`/`CUfunction` in it. A module belongs to the context it was loaded
into, so a rank whose operands live on CUDA ordinal 1 launched its kernels in
device 0's context against device-1 pointers. Now `_Driver` is one instance per
device (`_Driver.get(ordinal)`, `_insts`), `run()` selects the operands' device
(and refuses mixed-device operands), and `ensure_ctx()` asserts both the driver
context and the runtime device so the guarded bounce buffers and the launch
agree. Regression test:
`compat/tests/triton/test_triton_backend.py::TestDriverIsPerDevice` (also
repairs that file's stale `_Driver.instance()` call, which no longer existed).

### 32.3 Why device 1 faulted, and the repro that found it (closed)

Both target configurations (`ATTN=FLASH_ATTN` TP2 with and without
text_encoder offload) used to fail on rank 1 with a sticky
`cudaErrorIllegalAddress`, 20-75 s in, reported at whatever CUDA call came next
(`cudaMemGetInfo`, a NCCL broadcast, a triton `cudaMalloc`, the triton launch) --
a different site per run, always device 1.

`probe_triton_device.py` reproduces it in **seconds** with no model: one
process, `CUDA_VISIBLE_DEVICES=1,2`, tensors on ordinal 1, one `@triton.jit` add
through the bridge. device 0 launched fine, device 1 raised
`cudaDeviceSynchronize -> cudaError 700`.

The cause was ordering inside the bridge, not the kernels. `_Driver.__init__`
called `cuCtxSetCurrent`, and the compilation path calls `_Driver.get()` for its
compute capability -- so the *first* triton launch of a rank-1 process switched
the current context to device 0 before the materialising `sync_all` ran. The
sync then waited on device 0's streams, device 1's operands were not committed,
and the kernel was handed device-1 pointers while device 0 was current:
uncommitted memory on device 0 (stale values), an illegal address on device 1.
Fixed (`95ad1ea2`): `__init__` only `cuDevicePrimaryCtxRetain`s (no context
switch), and `run()` selects the operands' device with `ensure_ctx()` *before*
compiling and materialising, so the runtime device and the driver context agree
for both the bounce buffers and the launch.

Verified: the probe is green on device 0 and device 1, and both real
configurations pass -- TP2, 2 steps, 256x256, `ATTN=FLASH_ATTN`:
`RESULT completed in 495.3 s` with no offload (`tp2-ctxfix-1.log`), and
`RESULT completed in 50.1 s` with the factory default text_encoder offload
(`tp2-ctxfix-offload.log`). Both logs contain zero `cudaErrorIllegalAddress`,
zero `as_strided` and zero worker tracebacks; the offload one also exercises
section 32.1's storage/`set_` fix end to end. Two operational notes the probe also produced, which are *not* this
fault: a stride-0 operand (`torch.ones(n)` through the shim is a broadcast view,
`s=[0] r=1/4096`) is read by such a kernel as `n` dense elements, which gives
wrong values on device 0 as well; and an operand whose last stride is not 1
cannot be served by the kernels the model uses.

## 33. Speed: jittor shim vs real torch, and TP2 vs TP1

Same script (`infer_h3.py`), same request (512x512, 124 frames, 6 steps, seed 0,
`--vae-dtype float16`, injected latents), one GPU each, both idle. The jittor side
uses its flash-attn extension; the oracle has no real flash-attn, so it runs
diffusers' `_native_flash` (torch SDPA's flash kernel). `load_seconds` is **not**
comparable and is omitted: the shim loads weights eagerly (385-392 s) while the
oracle's `ComponentsManager` builds lazily (2.13 s).

`generate_seconds`, and the per-phase split the script prints:

| phase | jittor, cold | jittor, warm | torch | warm jittor / torch |
| --- | --- | --- | --- | --- |
| dit | 61.41 | 25.92 | 23.35 | 1.11x |
| text_encoder | 17.25 | 3.77 | 10.18 | **0.37x** |
| vae.video | 66.50 | 15.69 | 6.79 | 2.31x |
| vae.audio | 36.90 | 5.64 | 0.28 | 20.1x |
| phase sum | 182.1 | 51.0 | 40.6 | 1.26x |
| **generate_seconds** | **253.69** | **92.34** | **75.94** | **1.22x** |

So: the first jittor run pays ~2.7x for JIT compilation and must not be used as a
speed number; warm, the shim is 1.22x off torch overall, is *faster* than torch on
the text encoder, is at parity on the DiT (1.11x), and loses on the VAEs -- the
video decoder by 2.31x and the audio decoder by 20x (5.64 s against 0.28 s, the
single largest per-phase gap and the obvious next target). Both sides wrote a
512x512 6-step mp4 of the same size (359,623 vs 358,341 bytes).

For the vllm-omni server path (a different pipeline, so not comparable with the
table above), 2 steps 256x256, `ATTN=FLASH_ATTN`, warm:

| load | request |
| --- | --- |
| TP1 (1 GPU) | 41.9 s |
| TP2 (2 GPUs) | 55.1 s |
| TP2 + text_encoder offload | 50.1 s |

and at 8 steps 832x480: **TP1 309.2 s against TP2 180.2 s** -- 1.72x, i.e. TP2 pays
off only once per-step compute outweighs the per-layer NCCL traffic, which a
2-step 256x256 request is far too small for.

The oracle needed its own venv on this box: `venv-oracle`'s torch is cu130 and
the driver is 535 (CUDA 12.9), and its `transformers 5.17` breaks the lab's
diffusers. `venv-oracle-cu129` (torch/torchaudio/torchvision `+cu129`,
`transformers==5.5.3`, venv-oracle's packages layered through a `.pth`) plus
`env-oracle-cu129.sh` / `run-oracle-cu129.sh` are the working pair; see
`agent/manuals/` or the lab notes for the recipe.

## 34. TP2 completed but its pictures were noise (closed: the shim's Generator had no stream)

`ATTN=FLASH_ATTN`, 2 steps, 256x256, seed 11223, same prompt: TP1 writes a
blurry-but-real frame, TP2 writes **coloured blocks**. Measured over the decoded
frames: TP1 mean 129.4 / std 30.5 / |dx| 9.8 (spatially smooth), TP2 mean 88.6 /
std 82.4 / |dx| 16.8 with |dt| 31.6 (flickering); per-pixel mean difference
82.6/255 (p99 183). Sheet: `runs/sheet-tp1-vs-tp2.png`.

It is not our recent work, and not the attention backend: a TP2 run from the day
before the storage/triton fixes (tag `tp2-sdpa-2step-256`, TORCH_SDPA) has the
same character, and single-GPU runs are fine (`runs/sheet-oldtp2-vs-single.png`).

Ruled out by measurement:

| hypothesis | result |
| --- | --- |
| too few steps | TP2 at 8 steps is still noise |
| text-encoder TP sharding | `TEXT_ENC_TP=1` (DiT still TP2) is still wrong |
| the shim's collectives | `probe_tp_dist.py` on 2 ranks: all_reduce sum=3.0, all_gather slots [0,1], broadcast, and the row-parallel "split + all_reduce" arithmetic all exact (max err 0) |

### The DiT *is* TP-sharded -- the "same footprint" reading is withdrawn

A first pass compared each rank's loaded footprint and found TP2 and TP1 equal
(10.2354 against 10.2402 GiB), which looked like unsharded weights. That number
covers the whole diffusion runner (VAEs included), so it does not isolate the
DiT, and a trace of the shim's collectives (`H3_TP_TRACE=1`, installed in the
deployed shim only: `_tp_trace` prints one line per distinct (kind, shape) and an
`[tptrace-sum]` count at exit) settles it the other way:

| rank | distinct collective operands seen |
| --- | --- |
| TP0 | `all_gather_into` (1,2688) (1,48384) (1,5376) (2368,2688) (289,2688) (3072,16) (3072,48) (414,2688); `all_reduce` (1,2688) (1,289,5120) (289,5376) (3072,5376) |
| TP1 | the same set, with **different operand heads** |

2688 = 5376/2 is the DiT hidden split, and the two ranks hand in different
values, so the DiT shards its weights and reduces them. The text encoder shows
up as `all_reduce (1,289,5120)` with 289 tokens unsharded (5120 = Qwen3-VL
hidden). So "the layers are not sharded" is **not** the fault.

### The sharpest remaining lead: packed rows versus their metadata

The DiT also moves a `(3072, 5376)` `all_gather_into`, i.e. the *packed rows*
are split across the two ranks as well as the hidden dim -- and
`MiniMaxH3SPPrepare` exists precisely to shard `hidden_states` *and* its
metadata (`rope_table`, `combined_indices`) together. The log says
`ulysses=1, ring=1, use_ulysses_low=True`, so the row split is not Ulysses'.
The modulation kernel that used to fault reads
`indices_ptr + row * stride_indices` for exactly those rows, so
"rows sharded, indices/rope not (or sharded differently)" is both the shape of
this failure and the shape of the original crash. Whether the three tensors
cross that boundary with consistent row splits is the next thing to look at.

### Every shim primitive on that path has now been checked (2026-09-17)

Each row is a probe in the lab, run against the shim, not a reading of the code.
`probe_tp_dist.py`, `probe_sp_chunk.py`, `probe_sp_gather.py` and the write
probes are single-purpose and take seconds to a minute:

| primitive / behaviour | probe | result |
| --- | --- | --- |
| `all_reduce` sum and mean, `all_gather`, `broadcast`, row-parallel split+reduce | `probe_tp_dist.py`, 2 ranks | exact |
| `Tensor.chunk` (1-D/2-D/3-D, non-divisible, `chunks > size`, on views) | `probe_sp_chunk.py` | matches numpy |
| the SP gather assembly `reshape([ws]+size) -> movedim(0,dim) -> reshape` at dim 1/2, and `movedim` alone | `probe_sp_gather.py`, 2 ranks | matches numpy |
| `copy_` into `param.data[a:b]`, both rank regions, fp32/bf16, cpu/cuda, and into a parameter that is itself a shard | `probe_slice_copy.py`, `probe_shard_write.py` | write lands |
| which model line calls each collective | `H3_TP_TRACE=1` + call-site capture | all from vllm `ColumnParallelLinear(gather_output=True)` / `RowParallelLinear` and the encoder, at shapes that are internally consistent (`video_out` 48 = 96/2, `audio_out` 16 = 32/2, DiT hidden 2688 = 5376/2, rows 3072 = the full sequence) |

So the fault is *not* a collective, a split, an assembly or a write: every
primitive that the TP2 path uses behaves as torch does. What remains is model
-side TP semantics -- which tensor is replicated where the code assumes a shard
(or the reverse), or a `shard_id`-to-region mapping -- and that lives in
vllm/vllm-omni. Finishing the localisation therefore needs a *dump-only*
diagnostic inside the lab's `vllm-omni` checkout (per-rank encoder weight shard
digests, or one forward's intermediate tensors); that is a change to the
component the work is not supposed to modify, so it is not done unilaterally.

### Root cause: the shim's `torch.Generator` did not own a stream (closed)

None of the primitives was wrong, because the bug was not in a primitive: it was
in **which numbers the two ranks started from**.

The H3 pipeline builds the initial latents with a *seeded CPU generator*:

    g = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(1, 24, latent_t, latent_h, latent_w, generator=g, dtype=torch.float32)

and its DiT shards **weights**, not the sequence -- so both ranks must denoise the
*same* latent. The shim's `Generator` kept only a seed and left drawing to
jittor's **global** generator (`_seed_from` reseeded the global stream, then the
factory drew from it). Measured directly: two `manual_seed(1234)` generators
returned *different* numbers, and a fresh `manual_seed(1234)` generator returned
different numbers again after the process had sampled 100 more values. So each
rank drew from its own global stream, advancing by different amounts (the
text-encoder sharding, the collectives, warmup), and the two ranks denoised
*different* latents. Every `RowParallelLinear` then all-reduced two halves
computed from different inputs: the sum is meaningless, and the picture is noise.
TP1 is unaffected because it has one rank -- which is why this looked like a
"TP2 correctness" mystery while everything about the TP itself checked out. It
also explains the earlier observations: `TEXT_ENC_TP=1` changed the *appearance*
(it changes each rank's stream advance) but not the conclusion, 8 steps stayed
noise, and the attention backend did not matter.

Fixed (`0ea3448c`): `Generator.manual_seed` gives that generator its own
deterministic stream (`numpy.random.default_rng(seed)`), and the random factories
(`randn`/`rand`/`randn_like`/`rand_like`/`normal`/`randint`/`randperm`) draw from
it when a generator is passed, honouring `dtype`. Without a generator the old
global-stream behaviour is unchanged. Regression test:
`compat/tests/torch/test_generator_streams.py`.

Verified:
* single process: same seed -> same numbers, different seed -> different,
  unmoved by the process's own sampling; the pipeline's exact call shape
  (`randn(1,24,T,H,W, generator=g, dtype=float32)`) reproduces;
* two ranks that first sample *different* amounts: `video_sum` identical
  (-350.171417) and `audio_sum` equal to the last print digit;
* TP2 request (2 steps, 256x256, seed 11223): the decoded frames go from
  `std 82.4` with `|dt| 31.6` (flickering noise) to `std 31.0` against TP1's
  `std 30.5`, i.e. the same statistics as a real sample
  (`runs/genfix-tp2.mp4`, `runs/sheet-tp1-vs-genfix-tp2.png`).

The lesson generalises: for a diffusion model whose TP shards weights, any
per-rank divergence in *inputs* (RNG streams above all) is fatal and looks
exactly like "the TP math is wrong". Check the streams before the kernels.

### How close TP1 and TP2 are now (same seed, 512x512)

| pair | mean abs diff | per-frame corr | block SSIM (16 px) |
| --- | --- | --- | --- |
| TP1 vs TP1 (same config twice) | 1.50/255 | 0.971-0.994 | 0.942-0.985 |
| TP1 vs TP2, **2 steps** | 8.29/255 | 0.982-0.985 | **0.972-0.985** |
| TP1 vs TP2, 8 steps | 16.70/255 | 0.40-0.76 | 0.693-0.958 |

Read it as: at two steps the two paths agree to block-SSIM ~0.98, i.e. the TP
computation is numerically equivalent to one rank; the 8-step figure is the
sampler amplifying that bf16 rounding (the same run twice is 1.5/255, so it is not
run-to-run noise, and it is not a structural error either -- the frames stay the
same scene and the same overlaid text, see
`runs/sheet-8step-tp1-vs-tp2.png`). Pixel-identical output between a weight-sharded
bf16 TP and a single GPU is not something a chaotic 8-step sampler can give, so
"correct" here means numerically equivalent per step, same content overall.

## 35. The pink title card: the over-read guard bounced a strided operand

The server path rendered a pink card with gibberish glyphs for the pottery
prompt at every TP and every step count, while the diffusers path -- same shim,
same weights, same prompt, same 8 steps and 512x512 -- rendered the prompted
scene. TP1 and TP2 agreed with each other, which is why this read as a TP
problem for as long as it did.

### What the card was not

Every candidate below was *measured* on the server's own configuration before the
real cause was reached. Each row is a lab probe, not a reading of the code.

| suspect | probe | result |
| --- | --- | --- |
| text conditioning | `probe_vllm_encoder_cond.py`, `probe_capture_server_cond.py` vs `oracle-512.cond.npy` | matches to bf16 slop (cos 0.9999960, rms 0.0374); the diffs path's own parity is 0.0372. Holds with the encoder offloaded and with the DiT offloaded. `encoder_hidden` reaches the DiT as `prompt_embeds` bitwise |
| initial noise | `probe_server_noise.py` vs the captured `dit_x`/`dit_audio_x` | reproduced bit-for-bit from `seed=0` (max\|d\| = 0), right shape, scale, tags and padding |
| sigma schedule, sampler | `minimax_h3_time_shift_sigmas` vs `MiniMaxH3Scheduler`; `minimax_h3_rf_v_to_x0` + `euler_eta0_step` vs `MiniMaxH3Scheduler.step` | identical at shift 12/3 for 2/6/8/50 steps; same `x0 = x_t + sigma*v`, `x_next = r*x_t + (1-r)*x0` |
| DiT weights | `probe_ckpt_layout.py` | `FL2VA/transformer` (native) and `transformer` (diffusers) agree exactly under the documented transforms (qkv grouped->qkv, `fc1` gate-first); no "Skipping" warnings |
| row packing | `probe_index_add_bf16.py` | `index_add_` correct in bf16/fp16/fp32, with duplicate indices and slice destinations |
| attention | `probe_sdpa_parity.py`, `probe_varlen_parity.py` | dense 4-D SDPA and packed `flash_attn_varlen_func` at `cu_seqlens=[0,10175,10176]` both match float64; varlen is bit-identical to the dense flash entry |
| DiT inputs | `probe_capture_server_cond.py` + checkpoint matmuls | `condition_proj`, the token refiner (replayed from the diffusers checkpoint, cos 0.99998), both patch projections, the `_embed` assembly and `t_emb` (t=0 at step 0) all reproduce |
| fused qk-norm+RoPE | `probe_fused_qk_rope.py` | the Triton kernel matches its eager reference and a half-split torch chain |
| fused AdaLN modulation | `probe_modulation_ops.py` | all three kernels match their CPU eager branches **on contiguous inputs** |

The last row is the one that was wrong, and the last two words are the whole
story.

### Root cause: the guard's flat copy is only valid for a contiguous operand

`compat/triton/backend.py` routes small tensor arguments through a guarded
bounce buffer so that a masked over-read lands in zeroed slack instead of an
unmapped page (`jittor` Vars are exactly sized, unlike a torch caching
allocator's rounded-up blocks). The copy is one flat `copy_dtod` of
`numel * elsize` bytes taken from `data_ptr()`, after which the kernel is given
the *bounce* pointer -- while the caller's own stride arguments still describe
the original layout. `_tensor_nbytes` sized it from the element count, with the
comment "contiguous tensors, which triton requires for these kernels anyway".

MiniMax-H3's DiT does not hand it contiguous tensors. `MiniMaxH3AdalnProj`
returns `tuple(x.chunk(expand_ratio, dim=-1))` of an `[M*3, 6*H]` projection, so
`shift_msa`/`scale_msa`/`gate_msa` are views with row stride `6 * H = 32256`,
not `H = 5376`. The bounce copied the first `numel` elements of the parent
buffer instead of the view's rows, and the kernel then walked past the copied
payload with stride 32256 into the zeroed guard. The launch succeeded and the
numbers were silently wrong -- in `rms_norm_indexed_scale_shift`,
`indexed_gate_rms_norm_scale_shift` and `indexed_gate`, twice per block, in all
50 blocks, at every step.

Measured directly (`probe_modulation_strides2.py`, same values, four layouts):

| shift / scale layout | fused op vs float64 |
| --- | --- |
| both contiguous | cos 0.99999863, max\|d\| 0.124 |
| shift strided (offset 0), scale contiguous | cos 0.99293, max\|d\| 2.39 |
| shift contiguous, scale strided (offset `H`) | cos 0.99325, max\|d\| 21.96 |
| both strided -- the DiT's real call | cos 0.98616, max\|d\| 21.21 |

and the op's *own CPU eager branch* handles those same strided tensors correctly
(cos 0.99999754), which is why comparing the Triton path against its eager
reference with contiguous random data had passed. The error is small per step and
systematic, so the sampler compounds it: at 2 steps the frames are a blur, at 8 a
wooden disc on a pink field, at 50 a pink card with glyphs -- the conditioning
being correct, the model simply degenerates. It is identical on both ranks,
which is exactly why TP1 and TP2 "agreed" while both were wrong.

### The fix

`_tensor_is_contiguous` in `compat/triton/backend.py`, and the bounce is taken
only for operands it accepts. It reads `Var._storage_is_contiguous` for jittor
Vars and `is_contiguous()` for the shim's torch-shaped tensors, falls back to a
row-major stride check, and trusts only a genuine `bool` -- `Var.__getattr__`
synthesises a proxy for unknown names. An undetermined layout is reported as
non-contiguous: skipping the bounce only risks a masked over-read reaching the
allocator's slack, whereas bouncing something strided corrupts results. The guard
itself is unchanged for the operands it was written for (weights, index and
segment tables, all contiguous).

### Verification

* `probe_modulation_strides2.py`: all four layouts now cos 0.99999863,
  max\|d\| 0.124 -- identical to the contiguous case.
* Block-level A/B on the rows a live request produced
  (`probe_dit_block_parity.py`, vLLM-Omni's block 0 vs diffusers' block 0):

  | stage | before | after |
  | --- | --- | --- |
  | adaln shift/scale/gate | cos 0.9999976 | unchanged |
  | norm1 + modulation | cos 0.84167 | **0.99998** |
  | attention output | 0.93299 | 0.99727 (bf16 at these magnitudes) |
  | gated residual 1 | 0.98274 | 0.99938 |
  | norm2 + modulation | 0.54199 | 0.99915 |
  | mlp output | 0.51228 | 0.99957 |
  | block output | cos 0.18744 | **0.99978** |

* Regression test `TestGuardedBounceRequiresContiguous` in
  `compat/tests/triton/test_triton_backend.py` (strided operand through
  `matmul_kernel`'s explicit strides, contiguous control, helper semantics).
  Against the pre-change bridge the strided case fails with
  `Mismatched elements: 2047 / 4096 (50%)`; with the fix all three pass.
* End to end through the OpenAI endpoint, `t2va`, the pottery prompt, 8 steps,
  512x512, `seed=0`, after the fix:

  | run | result | request time |
  | --- | --- | --- |
  | TP1, one GPU, 512x512 | prompt-following clip (`runs/fix-tp1-8s.mp4`) | 118.9 s |
  | TP2, two GPUs, 512x512 | prompt-following clip (`runs/fix-tp2-8s.mp4`) | 130.1 s |
  | TP1, one GPU, 832x480 | prompt-following clip (`runs/fix-tp1-8s-832.mp4`) | 188.3 s |
  | TP2, two GPUs, 832x480 | prompt-following clip (`runs/fix-tp2-8s-832.mp4`) | 170.2 s |

  All four clips show the potter's hands shaping a clay bowl on the spinning
  wheel with the studio shelves behind, i.e. the structured prompt, in place of
  the pink card, and TP1 and TP2 agree on composition. The pair ordering is the
  known profile: at 512x512 the per-step collectives are paid without the rows
  being long enough to win them back, so TP2 is 9% slower; at 832x480 TP2 is 10%
  faster. Against real torch the denoise parity measured earlier stands (shim
  1.11x on the DiT, 1.22x end to end on the diffusers path). The fix cannot slow
  anything down -- for the strided operands it removes a device-to-device copy
  per launch; the contiguous operands it still guards are unchanged.

### Lesson

A fused kernel verified against its own eager reference proves nothing about the
*bridge*: the two agreed bitwise here because both were fed contiguous tensors,
which is not what the model passes. When a compiled path diverges while every
kernel looks right, check the layouts the model actually hands over -- views,
`chunk`s, `split`s, slices -- not just the values.

## 36. Speed: what the server's defaults actually cost

Asked whether the server is slower than the diffusers path, the phases were
measured on both. The two are **not** comparable as configured -- the diffusers
harness runs the video VAE in float16 while vLLM-Omni's H3 VAE is loaded and
decoded in float32 -- and once that is equalised the server is still ahead of
neither: it is 1.35-1.55x slower on the VAE at matched precision (section
below). The gap is real, and it is mostly the VAE.

`t2va`, the pottery prompt, 8 steps, 512x512, seed 0, jittor shim on both sides.
vLLM-Omni is TP1 with the recipe's default offload; the server emits these with
`--enable-diffusion-pipeline-profiler`.

| phase (s) | diffusers, fp16 VAE | diffusers, fp32 VAE | vLLM-Omni (fp32 VAE) |
| --- | --- | --- | --- |
| text encoder | 3.91 | 3.75 | 8.65 |
| denoise | 35.63 | 37.36 | 42.25 |
| **video VAE decode** | **15.75** | **335.64** (offload artifact) | **67.11** |
| audio VAE decode | 6.04 | 5.59 | 12.50 |
| total | 102.85 | 420.98 | 130.92 |

Flipping only `--vae-dtype` on the diffusers harness moves `vae.video` from
15.75 s to 335.64 s and leaves every other phase alone. That 335.64 s is **not**
a decode cost, and reading it as one is how an earlier version of this section
reached the wrong conclusion: the same latent decoded standalone takes 30.14 s
under the same fp32 VAE (row below). What the harness pays there is its
auto-CPU-offload (`--offload-margin 12GB`) working against a VAE that just
doubled in size -- an offload effect, not arithmetic.

### The VAE at matched precision

No serve flag exposes vLLM-Omni's VAE dtype (both H3 VAEs hard-code float32 at
load and the decode re-upcasts), so the two modules were driven directly on the
same latent -- the one a live server request decoded -- in `probe_vae_precision.py`:

| video VAE decode of `(1, 24, 37, 32, 32)`, same shim | fp32 | fp16 | bf16 |
| --- | --- | --- | --- |
| vLLM-Omni `MiniMaxH3VideoVAE` | 40.82 | 31.20 | 28.92 |
| diffusers `AutoencoderKLMiniMaxH3` | 30.14 | 20.07 | -- |

Read across:

* **at matched precision the server's video VAE is 1.35x (fp32) and 1.55x (fp16)
  slower than the diffusers one.** So the answer to "is the server slower than
  the diffusers path" is yes, and the VAE is where. The earlier claim in this
  section that equal precision puts the server 3.2x ahead is withdrawn: it
  compared vLLM-Omni's fp32 decoder (which keeps fp16 decoder-block Linears under
  the decode's autocast plus fused qk-norm-rope / scaled-residual / silu-and-mul)
  against the diffusers *pipeline* fp32 figure of 335.64 s, which is the offload
  artifact above;
* vLLM-Omni's default decodes in fp32, worth another 1.3x over its own fp16 on
  this latent. That is its own choice for fidelity and is not exposed as a serve
  flag;
* the other phases, same round: the text encoder is 8.65 s against 3.91 (its
  encoder is layerwise-offloaded, and that offload is forced -- a no-offload TP1
  run OOMs, 51.5 GB encoder + DiT + fp32 VAE against 95 GB), the denoise 42.25
  against 35.63 (1.19x) and the audio VAE 12.50 against 6.04 (the audio decode is
  wrapped in `_AudioVAEDeterminismContext`, which disables flash/mem-efficient
  SDPA and cuDNN for reproducible soundtracks).

Net for the request as configured -- diffusers harness at fp16 (102.85 s) against
the server at its defaults (130.92 s) -- is about 27%, and the VAE accounts for
most of it. For the like-for-like shim-vs-torch ratios the earlier sections still
stand (DiT 1.11x, end to end 1.22x, both measured at one precision on one stack).

## 37. The VAE gap is the shim's fp16 path, not the dtype choice

Section 36 left the video VAE as the largest single cost on the server and
1.35-1.55x behind the diffusers class at matched precision. Putting a real-torch
baseline beside it turns "slower" into a specific defect.

Same latent (`server_latent512.npz`), same class where possible, one decode each,
on an H20. The reference row runs the checkpoint's own decoder class under real
PyTorch (`probe_vae_torch_baseline.py`); the other two run the shim
(`probe_vae_precision.py`, `probe_vae_fp16_gap.py`).

| decode of `(1, 24, 37, 32, 32)` | autocast fp16 | fp32 |
| --- | --- | --- |
| real torch, checkpoint's class | **6.50 s** | 28.54 s |
| shim, vLLM-Omni wrapper | **83.65 s** | 29.11 s |
| shim, diffusers class | 20.07 s | 30.14 s |

Two things follow:

* **in fp32 the shim is fine**: 29.11 s against torch's 28.54 s on the same class
  and latent. Whatever else is going on, this is not a general VAE slowdown;
* **under `autocast(fp16)` the shim is 2.9x slower than its own fp32**, while real
  torch is 4.4x *faster* than its own fp32. The shim's autocast is not a no-op --
  a matmul inside the region does come back fp16 (`probe_vae_fp16_gap.py`) -- so
  the regime the reference decode and the server both use is genuinely fp16, and
  it is the slow one. Casting the module to fp16 outright (no autocast) gives
  31.20 s, i.e. no better than fp32 either.

So the server's 67.11 s in-pipeline decode is the shim's fp16 execution, not a
precision choice that could be configured away. Which op pays for it is **not
established**: `probe_autocast_ops.py` timed conv3d, group_norm, silu, padding and
matmul with and without autocast, but its conv3d figures (0.04-0.05 ms for a
512->512 3x3x3 conv over 4x32x32, i.e. >1 PFLOP/s) are past what the part can do,
so those calls were not executing inside the timed loop and the per-op numbers
must not be used. The one plausible-looking reading, group_norm at 352 ms fp32
against 795 ms under autocast, is itself suspiciously far above a sane cost for
16.8M elements. Attributing the 83.65 s needs a real profile of the decode
(jittor's profiler or nsys), which is the next step.

This is the one place in this whole enablement where the shim is behind torch by
more than a small factor, and it is worth stating plainly: at matched precision
and matched op the server's video VAE runs 3-13x slower than real torch, and the
fp16 path is the reason.

## 38. The fp16 VAE gap was the Triton bridge: a device sync per operand

Section 37 established that the server's fp16 VAE decode is the slow one and left
the cost unattributed. It is the Triton bridge, and it is three separate defects
-- one of them a silently stale deployed binary.

### What the profile says

`JITTOR_TRITON_STATS=1` on one autocast decode (`probe_decode_nsys.py autocast`)
reports, over its two decodes and 9,072 launches:

```
launches=9072 total=131.3s sync=45.8s pack=42.9s launch=0.1s final=42.4s
avg=0.0145s | ptr=41.5s/36288 alloc=0.0s/3 memset=0.3s dtod=0.2s
```

`ptr` is 41.5 s over 36,288 calls -- **1.14 ms to read one device pointer**. nsys
agrees on the mechanism without a per-op Python tax: the autocast decode issues
63,816 `cudaDeviceSynchronize`, 54,444 `cudaMemcpy` and 27,216 `cudaMemset`,
against fp32's 312 / 12 / 0. Four pointer reads + one `jt.sync_all` + two
`drv.synchronize` per launch is 7 x 9,072 = 63,504, which is the 63,816.

So the GPU is not the problem: nsys puts autocast's kernel time *below* fp32's
(32.1 s against 60.6 s over the same two decodes) while GPU busy-ness falls from
87% to 19%. fp32 never launches Triton at all -- which is why only autocast moved.

### Defect 1: the deployed core predated `Var.device_ptr_ready`

`compat/triton/backend.py::_tensor_ptr` prefers `v.device_ptr_ready`, added in
commit `c34b1240` ("H3 VAE 解码 671s -> 9.5s") for exactly this loop. The lab's
`jittor_core.so` (built 2026-09-17) exported `device_raw_ptr` but not
`device_ptr_ready` -- `strings <so> | grep -c ptr_ready` was 0 -- so the shim fell
back to the accessor that calls `sync(true, false)`, i.e. a full device drain for
every operand of every launch.

Rebuilding exposed a real bug in that upstream accessor: `Var::allocator` is null
until `Var::alloc`, and the residency test dereferences it, so reading the pointer
of a never-materialised holder was a null dereference. Importing such a core
segfaults inside the generated getter with no traceback. Fixed by materialising
once when there is nothing to point at:

```cpp
inline uint64 device_ptr_ready() {
    if (!var->mem_ptr) sync(true, false);
    ...
}
```

Nothing covered the accessor, which is why it shipped broken:
`tests/backends/cuda/test_device_ptr_ready.py` now does (fresh holder, device
pointer identity, host-resident migration).

### Defect 2: the over-read guard copied back with blocking `cudaMemcpy`

The guard bounces each contiguous, exactly-sized operand into a zeroed-slack
buffer and copies the payload back afterwards. Both directions used `cudaMemcpy`
(host-blocking) and the copy-back added its own `drv.synchronize()`, so four
bounced operands meant four device drains plus an extra full-device wait per
launch. The copies are strictly ordered around the kernel anyway -- copy-in,
kernel and copy-back are all on `_launch_stream()` -- so they now go through
`cudaMemcpyAsync` on that stream, before the single wait.

### Defect 3: the barrier before packing waited for the device

`run()` did `jt.sync_all(True)` before reading pointers, which submits the operand
graph *and* waits for it. jittor puts its own launches on `cudaStreamPerThread`,
and the bridge launches on that same stream, so the kernel is already ordered
behind its producers; waiting only idles the device with nothing to overlap. In
the shim's fast-sync mode it is now `jt.sync_all(False)`, which still plans,
allocates and enqueues and skips only the trailing device wait.
`TestLaunchFollowsItsProducers` pins the invariant this rests on, reading the
operand's device buffer back with `cudaMemcpy` to prove each launch saw the value
its producer had just written.

### Measured effect

Per launch of the VAE's own kernel shape (8,192 x 8 x 64 fp16 q/k plus a 48-wide
cos/sin, `probe_triton_bridge_launch_cost.py`):

| build | ms/launch |
| --- | --- |
| baseline | 16.65 |
| + `device_ptr_ready` | 4.86 |
| + async bounce copy-back | 2.72 |
| guard disabled (floor) | 2.55 |

The same decode, same protocol, `probe_vae_checksum.py` / `probe_autocast_env.py`
(fp32 then autocast, three decodes per regime, warm caches):

| | fp32 | autocast fp16 |
| --- | --- | --- |
| baseline | 28.1 s | 84.4 s |
| + per-op import caches | 28.1 s | 79.7 s |
| + `device_ptr_ready` | 28.1 s | 51.3 s |
| + async bounce copy-back | 28.1 s | 47.5 s |
| + non-waiting operand barrier | 28.3 s | 30.1-32.5 s |

fp32 is untouched throughout, as it must be -- it never enters the bridge. The
bridge's own totals fall from 131.3 s to 59.4 s (`sync=41.9s pack=1.5s
final=15.8s`), which is the same 2x.

### Verification

* **Values.** The decode is not bit-reproducible, so the check is tolerance-based
  and controlled: two runs of one build already differ (fp32, which never enters
  the bridge, by max 3.2e-06; autocast by 2.0e-03). The barrier change moves
  autocast by 4.4e-03 -- the same order, with no non-finite values. Operand
  freshness is proven directly, not by readback (see `TestLaunchFollowsItsProducers`).
* **Triton suite.** `compat/tests/triton/test_triton_backend.py` is 5/15 before
  and after every change, with an identical failure fingerprint (same md5 of the
  mismatch lines) on the pre-fix core as well, so none of it is a regression.
* **End-to-end, both parallelisms.** `t2va`, pottery prompt, 512x512, 8 steps,
  seed 42, FLASH_ATTN, through the server's own `/v1/videos` endpoint:

  | | TP1 | TP2 |
  | --- | --- | --- |
  | inference | **85.7 s** (was 118.9) | **85.5 s** (was 130.1) |
  | clip | 124 frames, 512x512, stereo audio | same |
  | frame mean/std | [101.1, 73.3, 49.8] / 64.26 | [101.0, 73.3, 50.3] / 64.41 |

  Both render the expected pottery scene, and TP1 is within 0.2 s of TP2. The
  residual TP1-vs-TP2 pixel difference (max 179-232, mean 11-19 of 255) matches
  the pre-fix validated pair (max 201-211, mean 10.0-10.4), so it is the inherent
  TP1/TP2 parallelism difference and not something this change introduced.

### Two pre-existing findings this turned up (not addressed here)

* **`jt.zeros` is a zero-stride broadcast view.** `jt.zeros(n)._storage_strides()`
  is `[0]` where `jt.array(np.zeros(n))` gives `[1]`. A kernel writing through a
  raw device pointer then fills the allocation correctly -- the device buffer
  verifies bit-exact against the reference -- but jittor's *readback* honours the
  stride and returns element 0 broadcast. That is what the triton suite's 9
  `TestTritonBackend` cases actually hit (their outputs are `jt.zeros`), and it
  predates all of this. The bridge does not validate an output operand's layout;
  a non-contiguous output is neither bounced nor refused.
* **TP2 startup needs a working `mpicc` first on `PATH`.** jittor's `setup_mpi`
  probes `mpicc --showme:compile`; `/jizhicfs/leoyizhang/anaconda3/bin/mpicc` is
  broken (`x86_64-conda-linux-gnu-cc: command not found`) and now that the probe
  cache is cold it aborts startup instead of warning. Putting
  `/usr/local/openmpi/bin` first resolves it.

## 39. The last per-launch wait, and the four-way VAE matrix behind it

Section 38 left the bridge at 2.72 ms/launch with `final` = 25.0 s of its 33.2 s
total. That `final` was almost entirely one `drv.synchronize()` after the launch.

### Where the 2.5 ms actually was

`probe_bridge_floor.py` ablates the post-launch work in-process, and timing each
step from inside `run` gives the same answer:

| step of one launch | ms |
| --- | --- |
| `drv.synchronize()` | 2.070 |
| the four bounce copy-ins | 0.069 |
| the four guard memsets | 0.028 |
| `cuLaunchKernel` | 0.011 |
| `jt.sync_all` (the operand barrier) | 0.007 |
| Python in `run` (compile lookup, packing) | ~0.32 |

The wait looked load-bearing until it was measured against the GPU: queueing 60
launches with no per-launch sync takes 22.7 ms of host time and then draining the
launch stream takes **0.5 ms in total** -- 0.008 ms/launch of real GPU work. A
`cudaDeviceSynchronize` on an idle device is 1.7 us. So the 2.07 ms bought nothing
but latency, and every sync primitive (cudart device, cudart stream, jittor's own)
cost the same.

It bought nothing because the bridge now launches on `_launch_stream()`, the same
`cudaStreamPerThread` jittor schedules its own ops on, so kernel, bounce copies and
producers are ordered against each other by the stream. The wait dates from when
the bridge launched on the legacy NULL stream, which has no such order -- that is
the hazard it was papering over, and the reason the `cudaErrorIllegalAddress` in
section 34 needed the launch moved rather than the wait kept.

So `_sync_after_launch_enabled()` now defaults to the conservative answer *outside*
the shim and to no wait inside it (`JITTOR_TRITON_SYNC_AFTER_LAUNCH=1` restores it,
`=0` removes it everywhere), and a launch that allocates global scratch still waits
because `drv.free` returns that memory to the driver rather than to a stream. The
guard pool is safe to reuse without a wait: `guard_acquire`'s memset covers only
the guard tail, disjoint from the payload the copy-back reads.

| | before | after |
| --- | --- | --- |
| per launch | 2.65 ms | **0.515 ms** |
| VAE autocast decode | 30.9 s | **17.25 s** |
| VAE fp32 decode | 28.33 s | 28.33 s |
| end-to-end TP1 512x512 | 85.7 s | **78.7 s** |
| end-to-end TP2 512x512 | 85.5 s | **77.0 s** |

Verified as in section 38: the decode differs from the previous build by max
2.2-2.4e-03 against a 1.95e-03 same-build noise floor (smaller mean, no non-finite
values); the triton suite is 6/16 with a fingerprint identical to the pre-change
run; TP1 and TP2 both render the pottery scene, and their divergence (max 204-219,
mean 9.4-17.7) matches the pre-fix validated pair (max 201-211, mean 10.0-10.4).

### The four-way matrix: where the remaining gap is

One protocol, one latent `(1,24,37,32,32)`, 1 warm-up + 2 timed decodes
(`probe_vae_matrix.py`):

| runtime | VAE implementation | fp32 | autocast fp16 |
| --- | --- | --- | --- |
| jittor shim | vLLM-Omni wrapper (fused Triton) | 28.25 | 29.80 -> **~17** |
| jittor shim | diffusers class | 27.50 | 14.09 |
| jittor shim | checkpoint class `decode_base` | 27.61 | 13.66 |
| real torch | checkpoint class `decode_base` | 28.55 | **6.17** |

Two cells cannot be measured on this box, both for environmental reasons: real
torch's `vllm` wheel is built against CUDA 13 and only 12.9 is installed, so
`import vllm_omni` fails there; and torch's diffusers class allocates ~88 GiB even
at the 256x256 latent and OOMs on a 95 GiB card, where jittor decodes it in 3.4 s.

That leaves **two separate gaps**, which the table separates cleanly:

* **fp32 -- none.** 27.5 / 27.6 / 28.3 against torch's 28.6. Where the bridge is not
  involved the shim is at parity.
* **Plain fp16 execution -- ~2.2x, no Triton involved.** The checkpoint class under
  autocast is 13.66 s on the shim and 6.17 s on torch, and the shim run logs no
  Triton launch at all. This is ordinary fp16 op execution, and it is the next
  defect. `JITTOR_TORCH_KEEP_FAST_MATH` is the first knob to try: the shim forces
  `cuda_kernel_math='strict'` (`compat/shim/preflight.py`) unless that is set, and
  strict math is exactly what would keep fp16 off the fast kernels.
* **The bridge -- ~2.1x on top.** Under the shim, vLLM-Omni's wrapper (29.80) is
  twice the eager classes (13.66/14.09) at the same dtype, because vLLM-Omni's
  `install_h3_vae_optimizations` routes to fused Triton kernels and the diffusers
  and checkpoint classes do not. On torch that same fused path is what buys 6.17 s.

## 40. The request is GPU-bound now, so the rest of the gap is kernel work

Sections 38-39 removed the host-side bottlenecks in the Triton bridge. The next
question is what is left, and the answer is: not host work.

### Where the request spends its time

`--enable-diffusion-pipeline-profiler` on a 512x512, 8-step request (one TI1
server, seed 42):

| phase | time |
| --- | --- |
| `MiniMaxH3Pipeline.encode_prompt` (text encoder) | 8.80 s |
| `MiniMaxH3Pipeline.diffuse` (8 steps of the DiT) | **40.90 s** |
| `MiniMaxH3Pipeline.video_vae.decode_latent` | 21.63 s |
| `MiniMaxH3Pipeline.audio_vae.decode_latent` | 6.01 s |
| `forward` (whole request) | 77.67 s |

`JITTOR_TRITON_STATS=1` on the same server reports the bridge across the whole
pipeline as 6,250 launches and 23.9 s -- `sync=9.8s pack=4.9s launch=5.6s
final=3.6s | dtod=7.4s memset=0.2s ptr=0.1s`, i.e. 3.8 ms/launch. That looks like
30% of the request, and the guard's bounce copies (`dtod`) alone look like 7.4 s.

### But that time is hidden, and the device is saturated

Turning the over-read guard off removes almost all of it -- and the request gets
*slower*:

| | guard on | guard off |
| --- | --- | --- |
| `total` / `sync` | 23.9 / 9.8 s | **12.3 / 11.3 s** |
| `pack` / `launch` / `final` | 4.9 / 5.6 / 3.6 s | 0.7 / 0.3 / 0.1 s |
| `dtod` / `memset` | 7.4 / 0.2 s | 0.0 / 0.0 s |
| request wall | 77.7 s | 84.4 s |

The phases trade against each other (`sync` rises by exactly what `launch` and
`final` lose) because the host runs ahead of the GPU and blocks wherever it can
no longer go further. Sampling `nvidia-smi` through a request shows **100% GPU
utilisation from start to finish**. So the bridge's 23.9 s is not 23.9 s of
request: it is host work overlapped with a saturated device, and shaving it
further cannot help. That is why the host-side fixes of sections 38-39 moved the
wall clock a great deal (118.9 -> 78.7 s end to end) while this one moves it not
at all -- they were removing the bottleneck, this is past it.

### What the remaining gap is, kernel by kernel

Same class, same latent, same protocol, one decode each way, nsys, three decodes
per report (one warm-up plus two timed):

| | GPU total | instances | the three big gemms | attention | elementwise |
| --- | --- | --- | --- | --- | --- |
| shim | 29.1 s | 526,900 | nvjet_hsh: 12.35 s | flash: 2.84 s | ~8 s of `func_*` jit kernels |
| real torch | 20.4 s | 262,717 | nvjet_hsh **`_bias_`**: 13.11 s | cuDNN flash: 1.89 s | ~3 s of `at::native` |

The dominant matmuls are the *same kernels* and the shim's are marginally cheaper
(12.35 against 13.11 s -- torch's are the bias-fused variants, so they do more per
call). The 8.7 s GPU delta is entirely in the small stuff:

* **jittor's elementwise kernels are scalar.** One f32->f16 cast costs 2.06 s over
  33,894 calls against torch's 0.95 s over 34,020 calls -- same count, 2.2x per
  call. The generated kernel is
  `op0_yp[id1] = ((float16)(float16(op0_xp[id1])))` in a thread-strided loop, one
  element per iteration, with a redundant double cast. jittor's vectorisation
  machinery exists (`VectorizePass`) but is gated on `cc_type == "icc"` because
  "only icc supports the pragmas these emit", and `float4`/`half2` appear nowhere
  in `src/codegen/` -- so on nvcc there is no 128-bit computed-elementwise path at
  all. (`src/type/fp16_compute.h` *does* vectorise bulk `vload`/`vfill` copies, so
  this is specific to computed loops.)
* **no bias fusion.** torch's gemms are `nvjet_hsh_..._bias_`; jittor computes the
  bias add as a separate elementwise pass.
* **flash-attn 1.5x cuDNN's.** 2.84 s against 1.89 s for the same 6,804 calls, once
  the server's own attention env is set (without it, jittor decomposes attention
  into explicit cutlass gemms costing 6.06 s more).

So the honest position is that the remaining gap is three kernel-level items, of
which the first is the largest and needs a new codegen path rather than a fix.

### Which torch references this box can and cannot produce

Getting a torch number for every phase turns out to be environment-limited, and it
is worth writing down so nobody re-runs these:

* **torch + vLLM-Omni: impossible here.** `venv-oracle-cu129`'s `vllm` wheel is
  built against CUDA 13 (`libcudart.so.13`), and only 12.9 is installed, so
  `import vllm_omni` fails outright. The checkpoint's own decoder class is the
  stand-in for the VAE (`MiniMaxH3VideoVAE.decode_latent` calls exactly
  `model.decode_base`), and that is what sections 38-40 measure against.
* **torch + the diffusers pipeline: blocked.** `run-oracle-cu129.sh` defaults to
  `--attn-backend flash` and the oracle venv has no usable `flash-attn`
  (`Please install flash-attn>=2.6.3`), so the run dies at
  `set_attention_backend`. A torch phase breakdown with a *matched* attention
  backend (the shim runs the bridged official flash-attn) is therefore not
  available, which is why there is no torch column for the denoise or the VAE in
  the phase table above.
* **torch + the diffusers VAE class: OOMs.** ~88 GiB for a decode the shim does in
  3.4 s, at the 256x256 latent as well as 512x512, on a 95 GiB card.

So the trustworthy torch baseline is the VAE class (28.55 s fp32 / 6.17 s
autocast), and the phase-level torch reference is missing for environmental
reasons rather than for want of trying.

### Closing the rest is a kernel library, not a fix

The remaining delta needs per-op vectorized kernels for the CUDA backend, and it
is worth being precise about why no smaller change gets there. jittor's
`ParallelPass` turns an elementwise loop into a thread-strided scalar loop whose
*body* is generated C++ text (`yp[id1] = op(xp[id1])`). Rewriting the loop's index
math cannot vectorize a body whose `op` is arbitrary -- `exp`, a comparison, a
reduce -- which is exactly why torch's fast paths say `vectorized`: they are
hand-written per-op functors (`vectorized_layer_norm_kernel`,
`gpu_kernel_impl_nocast` with vectorized elementwise functors). `VectorizePass` is
not that; it emits `#pragma vector` for icc and lets a C compiler do the work, and
`float4`/`half2` appear nowhere in `src/codegen/`. `src/type/fp16_compute.h` shows
the shape of the work -- it has vectorized `vload`/`vfill` for *bulk copies* -- but
nothing for computed loops.

The measured prizes, per elementwise family, are modest individually and real
together: the f32->f16 cast 2.06 -> ~0.95 s, flash-attn 2.84 -> ~1.89 s (cuDNN's),
and the bias adds folded into the gemm as torch does. On the VAE class that is
~2.9 s of a 29.1 s GPU budget, and the same codegen serves the denoise, which is
53% of the request.
