# Exit-time "corrupted double-linked list" in the MiniMax-H3 run

- Status: Fixed; verified over repeated real runs
- Date: 2026-09-14
- Baseline commit: `f43b90d6` (the fix, on top of `7f7c8c0e`)
- Owner: core runtime / compiler maintainers
- Review when: the liveness queue, the free buffer, or the compiled-fused-op
  cache lifetime changes

## Symptom

Every MiniMax-H3 tiny 32x32 t2va run finished normally -- the mp4, the
soundtrack and the summary were all written -- and then aborted during
interpreter exit with glibc's `corrupted double-linked list` and status 134.
The address it named drifted from run to run (`~jit_cache_map`,
`~SFRLAllocator`, a PyAV/gnutls destructor, a bare `free`). That drift is the
signature of "some later free walked a chunk that had already been overwritten",
not of a fault at any one of those places.

## How it was found

**CUDA does initialize under AddressSanitizer.** The earlier attempt concluded
it does not ("visible device count is 0"), but that is ASAN's shadow-gap and
TLS interception breaking the driver probe, not the driver. Adding

```
ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:intercept_tls_get_addr=0:\
detect_odr_violation=0:halt_on_error=0:abort_on_error=0:\
log_path=/tmp/asan-report:print_stacktrace=1
```

brings the device up and the full pipeline runs under ASAN, with both the core
and the JIT operators instrumented (`cc_flags=-fsanitize=address
-fno-omit-frame-pointer -g -O1`, `JITTOR_NO_DEEPBIND=1`, `LD_PRELOAD` the gcc
`libasan.so`). The flash-attn extension cannot load under ASAN, so attention
falls back to the native path; the tiny shapes are small enough for it.

ASAN then reports the first bad access. The write:

```
heap-use-after-free, WRITE of size 8
  std::vector<pair<Node*, void (Node::*)()>>::emplace_back
  Node::release_both_liveness            node.cc:302
  VarPtr::~VarPtr                        var.h:161
  VarRelayGroup::~VarRelayGroup          var_relay.cc:33
  FusedOpContext::~FusedOpContext        fused_op.h:27
  jit_cache_map<shared_ptr<FusedOpContext>>::~jit_cache_map
  __run_exit_handlers
freed by:
  ~vector<pair<Node*, void (Node::*)()>>  (the same liveness_queue)
  __run_exit_handlers
```

## Root cause

Static destruction order. `liveness_queue` and `free_buffer` in `node.cc` were
namespace-scope vectors, so their destructors run during `__run_exit_handlers`,
in an order relative to the compiled-fused-op cache's destructor that only the
linker decides. That cache's destructor tears down `VarRelayGroup`s, whose
`VarPtr`s release liveness -- and `release_both_liveness` appends to the queue
that has already been destroyed. The append writes a pair into freed storage,
which is what damages the allocator's chunk metadata; glibc detects it much
later, on an unrelated free, as `corrupted double-linked list`.

The six earlier teardown fixes did not address this because none of them changed
the lifetime of the containers the exit-time teardown writes into.

## Fix

Keep the two containers alive for the life of the process, the pattern already
used by `graph_mutation_mutex` (`node.cc`) and the cleanup-callback list
(`utils/log.cc`):

- `free_buffer` is a function returning a leaked heap vector (`node.h` declares
  it as a function, `SetupFreeBuffer` calls it);
- `liveness_queue` is a reference bound to a leaked heap vector.

Nothing is reclaimed at exit, so no exit-time write can land in freed storage.

## Verification

- `run-tiny-parity.sh` (32x32, 124 frames, VAE float32 under autocast) run four
  times: **exit 0, no `corrupted` in every run**; before the fix the same
  command exited 134 with the message. Each run still writes the real 44325-byte
  mp4 decoded by the VAEs.
- The ASAN run no longer reports any exit-time access; the only report left is
  the compile-time relay use-after-free below, which is unrelated to teardown.
- `tests/core/test_fused_op.py`, `test_new_fused_op.py`, `test_fuser.py`,
  `test_var_relay_members.py`, `test_node.py` pass on CPU. (`tests/core/test_core.py`
  and `test_auto_flush.py` have pre-existing failures in this environment:
  `test_var_holder` asserts an old `'int' object has no attribute 'shape'`
  message where the code now says `ndim`, and several tests need optional
  libraries; each passes or fails independently of this change.)

## Separate finding (not fixed here)

With the exit-time fault gone, ASAN's next report is a different, compile-time
use-after-free in the relay path -- `add_relay_group` drops a relayed
subgraph's fused-op inputs and `release_inputs` frees them while
`FusedOp::vars` and the relay op's own Var members still point at them. It is
recorded as `KI-COMPILER-005` in `agent/manuals/known-issues.md`.

Raw ASAN reports and run logs live under `$JITTOR_LAB_ROOT/_state/h3/` and
`/tmp/asan-report.*` and are unversioned.
