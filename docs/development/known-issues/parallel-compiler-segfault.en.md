# Parallel compiler: a file-level deadlock and an operator-level segfault

- Status: **the file-level compile-pool deadlock is fixed** (2026-09-16); the segfault /
  state corruption of the parallel **operator** compiler is still unresolved under
  non-Jupyter workloads
- Last reviewed: 2026-09-16
- Baseline: `4b01857b`
- Owner: compiler / executor maintainers
- Workarounds:
  - state corruption (operator level): `jt.flags.use_parallel_op_compiler = 0`
  - deadlock (file level, before the fix): `DISABLE_MULTIPROCESSING=1`.
    **`use_parallel_op_compiler=0` does not help with this deadlock** -- it never goes through
    `run_cmds`; see the comparison data below
- Exit criteria: with parallel compilation enabled, a minimized stress test passes repeatedly
  with no cache corruption and no obvious performance regression (the deadlock item is already
  closed by the regression test and comparison data below)

This page is **the investigation record of a single problem**, not an inventory of known
issues. The ledger is
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md),
where this page corresponds to `KI-COMPILER-001`; the status, workaround and exit criteria on
both sides must agree, so changing one means changing the other. The remaining ledger entries
exist only as entries, without a corresponding long-form page.

The filename is a historical accident (early on only the segfault side was known). What is
recorded here is **two different failures** that share the name "parallel compilation" but
differ in mechanism, trigger conditions and fix:

| | File level | Operator level |
| --- | --- | --- |
| Code | `run_cmds()` in `python/jittor/build/utils/__init__.py` | `src/core/parallel_compiler.cc` |
| Parallel primitive | `multiprocessing.Pool` | `std::thread` / `std::async` |
| Switch | `DISABLE_MULTIPROCESSING=1` | `use_parallel_op_compiler` |
| Failure | **deadlock** (fixed) | segfault / heap corruption (unresolved) |

## Fixed: the deadlock of the file-level compile process pool

### Symptoms

A bare `python3 某个脚本.py` **hangs deterministically** when it genuinely needs to compile
(cold cache, changed `src/`, a new custom operator), not intermittently:

- the parent process is `S (sleeping)` and `/proc/<pid>/wchan` is `unix_stream_read_generic`;
- one child process is in `hrtimer_nanosleep`;
- **the number of `cc1plus` processes is 0** -- nothing is compiling at all, so "compilation is
  slow" is the wrong diagnosis;
- the same compilation work **succeeds** under `python3 -m pytest`.

### Minimal reproduction

A three-line script and a cold cache:

```bash
export JITTOR_HOME=/tmp/ki-compiler-001
export cache_name=ki-compiler-001
rm -rf "$JITTOR_HOME/.cache"
cat > /tmp/bare.py <<'PY'
import jittor as jt
print((jt.ones((3, 3)) * 2).sum().item())
PY
python3 /tmp/bare.py          # 修复前：永远不返回
```

Before the fix this script **never made progress in any of seven** runs limited to 1800s, 900s,
600s, 300s, 240s, 240s and 600s, and `cc1plus` stayed at 0. Two controls let the same cold
cache go through: change the entry point to `python3 -m pytest` (whose `__main__` has a
`__spec__`), or add an `if __name__ == "__main__":` guard to the script (so the script body
does not execute when it is re-run). That difference is exactly the entry to the mechanism.

### Mechanism

Three things combine, and all three are required:

1. **`run_cmds()` builds the process pool while holding `jittor.lock`.** It is reached from
   `import jittor` via `build_core()`, and the body of `jittor/__init__.py` runs inside
   `with _lock.lock_scope():` (`__init__.py:103`). This process **holds that file lock**.

2. **Python 3.14 changed the default `multiprocessing` start method on Linux from `fork` to
   `forkserver`.** On this machine `multiprocessing.get_start_method()` returns `forkserver`.

3. **`forkserver` re-executes `__main__`.** `ForkServer._preload_modules` is always
   `['__main__']`; when `__main__.__spec__` is `None` (that is, the `python3 脚本.py` style
   of invocation), `spawn.get_preparation_data()` takes the `init_main_from_path` branch, so the
   first thing the fork server does after starting is
   `spawn.import_main_path()` -> `runpy.run_path(脚本, run_name='__mp_main__')` -- **running
   the user's script again, verbatim**.

That closes the loop: the re-run script reaches its own `import jittor` and polls in
`lock.py:_acquire()` for the lock the parent holds; the parent sits in
`forkserver.connect_to_new_process()` waiting for the fork server to hand back a worker pid,
while the fork server never reaches its `accept()`.

```
父进程   持有 jittor.lock ──等待──> fork server 交回 worker pid
                                        │
fork server  重跑脚本 → import jittor ──等待──> jittor.lock
```

Neither side moves, and the compiler is never started even once.

### Evidence

The fork server's command line literally names the file to re-run:

```
python3 -c 'import sys; from multiprocessing.forkserver import main;
  main(14, 15, ['__main__'], ..., **{..., 'main_path': '/tmp/bare.py', ...})'
```

In the same hang, the `python3 -m pytest` fork server command line has **no `main_path`** --
because `pytest/__main__.py` has a `__spec__`, so it takes the `init_main_from_name` branch.

The two stacks printed by `faulthandler` close the loop (fork server):

```
forkserver.py:230 main -> spawn.py:307 import_main_path
  -> spawn.py:297 _fixup_main_from_path -> runpy.run_path
  -> bare.py:1 <module> -> jittor/__init__.py:103 <module>
  -> build/utils/lock.py:269 lock_scope.__enter__
  -> build/utils/lock.py:232 Lock.lock -> build/utils/lock.py:176 _acquire
```

and the parent:

```
bare.py:1 <module> -> jittor/__init__.py:107 -> build/compiler.py:1402 build_core
  -> build/compiler.py:102 compile_backend_sources
  -> build/utils/__init__.py run_cmds -> pool.py __init__ -> _repopulate_pool
  -> popen_forkserver.py:51 _launch -> forkserver.py:106 connect_to_new_process
  -> connection.py:995 answer_challenge -> connection.py:226 recv_bytes
```

`/proc/locks` shows who holds the lock -- only the parent; the fork server merely opened the
same file:

```
29: FLOCK  ADVISORY  WRITE 812441 00:6a:56864199665 0 EOF
```

`hrtimer_nanosleep` corresponds to the `time.sleep(0.05 / 0.5)` polling in `_acquire()`.

**One red herring**: a third process is present at the scene, `python3 <脚本>`, `pipe_wait`,
whose parent is the main process. It is not part of the deadlock; it is the **pre-forked
symbolization helper** in `src/utils/tracer.cc`, which by design blocks forever in
`read(request_fd, ...)`. Do not count it when attributing the failure.

### The fix

`run_cmds()` wraps the whole process-pool lifetime (creation **and** use) in
`_main_module_not_reexecuted()`: it temporarily gives `__main__` a stand-in `__spec__` whose
only job is to answer `.name`, so `get_preparation_data()` takes the `init_main_from_name`
branch, the fork server gets no `main_path`, and each worker's
`_fixup_main_from_name('__main__')` returns immediately as well.

Everything the compile pool has to send to a worker (`do_compile`, `pool_initializer`) is an
attribute of `jittor.build.utils`; nothing lives in `__main__`, so the child loses nothing.

That code **already existed**, merely confined to Windows by `if os.name == 'nt'`, with the
comment "a hack way to by pass windows multiprocess spawn init_main_from_path". Once Python 3.14
made `forkserver` the Linux default, the same workaround became mandatory on every platform. The
scope also widened from "wrapping only the `Pool()` construction" to "wrapping all of
`run_cmds`", so that workers added later by `_maintain_pool` during task execution receive the
same preparation data.

Regression test: `tests/build/test_compile_pool_main_reexec.py`. It pins **the property that
actually breaks the loop** -- no `init_main_from_path` reaching the child -- rather than the
deadlock itself, which cannot be reproduced in-process.

### The global build lock: what it explains and what it does not

`jittor.lock` is a process-level flock, and Python and C++ share the same descriptor and the
same `_has_lock` flag (`Lock.bind_core` -> `set_lock_fd`). It appears at both ends of this
problem, so it is worth stating its scope clearly, to avoid counting into the mechanism things
that never happened.

**It really is half of the loop.** The process holds the lock while `run_cmds()` builds the
process pool -- that is item 1 above.

**But "the child inherited the descriptor and the held flag, so parent and child wait for each
other" is not what happened here.**

- The stuck child is the **fork server**, created by `util.spawnv_passfds()` via
  `close_fds=True` `posix_spawn`, passing only the listener / alive / authkey descriptors. It
  is a **fresh interpreter** that inherited neither `lock_fd` nor `_has_lock`. Measured, the
  `jittor.lock` descriptor it holds is **one it opened itself while re-running the script**,
  and its stack also sits in `_acquire()` -- it is **requesting** the lock, not inheriting it.
- `/proc/locks` contains exactly one FLOCK holder from start to finish, the parent.
- Conversely, **inheriting `_has_lock=1` never causes a deadlock**: the child would think it
  already holds the lock and skip requesting it. That is exactly what happened with the
  `fork` start method before Python 3.14, and exactly why this deadlock did not exist before
  3.14. (What it buys instead is a different risk: a compile child that thinks it holds the
  lock while it does not. This change does not touch that; it is recorded here only.)
- The `pipe_wait` child at the scene is the symbolization helper from `tracer.cc`, not a pool
  worker.

**"Bare python deadlocks while pytest succeeds" is also not a difference in lock state.** Both
enter `run_cmds` from the same `lock_scope()` in `import jittor`, with identical lock state. The
only difference is `__main__.__spec__`: whether the fork server command line has a `main_path`.
That has direct evidence (see the "Evidence" section above) and needs no inference.

**The lock's real effect on parallelism is between processes, not within one.** `_has_lock` is
an ordinary global `int`, not thread_local: `parallel_compile_all_ops` takes a `lock_guard` on
the main thread first (`parallel_compiler.cc:210`), and the `lock_guard`s that the 16 worker
threads then each construct in `OpCompiler::do_compile` (`op_compiler.cc:1361`) all see
`_has_lock != 0` and become no-ops. So **this lock does not serialize the 16 workers inside one
process**. What it serializes is **other processes sharing the same cache** -- and at the
granularity of an entire compilation batch. The 8 pytest-xdist workers sharing one cache land
exactly on that granularity, which is the explanation most worth verifying for "8 xdist workers
are only 7% faster than serial".

One **unfixed hazard** while we are here: `_has_lock` is written by the main thread and read by
the 16 compile threads with no synchronization, which is a real data race. It is currently
unexposed only because "the main thread always takes the outer `lock_guard` first"; if some
worker entered `OpCompiler::do_compile` while `_has_lock == 0`, the process-level flock it
acquires would be released by another worker's `lock_guard` destructor. This belongs to the
unresolved half of this page (the operator level) and is out of scope for this change.

### Directions tried and rejected

- **`jt.flags.use_parallel_op_compiler = 0`** (the workaround originally recorded in the
  ledger): ineffective. That flag only governs the `std::thread` path in
  `parallel_compiler.cc`; `run_cmds()` never looks at it. Comparison data below.
- **Making the module-level `if os.name=='nt' and _inheriting: DISABLE_MULTIPROCESSING=1`
  cross-platform**: not enough. The re-run script blocks on the lock at
  `jittor/__init__.py:103`, **before** `jittor.build.utils` is imported, so the guard never
  takes effect. Moreover, `_inheriting` is true for **every** compile worker inside
  `spawn._main()`, so making it cross-platform would also set `log_silent=1` on all workers and
  hide compilation diagnostics. It therefore stays Windows-only.
- **Switching to the `fork` start method**: it would work around the problem, but it is the
  wrong direction. The existing MPI escape hatch in the repository exists precisely because
  `fork()` is unsafe after MPI_Init; CPython changed the default precisely because `fork()` is
  unsafe in multithreaded processes. Going back would invite a harder-to-diagnose problem back
  in.
- **Adding timeouts / retries**: unacceptable. This is a closed resource loop; a timeout only
  turns "never moves" into "fails much later", and the compiler still never runs.

### Comparison numbers

Same machine, same script, each run starting from a cold cache after
`rm -rf $JITTOR_HOME/.cache`:

```
                     修复前                      修复后
trial 1     240s 到期，DONE=0，cc1plus=0      61s，跑完
trial 2     240s 到期，DONE=0，cc1plus=0      61s，跑完
```

240s was this round's limit; the same reproduction had previously also hung once each under
1800s, 900s and 600s limits, so across six runs **not one file was ever compiled**. After the
fix both runs take 61s, of which about 27s is the cold compilation of the 222 `jittor_core`
files.

Control runs (after the fix, same code):

```
python3 guarded.py            5.7s   （脚本带 if __name__ == "__main__" 保护）
python3 bare.py（热缓存）      1.7s
python3 -m pytest tests/core  37.3s
```


Which switch actually works (same cold cache, pre-fix code):

```
W1  use_parallel_op_compiler=0    rc=124（600s 到期），DONE=0   —— 无效
W2  DISABLE_MULTIPROCESSING=1     rc=0，336s，DONE=1           —— 有效
```

W1 hangs in exactly the same place as the default setting (both logs end with
`Create cache dir ... /checkpoints`), and a fork server with a `main_path` is present at the
scene as well. That is no coincidence: nothing in `run_cmds()`'s code reads
`use_parallel_op_compiler`; that flag only feeds `src/core/parallel_compiler.cc`. W2, by
contrast, replaces `run_cmds` with serial in-process compilation outright, and the loop
disappears -- at the cost of compiling 222 files serially, 336s against 61s in parallel.


## Unresolved: the segfault of the parallel operator compiler

Certain large-model and device-parity workloads can terminate with an allocator error or a
segfault when **several fused operations are compiled concurrently**. The failure is
timing-sensitive and may only surface **after** the compilation that corrupted the state --
so **the operator name printed last is not reliable attribution**.

The dataset, ACL and parity test paths already use serial compilation successfully. This is a
**containment measure, not evidence that "runtime operator execution is defective"**.

### Resolved Jupyter sub-problem

The earlier Jupyter reproduction had a separate, proven cause. The parallel operator compiler
uses `std::thread` and **does not fork compile workers**. But Jittor still installed a
process-level `SIGCHLD` handler inside ipykernel that made the host exit quickly whenever any
child process was killed by a signal. A minimal process containing only `import jittor` and an
unrelated child killed by `SIGKILL` reproduced an exit with `si_code=CLD_KILLED` and
`si_status=SIGKILL`.

The Jupyter kernel now keeps its existing SIGCHLD disposition (just as it already keeps
SIGINT), and Jittor keeps fault diagnostics for SIGILL and SIGBUS. The deterministic
host-ownership regression test and the targeted CPU/CUDA cold-operator probes pass.

But **a later full nbclient smoke still died twice under a transformer attention workload with
eight compile workers**, one of those times with `JT_NO_SIGNAL_HANDLER=1` set; the same smoke
passes under serial compilation. So until this larger problem is closed, **the offline notebook
gate stays serial**.

### Current hypothesis

The main hypothesis is: **a synchronization boundary is missing between the parallel
compilation workers and the process-level compile/cache locking**. A worker may observe lock
state established by the owning process and wrongly conclude that shared compiler or relay
state is protected; the subsequent concurrent modification then manifests as heap corruption.

For non-Jupyter workloads **this is still just a hypothesis**: the relay group ownership, the
exact shared mutable objects and the first illegal access have not yet been demonstrated with a
sanitizer trace. So **adding one big lock is not an acceptable fix** -- it could deadlock
nested compilations, erase the intended parallelism, or fail to protect multi-process cache
access at all.

### What the parallel operator compiler is worth

A dedicated probe (`use_cuda=0`, 50 distinct CPU fused kernels, an isolated `JITTOR_HOME`,
the `jit/` directory cleared before each measurement, two rounds):

```
          单进程编译 50 个核     两个进程各编译 50 个互不相同的核（同一缓存）
par=16          7s / 7s                       14s / 13s
par=0          43s / 42s                       88s / 86s
```

Two conclusions, opposite in direction, both worth reading:

- **Within a process: the parallel operator compiler is worth about 6x** (7s against 42s). So
  the hypothesis that "it brings no speedup and only risks remain with it on by default" **does
  not hold**, and this change does not alter the default of `use_parallel_op_compiler`.
- **Between processes: fully serialized.** Two processes compiling **disjoint** kernels take as
  long as the sum of the two (≈ 2x) rather than the maximum -- on a 384-core machine with a
  load of about 24. Both `par=16` and `par=0` show 2x, which means the serialization comes from
  the process-level build lock itself and is unrelated to batch granularity. **This is the
  explanation most worth verifying first for "8 pytest-xdist workers are only 7% faster than
  serial"**: the workers share one cache, and the build phase is queued by this lock.

In other words, to raise compilation throughput one should change **cross-process lock
granularity** (for example lock per artifact instead of per batch / per operator), not turn
`use_parallel_op_compiler` off by default -- the latter would throw away the in-process 6x for
nothing. Cross-process lock granularity is out of scope for this change.

### `45:32` against `3:39` is cache warmth, not this flag

The whole `tests/ops` suite, **restored from the same hot-cache snapshot with `rm -rf` before
every run**:

```
                          耗时    通过/失败/跳过        并发编译器数(峰值/均值)
修复后 par=16 (1)         183s    740 / 4 / 91                18 / 2.9
修复后 par=0  (1)         219s    740 / 4 / 91                19 / 3.6
修复后 par=16 (2)         216s    740 / 4 / 91                17 / 3.4
修复后 par=0  (2)         204s    740 / 4 / 91                 4 / 2.1
修复前 par=16             201s    740 / 4 / 91                 3 / 1.3
```

The `FAILED` sets of the five runs are **identical entry by entry** (the same 4 pre-existing
`test_matmul` failures), and the pass and skip counts are identical too -- this change causes no
regression. The spread between 183s and 219s is entirely noise: **with the same cache state,
`use_parallel_op_compiler` has no measurable effect on the whole `tests/ops` suite**.

By contrast, running the same tests from a **cold JIT cache** (core compiled, `jit/` empty)
spent 47 minutes to reach 74% -- because every CUDA kernel has to go through nvcc again. That is
the real source of `45:32` against `3:39`: the first run warmed the cache, and the second run's
3-plus minutes had nothing to do with parallel versus serial. **Running once with
parallelism first and then once serially measures cache warmup.**


## Reproduction protocol

Use a dedicated state directory, and preserve the exact commit, compiler, device and
environment:

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/parallel-compiler/repro/jittor-home"
export cache_name=parallel-compiler-repro
python -m pytest -v tests/backends/parity/test_device_parity.py
```

Run once with the default parallel compilation settings, then once with:

```python
import jittor as jt

jt.flags.use_parallel_op_compiler = 0
```

**Do not run the two variants concurrently, and do not let them share a cache.** A useful
reproduction records: whether the cache was cold, the worker count, the last completed
compilation, the signal/backtrace, and whether the repeated serial run was clean.

**And: both variants must start from the same cache snapshot.** Running parallel first and then
serial and crediting the second run's time to serial measures cache warmup, not compilation
strategy -- that is exactly how the `45:32` against `3:39` pair came about. Take a `cp -a`
snapshot of the cache directory first, and `rm -rf` and restore before each variant.

## Investigation plan

1. Minimize the workload while preserving the cold-cache failure;
2. Instrument compile-task creation, relay-group ownership, cache-lock acquisition and worker
   completion with stable identifiers;
3. Run the minimized native compilation path under AddressSanitizer or ThreadSanitizer;
4. **Locate the first illegal access or race before** changing synchronization;
5. Apply the narrowest ownership or locking fix, and add a deterministic stress regression test.

## Acceptance bar

A fix must demonstrate all of the following:

- repeated stress runs with cold and hot caches no longer crash;
- a run with a timeout shows no deadlock;
- two processes with separate caches and with a shared cache respectively do not corrupt
  artifacts;
- parallel compilation time shows **no obvious regression** against the recorded baseline;
- after re-enabling parallel compilation, the compiler, device-parity and representative model
  tests all pass;
- **the serial workaround and the ledger entry are removed in the same change**.

Until that evidence exists, callers who prefer deterministic validation may explicitly turn off
the parallel compiler, **and should report that choice in their results**.
