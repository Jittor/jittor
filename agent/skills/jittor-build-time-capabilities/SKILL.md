---
name: jittor-build-time-capabilities
description: Why two agents on the same source tree can get a jittor with different capabilities (MPI, NCCL, CUDA), how to tell that your build is the crippled one before you conclude "this machine cannot do X", and how to rebuild with the capability on. Use whenever a capability probe says False, whenever a whole class of tests reports "skipped - no X found", or before reporting that something is unverifiable in this environment.
---

# `has_mpi` is a build-time constant, and PATH is not the same for everyone

`jt.compile_extern.has_mpi` (and the CUDA/NCCL equivalents) is decided **once,
when the tree is compiled**, by probing PATH for `mpicc`. The result is baked
into the compiled artifacts under `$JITTOR_HOME/.cache`. Changing PATH
afterwards does nothing: the ops that would call MPI were never built.

So two agents sharing one source checkout, with separate `JITTOR_HOME`s, can
hold jittor builds with **different capabilities** — and neither is told.
2026-09-03: the compat partition concluded "this machine has no MPI, 7.02
cannot be verified" while the distributed partition had been running 2-rank
MPI tests all along. `mpicc` lives in the conda env's `bin`, which is not on
the default PATH; whichever agent happened to compile without it got a jittor
with MPI compiled out.

## The symptom, and why it is easy to misread

    $ pytest tests/distributed/test_mpi.py -q
    12 skipped     # "skipped: no mpi found"

A wall of skips reads like "the machine does not have it". It actually means
"**your build** does not have it". The tests are honest; the build is the
variable.

## Check before you conclude

Never report "unverifiable in this environment" on the strength of `which` or a
skip count. Ask the build, then ask the machine, and compare:

    # what YOUR build decided, at compile time
    PYTHONPATH=<your worktree>/python python -c \
      "import jittor as jt; print('has_mpi =', jt.compile_extern.has_mpi); \
       print('mpicc  =', repr(jt.compile_extern.mpicc_path))"

    # what the MACHINE actually has (note the conda bin, not the default PATH)
    ls /home/zy/miniconda3/envs/jt311/bin/mpirun /home/zy/miniconda3/envs/jt311/bin/mpicc

`has_mpi = False` **and** the binaries exist ⇒ your build is the crippled one.
`mpicc_path` being the empty string is the tell: the probe found nothing.

Corollary worth knowing: `mpirun_path()` in `tests/_helpers/child_process.py`
is `mpicc_path.replace("mpicc", "mpirun")`, so an empty `mpicc_path` makes it
return `""` — the launcher is silently unusable rather than reported missing.

## Rebuild with the capability on

    rm -rf $JITTOR_HOME/.cache
    PATH=/home/zy/miniconda3/envs/jt311/bin:/usr/local/cuda/bin:$PATH \
    JITTOR_HOME=... TMPDIR=... PYTHONPATH=<worktree>/python \
      python -c "import jittor as jt; print(jt.compile_extern.has_mpi)"

Full recompile, ~1-2 minutes. Then **keep that PATH on every later command** —
not because the capability is re-probed, but because the launcher you are about
to run (`mpirun`) has to be findable.

Confirm it took, with a test that executes rather than skips:

    pytest tests/distributed/test_mpi.py -q        # expect "1 passed", not all-skipped

`1 passed, 6 skipped` is the healthy shape: the one that passes is the launcher
test that spawns `mpirun -np N`; the six that skip only run *inside* a rank.

## Running two ranks yourself

Use the sanctioned helper (`_helpers.child_process.run_mpi_python`) in tests —
bare `sys.executable` subprocesses are rejected by
`tests/structure/test_child_process_contract.py`. For an ad-hoc probe, pin
PYTHONPATH explicitly or every rank imports the *installed* jittor instead of
your worktree:

    PATH=/home/zy/miniconda3/envs/jt311/bin:$PATH JITTOR_HOME=... TMPDIR=... \
    mpirun -np 2 --allow-run-as-root \
      env PYTHONPATH=<worktree>/python JITTOR_TORCH_SHIM=1 python probe.py

To compare a value **across** ranks without a side channel, use a collective:
`float(jt.abs(v - v.mpi_all_reduce("mean")).max().item())` is 0 exactly when
every rank holds the same `v`. That is the cheapest "the ranks agree" assertion,
and it works for parameters and gradients alike.

## The general lesson

A capability decided by an unrecorded probe is a capability nobody can reason
about later. When you hit one, record **why** the probe answered as it did, not
just what it answered — "no mpicc on PATH at build time" is actionable, `False`
is not.

## Related

- `jittor-distributed-verification` — what to assert once you have two ranks.
- `codebase-wide-fix-as-rule` — the same theme: state the reason, not the result.
