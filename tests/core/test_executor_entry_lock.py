"""Two Python threads may now be inside the executor's *call*, but not inside
its batch.

`Executor::run_sync` is documented as not reentrant: a batch keeps state on the
executor object across its phases (`last_is_cuda`, the allocator pointers, the
batch numbering) and a second batch interleaved with it would read and write
the first one's. Nothing enforced that -- the executor simply held the GIL from
entry to return, so for Python threads the GIL *was* the exclusion.

Releasing the GIL during the device wait removes that, so the exclusion had to
become an actual lock (`runtime/executor_entry.h`). This is the control for
that half of the change: without it the release would have traded a
serialization point for a data race, and a race is worse than a wait.

What it checks is what a race would break: results. Every thread's answer is
exactly representable in float32, so the assertion is equality, not tolerance
-- a torn batch shows up as a wrong number, an abort, or a hang, and all three
fail here.
"""
import threading

import numpy as np


THREADS = 8
ROUNDS = 25
SIDE = 256
CHAIN = 5


def _expected(base):
    # `base` <= THREADS and (base * (CHAIN+1)) * SIDE**2 < 2**24, so every
    # intermediate and the final sum are exact in float32.
    return float(base * (CHAIN + 1)) * SIDE * SIDE


def _worker(jt, tid, barrier, failures):
    base = float(tid + 1)
    expected = _expected(base)
    try:
        for round_index in range(ROUNDS):
            x = jt.ones((SIDE, SIDE)) * base
            for _ in range(CHAIN):
                x = x + base
            total = x.sum()
            # Line the threads up so they reach the executor together, which
            # is the situation under test. A thread that died on an earlier
            # round aborts the barrier instead of hanging the rest.
            barrier.wait(timeout=120)
            # Alternate the two entry paths: `item()` syncs and reads a
            # scalar, `numpy()` syncs and then migrates the whole tensor to
            # the host. Both take the entry lock, and the migration is where
            # the host copy releases the GIL.
            if tid % 2:
                got = float(total.item())
            else:
                got = float(np.asarray(total.numpy()).reshape(()))
            if got != expected:
                failures.append(
                    f"thread {tid} round {round_index}: {got} != {expected}")
                barrier.abort()
                return
    except threading.BrokenBarrierError:
        # Another thread failed and aborted the barrier; it reports.
        return
    except BaseException as error:  # noqa: BLE001 - the point is to report it
        failures.append(f"thread {tid}: {type(error).__name__}: {error}")
        barrier.abort()


def test_concurrent_run_sync_entries_stay_correct():
    import jittor as jt

    # Touch the executor once from the main thread so every kernel these
    # shapes need is compiled before the threads start: a first-time compile
    # inside a worker would make the test about the parallel op compiler.
    (jt.ones((SIDE, SIDE)) * 1.0).sum().item()

    barrier = threading.Barrier(THREADS)
    failures = []
    threads = [
        threading.Thread(target=_worker, args=(jt, tid, barrier, failures))
        for tid in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    alive = [thread.name for thread in threads if thread.is_alive()]
    assert not alive, f"threads never returned (deadlock?): {alive}"
    assert not failures, "\n".join(failures)
