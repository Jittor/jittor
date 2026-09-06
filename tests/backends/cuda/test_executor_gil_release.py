"""The executor hands the GIL over while it waits on the device.

`Executor::run_sync`'s phase 7 waits for every device the batch launched on.
Until this test existed it did so holding the GIL, so a diffusion-UNet step
spent 9.78 of its 14.21 ms with the whole interpreter stopped and the CPU
idle -- a hard serialization point for anything that needs a background
Python thread (a scheduler, a data loader, a server's request loop).

The claim is functional, not a speed measurement: *another Python thread runs
during the wait*.  Two ways for such a test to pass while the GIL is in fact
never released, both of which it has to rule out:

  * the counting loop yields the GIL by itself.  Anything that touches I/O
    does -- `time.sleep`, `print`, a socket, a file.  `Counter.spin` below is
    `self.n += 1` and nothing else, so the only way it advances is if the
    interpreter is actually free.
  * CPython hands the GIL to another thread every
    `sys.getswitchinterval()` regardless of what the main thread is doing,
    which covers the *boundary* of the C call even when the call itself holds
    the GIL throughout.  Measured before the fix, at the 5 ms default: 72081
    increments on a 15.6 ms wait -- 26% of full speed, which reads exactly
    like success.  Ruled out by shrinking the interval to 0.5 ms and making
    the device wait ~1 s, three orders of magnitude larger.

With both traps closed the two states are unmistakable.  Measured on one
RTX 4090, three trials each:

    before   wait 1148-1366 ms   counter +7427..7850     0.03-0.04% of full speed
    after    wait  829-1290 ms   counter +14.7M..23.7M    101-105% of full speed

so the threshold below (half of full speed) sits ~1500x above the pre-fix
number and ~2x below the post-fix one.
"""
import sys
import threading
import time

import pytest


# ~275 GFLOP each on this shape, and k is large enough that the outputs stay
# small: 80 of them queue about a second of device work in 1.8 GB.
STEPS, M, K = 80, 2048, 32768


class Counter:
    """A counter advanced by pure bytecode.  No sleep, no I/O, nothing that
    would hand the GIL over on its own."""

    def __init__(self):
        self.n = 0
        self.go = True

    def spin(self):
        while self.go:
            self.n += 1


def _queue_device_work(jt):
    a = jt.rand(M, K)
    b = jt.rand(K, M)
    return [jt.matmul(a, b) for _ in range(STEPS)]


def test_other_python_thread_runs_during_device_wait():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")

    with jt.flag_scope(use_cuda=1):
        # Compile every kernel these shapes need before measuring: the
        # parallel op compiler already drops the GIL while it waits for its
        # workers, and a compile inside the window would be indistinguishable
        # from the thing under test.
        for _ in range(2):
            queued = _queue_device_work(jt)
            jt.sync_all(True)
            del queued

        interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        counter = Counter()
        thread = threading.Thread(target=counter.spin, daemon=True)
        thread.start()
        try:
            time.sleep(0.2)  # let the thread reach its loop

            # Full speed: the main thread off the GIL entirely.
            start = counter.n
            time.sleep(0.5)
            full_rate = (counter.n - start) / 0.5
            assert full_rate > 0, "counting thread never ran"

            ratios = []
            for _ in range(3):
                queued = _queue_device_work(jt)
                # Launch every kernel without waiting, so that the call
                # measured below contains the device wait and nothing else --
                # no graph traversal, no allocation, no compilation.
                jt.sync_all(False)
                start, t0 = counter.n, time.perf_counter()
                jt.sync_all(True)
                elapsed = time.perf_counter() - t0
                del queued
                # A wait that short means the device queue had already
                # drained and the trial measured nothing.
                assert elapsed > 0.1, f"no device wait to measure: {elapsed}s"
                ratios.append(((counter.n - start) / elapsed) / full_rate)
        finally:
            counter.go = False
            thread.join()
            sys.setswitchinterval(interval)

    assert min(ratios) > 0.5, (
        "the counting thread was starved during the device wait, so the GIL "
        f"was held across it: rates {ratios} of full speed"
    )


def test_concurrent_cuda_batches_stay_correct():
    """The control for the release above, on the device that actually waits.

    Handing the GIL over during phase 7 lets another Python thread reach the
    executor, which `run_sync`'s contract forbids -- hence the entry lock. The
    CPU side of this is `tests/core/test_executor_entry_lock.py`; this is the
    same claim where the release is real.
    """
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")

    threads_n, rounds, side = 6, 20, 256
    barrier = threading.Barrier(threads_n)
    failures = []

    with jt.flag_scope(use_cuda=1):
        (jt.ones((side, side)) * 1.0).sum().item()  # compile before forking

        def worker(tid):
            base = float(tid + 1)
            expected = float(base * 2) * side * side
            try:
                for round_index in range(rounds):
                    x = jt.ones((side, side)) * base
                    total = (x + base).sum()
                    barrier.wait(timeout=120)
                    got = float(total.item())
                    if got != expected:
                        failures.append(
                            f"thread {tid} round {round_index}: "
                            f"{got} != {expected}")
                        barrier.abort()
                        return
            except threading.BrokenBarrierError:
                return
            except BaseException as error:  # noqa: BLE001
                failures.append(f"thread {tid}: {type(error).__name__}: {error}")
                barrier.abort()

        workers = [threading.Thread(target=worker, args=(tid,))
                   for tid in range(threads_n)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=300)

    alive = [thread.name for thread in workers if thread.is_alive()]
    assert not alive, f"threads never returned (deadlock?): {alive}"
    assert not failures, "\n".join(failures)


def test_fetch_callback_still_runs_under_the_gil():
    """`event_queue.flush()` stays outside the released region.

    The flush runs fetch callbacks, and those are Python: a fetch callback
    invoked without the GIL corrupts the interpreter rather than failing
    cleanly. It sits immediately after the phase 7 wait, and the only thing
    keeping it out of the window is where the scope ends -- which is exactly
    the kind of boundary that a later edit moves by accident.

    The callback below allocates Python objects and touches a closure, so it
    cannot run on a thread without a valid thread state.
    """
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")

    seen = []
    with jt.flag_scope(use_cuda=1):
        for value in range(1, 5):
            jt.fetch(jt.ones((32, 32)) * float(value),
                     lambda array, value=value: seen.append(
                         (value, {"sum": float(array.sum())})))
        jt.sync_all(True)

    assert [value for value, _ in seen] == [1, 2, 3, 4], seen
    assert [payload["sum"] for _, payload in seen] == [
        1024.0, 2048.0, 3072.0, 4096.0], seen
