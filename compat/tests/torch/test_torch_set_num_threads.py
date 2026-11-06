"""torch.set_num_threads controls Jittor's CPU runtime pool."""

import jittor as jt
import pytest
import torch


def test_set_num_threads_updates_the_native_runtime():
    original = jt.core.get_num_threads()
    try:
        for threads in (1, 2, 3):
            assert torch.set_num_threads(threads) is None
            assert torch.get_num_threads() == threads
            assert jt.core.get_num_threads() == threads
    finally:
        torch.set_num_threads(original)


@pytest.mark.parametrize("threads", (0, -1))
def test_set_num_threads_rejects_nonpositive_values(threads):
    with pytest.raises(RuntimeError):
        torch.set_num_threads(threads)


@pytest.mark.parametrize("threads", (True, 1.5))
def test_set_num_threads_rejects_noninteger_values(threads):
    with pytest.raises(TypeError):
        torch.set_num_threads(threads)


def test_rejected_values_leave_the_executing_native_pool_unchanged():
    original = jt.core.get_num_threads()
    try:
        torch.set_num_threads(2)
        for value, error in (
            (0, RuntimeError),
            (-1, RuntimeError),
            (True, TypeError),
            (1.5, TypeError),
        ):
            with pytest.raises(error):
                torch.set_num_threads(value)
            # Observe a real OpenMP region, reusing this code kernel after each
            # rejected request. A cached Python getter alone cannot prove this.
            actual = (
                jt.code(
                    (1,),
                    "int32",
                    cpu_header="#include <omp.h>",
                    cpu_src=r"""
                #pragma omp parallel
                {
                    if (omp_get_thread_num() == 0)
                        @out(0) = omp_get_num_threads();
                }
            """,
                )
                .numpy()
                .item()
            )
            assert actual == 2
            assert torch.get_num_threads() == 2
            assert jt.core.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)
