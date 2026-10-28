"""Saving an object executes its tensor dependencies, with no global roots."""

import io
import os
import textwrap

import jittor as jt
import numpy as np
import pytest
import torch

from _helpers.child_process import run_python_child
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def cuda_only():
    if os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0] != "cuda":
        pytest.skip("backend: CUDA serialization scope")
    assert torch.cuda.is_available()
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, auto_flush_ops=0), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_lazy_nested_snapshot_preserves_dtype_and_source_device(index, dtype):
    if index >= torch.cuda.device_count():
        pytest.skip("insufficient-devices: nondefault CUDA snapshot")
    device = torch.device(f"cuda:{index}")
    source = torch.tensor([7, 9], dtype=dtype, device=device)
    value = source * 3
    buffer = io.BytesIO()
    torch.save({"nested": [source, value, value], "step": 2}, buffer)
    assert source.device == value.device == device
    buffer.seek(0)
    restored = torch.load(buffer)
    assert restored["step"] == 2
    for tensor, expected in zip(restored["nested"], ([7, 9], [21, 27], [21, 27])):
        assert tensor.device == device and tensor.dtype == dtype
        np.testing.assert_array_equal(tensor.detach().cpu().numpy(), expected)
    assert source.device == value.device == device


def test_save_ignores_unrelated_failure_but_reports_real_dependencies():
    # Retain a failed provider in an isolated process, so cleanup of this test
    # cannot hide the bug or poison other tests' explicit sync_all calls.
    script = textwrap.dedent(r'''
        import io
        import jittor as jt
        import numpy as np
        import torch
        from jittor._runtime.fallback import forbid_backend_fallbacks
        jt.flags.use_cuda = 1
        jt.flags.auto_flush_ops = 0
        def expect_failure(operation):
            try:
                operation()
            except RuntimeError as error:
                assert 'JT_SAVE_DEPENDENCY_ERROR' in str(error), str(error)
            else:
                raise AssertionError('a requested tensor dependency did not fail')
        with forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            good = torch.tensor([7., 9.], device='cuda')
            good.sync(True, False)
            bad = jt.code((1,), 'float32', [], cuda_src=r"""
                fprintf(stderr, "JT_SAVE_FAILED_PROVIDER_RUN\n");
                fflush(stderr);
                USER_CHECK(false) << "JT_SAVE_DEPENDENCY_ERROR";
            """)
            expect_failure(lambda: bad.sync(True, False))
            metadata = io.BytesIO()
            torch.save({'step': 2, 'schedule': [.1, .05]}, metadata)
            metadata.seek(0)
            assert torch.load(metadata) == {'step': 2, 'schedule': [.1, .05]}
            lazy = good * 3
            buffer = io.BytesIO()
            torch.save({'nested': [good, lazy]}, buffer)
            buffer.seek(0)
            restored = torch.load(buffer)
            # Strong, object-scoped reads keep this assertion independent of
            # the separate implicit weak-root recovery contract.
            for tensor, expected in zip(restored['nested'], ([7., 9.], [21., 27.])):
                arrays = []
                jt.sync([tensor], False, False)
                jt.fetch(tensor, lambda array: arrays.append(np.array(array, copy=True)))
                jt.sync([], True, False)
                np.testing.assert_array_equal(arrays[0], expected)
            assert good.device.type == lazy.device.type == 'cuda'
            expect_failure(lambda: torch.save({'bad': bad}, io.BytesIO()))
            dependent = bad + 1
            expect_failure(lambda: torch.save({'dependent': dependent}, io.BytesIO()))
            expect_failure(lambda: jt.sync_all(True))
            assert not bad.is_finished
            assert jt.core.backend_fallback_count() == before
            peak = jt.core.device_memory_peak(0)
            assert peak > 0
            print('SAVE_OBJECT_SCOPE_OK', 'allocated_peak_bytes=' + str(peak), flush=True)
    ''')
    child = run_python_child(["-c", script], env={
        "JITTOR_TORCH_SHIM": "1",
        "JITTOR_HOME": os.environ["JITTOR_HOME"],
        "JITTOR_TORCH_PROJECT_ROOT": os.environ["JITTOR_TORCH_PROJECT_ROOT"],
    })
    assert child.returncode == 0, child.stdout[-3000:] + child.stderr[-3000:]
    assert "SAVE_OBJECT_SCOPE_OK" in child.stdout
    assert child.stderr.splitlines().count("JT_SAVE_FAILED_PROVIDER_RUN") == 4
    print(next(line for line in child.stdout.splitlines()
               if line.startswith("SAVE_OBJECT_SCOPE_OK")))
