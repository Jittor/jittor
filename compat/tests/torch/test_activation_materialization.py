"""Device-ready storage boundaries for native activation rematerialization."""

import os
import textwrap

import jittor as jt
import numpy as np
import pytest
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def accelerator():
    if os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0] != "cuda":
        pytest.skip("backend: CUDA materialization contracts")
    if not torch.cuda.is_available():
        pytest.fail("materialization acceptance requires a real CUDA device")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, auto_flush_ops=0, auto_graph_replay=0), forbid_backend_fallbacks():
        yield
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def materialize(*values):
    return jt.core._materialize_remat_outputs(list(values))


def snapshot(value):
    return value.detach().cpu().numpy().copy()


def lazy_exp(value):
    return jt.code(value.shape, value.dtype, [value], cuda_src=r'''
        __global__ static void forward_kernel(@ARGS_DEF) {
            @PRECALC
            int i = threadIdx.x;
            if (i < in0_shape0) @out(i) = expf(@in0(i));
        }
        fprintf(stderr, "JT_MATERIALIZE_PRODUCER\n"); fflush(stderr);
        forward_kernel<<<1, 32>>>(@ARGS);
    ''', cuda_grad_src=[r'''
        __global__ static void backward_kernel(@ARGS_DEF) {
            @PRECALC
            int i = threadIdx.x;
            if (i < in0_shape0) @out(i) = expf(@in0(i)) * @dout(i);
        }
        backward_kernel<<<1, 32>>>(@ARGS);
    '''])


def test_ready_read_preserves_original_graph_and_gradient(capfd):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync()
    original = lazy_exp(x)
    ready, = materialize(original)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 1
    assert not original.is_finished
    assert ready.is_finished and not ready.requires_grad
    np.testing.assert_allclose(snapshot(ready * 3), 3 * np.exp([.2, .4]), rtol=1e-5)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 0
    assert not original.is_finished
    gradient = jt.grad(original.square().sum(), x)
    np.testing.assert_allclose(snapshot(gradient), 2 * np.exp(2 * np.array([.2, .4])), rtol=1e-5)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 1


def test_shared_and_duplicate_outputs_execute_in_one_batch(capfd):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync()
    original = lazy_exp(x)
    a, b, duplicate = materialize(original, original * 2, original)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 1
    assert a.id != original.id
    assert a._mem_ptr_now == duplicate._mem_ptr_now
    np.testing.assert_allclose(snapshot(a), np.exp([.2, .4]), rtol=1e-5)
    np.testing.assert_allclose(snapshot(b), 2 * np.exp([.2, .4]), rtol=1e-5)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 0


def test_alias_shares_versions_without_parameter_trainability():
    parameter = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))
    parameter.sync()
    with torch.no_grad():
        ready, = materialize(parameter)
    assert parameter.requires_grad and not ready.requires_grad
    version = parameter._autograd_version()
    assert ready._autograd_version() == version
    with torch.no_grad():
        parameter.add_(1)
    assert ready._autograd_version() == version + 1
    with torch.no_grad():
        ready.add_(1)
    assert parameter._autograd_version() == version + 2
    assert parameter.requires_grad and not ready.requires_grad


def test_strided_offset_and_empty_storage_survive_source_drop():
    base = torch.arange(20, dtype=torch.float32, device="cuda").reshape(4, 5)
    base.sync()
    source = base[1:4].transpose(0, 1)
    empty = torch.empty((0, 3), device="cuda", dtype=torch.bfloat16)
    ready, zero = materialize(source, empty)
    assert ready.stride() == source.stride()
    assert ready.device == source.device and ready.dtype == source.dtype
    assert tuple(zero.shape) == (0, 3) and zero.dtype == torch.bfloat16
    del source, base, empty
    np.testing.assert_array_equal(snapshot(ready), np.arange(20).reshape(4, 5)[1:4].T)
    assert snapshot(zero).shape == (0, 3)


def test_empty_request_restores_scope():
    with jt.flag_scope(keep_graph=1):
        assert materialize() == []
        assert jt.flags.keep_graph == 1


def test_strong_materialization_does_not_submit_unrelated_graph(capfd):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync()
    unrelated = lazy_exp(x)
    requested = x * 2
    ready, = materialize(requested)
    assert capfd.readouterr().err.count("JT_MATERIALIZE_PRODUCER") == 0
    assert not unrelated.is_finished
    del unrelated
    np.testing.assert_allclose(snapshot(ready), [.4, .8], rtol=1e-5)


def test_failed_producer_returns_no_alias_and_restores_scope():
    from _helpers.child_process import run_python_child

    # Keep the deliberately failed graph out of the parent process teardown.
    script = textwrap.dedent(r'''
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        jt.flags.use_cuda = 1
        jt.flags.auto_flush_ops = 0
        jt.flags.auto_graph_replay = 0
        with forbid_backend_fallbacks():
            bad = jt.code((2,), 'float32', [], cuda_src=
                'USER_CHECK(false) << "JT_MATERIALIZE_EXPECTED_FAILURE";')
            try:
                jt.core._materialize_remat_outputs([bad])
            except RuntimeError as error:
                assert 'JT_MATERIALIZE_EXPECTED_FAILURE' in str(error)
            else:
                raise AssertionError('failed producer returned ready storage')
            assert jt.flags.keep_graph == 0
            assert jt.core.backend_fallback_count() == 0
            print('MATERIALIZATION_FAILURE_CONTRACT_PASS', flush=True)
    ''')
    result = run_python_child(["-c", script], timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MATERIALIZATION_FAILURE_CONTRACT_PASS" in result.stdout
