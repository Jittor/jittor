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


@pytest.fixture
def queued_callback():
    # Use the actual native FetchFunc conversion and EventQueue dispatch. Empty
    # payloads isolate callback reentry without a stream delay or D2H timing.
    return jt.compile_custom_op(r'''
        #pragma once
        #include "core/op.h"
        #include "ops/composite/fetch_op.h"
        namespace jittor {
        struct CompatMaterializationCallbackOp : Op {
            CompatMaterializationCallbackOp(FetchFunc&& callback);
            const char* name() const override { return "compat_materialization_callback"; }
        };
        }
    ''', r'''
        #include "compat_materialization_callback_op.h"
        #include "core/event_queue.h"
        namespace jittor {
        namespace {
        std::list<FetchResult> queued;
        void dispatch() {
            auto task = std::move(queued.front());
            queued.pop_front();
            task.call();
        }
        }
        CompatMaterializationCallbackOp::CompatMaterializationCallbackOp(FetchFunc&& callback) {
            VarPtr sentinel(NanoVector{0}, ns_int32);
            sentinel->set_stop_grad();
            sentinel->finish_pending_liveness();
            queued.emplace_back(FetchResult{std::move(callback), {}, {}});
            event_queue.push(&dispatch);
            forward(sentinel);
        }
        }
    ''', "compat_materialization_callback", warp=False)


@pytest.fixture
def inline_callback(tmp_path):
    header = tmp_path / "compat_materialization_inline_op.h"
    source = tmp_path / "compat_materialization_inline_op.cc"
    header.write_text(textwrap.dedent('''
        #pragma once
        #include "core/op.h"
        #include "ops/composite/fetch_op.h"
        namespace jittor {
        struct CompatMaterializationInlineOp : Op {
            FetchFunc callback;
            Var* output;
            CompatMaterializationInlineOp(FetchFunc&& callback);
            const char* name() const override { return "compat_materialization_inline"; }
            void run() override;
        };
        }
    '''))
    source.write_text(textwrap.dedent('''
        #include "compat_materialization_inline_op.h"
        #include <cuda_runtime.h>
        namespace jittor {
        CompatMaterializationInlineOp::CompatMaterializationInlineOp(FetchFunc&& cb)
            : callback(std::move(cb)) {
            set_flag(OpFlags::_cuda);
            output = create_output({1}, ns_float32);
        }
        void CompatMaterializationInlineOp::run() {
            CHECK(cudaMemsetAsync(output->mem_ptr, 0, sizeof(float), cudaStreamPerThread)
                == cudaSuccess);
            FetchResult task;
            callback.callback(&task);
        }
        }
    '''))
    includes = " ".join(f'-I"{path}"' for path in jt.compiler.cuda_include_dirs)
    module = jt.compile_custom_ops([str(header), str(source)], extra_flags=includes)
    return module.compat_materialization_inline


@pytest.mark.parametrize("outer_policy,inner_policy", [(0, 1), (1, 0)])
def test_nested_materialization_preserves_explicit_caller_policy(
        queued_callback, inline_callback, outer_policy, inner_policy):
    observed = []

    def body():
        with jt.flag_scope(keep_graph=inner_policy):
            queued_callback(lambda: observed.append(int(jt.flags.keep_graph)))
            materialize(torch.ones(1, device="cuda"))
        assert jt.flags.keep_graph == 2

    with jt.flag_scope(keep_graph=outer_policy):
        ready, = materialize(inline_callback(body))
        assert jt.flags.keep_graph == outer_policy
    assert observed == [inner_policy]
    assert jt.flags.keep_graph == 0
    np.testing.assert_array_equal(snapshot(ready), [0.])


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


def test_fetch_callback_new_graph_keeps_caller_policy(queued_callback):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync(True, False)
    requested = (x * 2).sum()
    delivered, foreign = [], []

    def callback():
        value = x.square().sum()
        foreign.append(value)
        jt.sync([value], True, False)
        delivered.append(int(jt.flags.keep_graph))

    queued_callback(callback)
    assert not delivered
    ready, = materialize(requested)
    assert delivered == [0]
    assert foreign[0].is_finished, "MATERIALIZER_CAPTURED_FOREIGN_FETCH_CALLBACK_WORK"
    np.testing.assert_allclose(snapshot(ready), 1.2, rtol=1e-6)
    gradient = jt.grad(foreign[0], x, retain_graph=True)
    np.testing.assert_allclose(snapshot(gradient), [.4, .8], rtol=1e-6)


def test_fetch_callback_preserves_explicit_keep_one(queued_callback):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    requested = x * 2
    delivered, foreign = [], []

    def callback():
        value = x * 3
        foreign.append(value)
        jt.sync([value], True, False)
        delivered.append(int(jt.flags.keep_graph))

    with jt.flag_scope(keep_graph=1):
        queued_callback(callback)
        materialize(requested)
        assert jt.flags.keep_graph == 1
        assert delivered == [1]
        assert not foreign[0].is_finished
    foreign[0]._release_kept()


def test_fetch_callback_nested_materializer_restores_policy(queued_callback):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    requested = x * 2
    policies, foreign = [], []

    def inner_callback():
        value = x * 4
        foreign.append(value)
        jt.sync([value], True, False)
        policies.append(int(jt.flags.keep_graph))

    def outer_callback():
        policies.append(int(jt.flags.keep_graph))
        inner_request = x * 3
        queued_callback(inner_callback)
        inner_ready, = materialize(inner_request)
        np.testing.assert_allclose(snapshot(inner_ready), [.6, 1.2], rtol=1e-6)
        policies.append(int(jt.flags.keep_graph))

    queued_callback(outer_callback)
    ready, = materialize(requested)
    assert policies == [0, 0, 0]
    assert jt.flags.keep_graph == 0 and foreign[0].is_finished
    np.testing.assert_allclose(snapshot(ready), [.4, .8], rtol=1e-6)


def test_fetch_callback_exception_restores_caller_policy(queued_callback):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    requested = x * 2
    policies = []

    def callback():
        policies.append(int(jt.flags.keep_graph))
        raise RuntimeError("MATERIALIZATION_CALLBACK_EXPECTED_ERROR")

    with jt.flag_scope(keep_graph=1):
        queued_callback(callback)
        with pytest.raises(RuntimeError, match="MATERIALIZATION_CALLBACK_EXPECTED_ERROR"):
            materialize(requested)
        assert jt.flags.keep_graph == 1
    assert policies == [1]
    requested._release_kept()


@pytest.mark.parametrize("with_view", [False, True])
def test_materialization_preserves_foreign_kept_storage(with_view):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync(True, False)
    foreign = (x.square() + 1) * 3
    with jt.flag_scope(keep_graph=1):
        jt.sync([foreign], True, False)
    address = foreign._mem_ptr_now
    assert address and not foreign.is_finished
    if with_view:
        # Keep the view as a distinct storage-sharing node, so the release
        # candidate's ring really contains the protected foreign member.
        with jt.flag_scope(no_fuse=1):
            value = foreign.view(1, 2)
    else:
        value = foreign
    ready, = materialize((value * 2).sum())
    assert foreign._mem_ptr_now == address, "MATERIALIZER_RELEASED_FOREIGN_KEEP1_STORAGE"
    if with_view:
        assert value._mem_ptr_now == address
    # Read via an independent destination: never migrate or reexecute foreign.
    destination = torch.empty_like(foreign)
    destination.sync(True, False)
    destination._copy_into(foreign, False)
    np.testing.assert_allclose(snapshot(destination), [3.12, 3.48], rtol=1e-6)
    np.testing.assert_allclose(snapshot(ready), 13.2, rtol=1e-6)
    foreign._release_kept()


def test_materialization_preserves_recorded_cuda_graph_buffers():
    assert jt.core.graph_capture_supported()
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    foreign = (x.square() + 1) * 3
    handle = 0
    try:
        with jt.flag_scope(keep_graph=1):
            jt.sync([foreign], True, False)
            assert jt.core.graph_capture_begin()
            try:
                jt.sync([foreign], False, False)
            finally:
                handle = jt.core.graph_capture_end()
        assert handle, "simple warmed CUDA arithmetic must produce a recorded graph"
        jt.core.graph_bind_leaves(handle, [foreign])
        address = foreign._mem_ptr_now
        launches = jt.core.graph_launch_count()
        jt.core.graph_launch(handle)
        jt.core.graph_wait()
        materialize((foreign * 2).sum())
        assert foreign._mem_ptr_now == address
        assert not jt.core.graph_leaves_moved(handle)
        updated = np.array([.3, .5], dtype=np.float32)
        x._write_inplace(updated)
        jt.core.graph_launch(handle)
        jt.core.graph_wait()
        destination = torch.empty_like(foreign)
        destination.sync(True, False)
        destination._copy_into(foreign, False)
        np.testing.assert_allclose(snapshot(destination), [3.27, 3.75], rtol=1e-6)
        assert jt.core.graph_launch_count() >= launches + 2
    finally:
        if handle:
            jt.core.graph_release(handle)
        foreign._release_kept()


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
