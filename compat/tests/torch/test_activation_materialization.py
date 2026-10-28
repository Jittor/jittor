"""Device-ready storage boundaries for native activation rematerialization."""

import textwrap

import jittor as jt
import numpy as np
import pytest
from _helpers.checkpoint_cuda import accelerator, snapshot, curand_position as _curand_position
import torch


def materialize(*values):
    return jt.core._materialize_remat_outputs(list(values))



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


def test_owner_lease_keeps_only_constructed_graph(capfd):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync(True, False)
    foreign = x.square() + 1
    lease = jt.core._RematerializationLease()
    try:
        original = lazy_exp(foreign).sum()
        ready, = materialize(original)
        assert foreign.is_finished, "OWNER_CAPTURED_FOREIGN_PENDING_GRAPH"
        assert not original.is_finished and lease.pin_count() > 0
        count = lease.pin_count()
        materialize(original)
        assert lease.pin_count() == count, "OWNER_DUPLICATED_NATIVE_PINS"
        assert jt.flags.keep_graph == 0
        lease.close()
        lease.release()
        gradient = jt.grad(original, x)
        np.testing.assert_allclose(snapshot(gradient),
                                   2 * np.array([.2, .4]) * np.exp([1.04, 1.16]), rtol=1e-5)
        np.testing.assert_allclose(snapshot(ready), np.exp([1.04, 1.16]).sum(), rtol=1e-6)
        assert "JT_MATERIALIZE_PRODUCER" in capfd.readouterr().err
    finally:
        lease.close()
        lease.release()


def test_owner_native_activation_freed_and_recomputed_with_live_lease():
    x = torch.tensor([[.2, .4], [.1, .3]], device="cuda", requires_grad=True)
    w = torch.tensor([[2., 1.], [3., 4.]], device="cuda", requires_grad=True)
    jt.sync([x, w], True, False)
    lease = jt.core._RematerializationLease()
    jt.core.step_trace_start(3)
    try:
        activation = x @ w
        activation_id = int(activation.id)
        original = activation.square().sum()
        ready, = materialize(original)
        cursor = len(jt.core.step_trace_memory())
        assert lease.pin_count() > 0
        assert activation._mem_ptr_now == 0, "OWNER_RETAINED_FORWARD_ACTIVATION"
        lease.close()
        lease.release()
        dx, dw = jt.grad(original, [x, w])
        jt.sync([dx, dw], True, False)
        memory = jt.core.step_trace_memory()
        rows = [(i, memory[i:i+11]) for i in range(0, len(memory), 11)]
        allocations = [(i, row) for i, row in rows
                       if row[1] == 0 and row[3] > 0 and row[7] == activation_id]
        before = [(i, row) for i, row in allocations if i < cursor]
        after = [(i, row) for i, row in allocations if i >= cursor]
        assert before and after, "OWNER_NO_ORIGINAL_ACTIVATION_REALLOCATION"
        ops = jt.core.step_trace_ops()
        strings = jt.core.step_trace_strings()
        names = {ops[i]: strings[ops[i+3]] for i in range(0, len(ops), 12)}
        assert all("matmul" in names.get(row[6], "") for _, row in before + after)
        first_i, first = before[0]
        assert any(first_i < i < after[0][0] and row[3] < 0
                   and row[4:6] == first[4:6] for i, row in rows), "OWNER_NO_FORWARD_FREE"
        xx = np.array([[.2, .4], [.1, .3]])
        ww = np.array([[2., 1.], [3., 4.]])
        np.testing.assert_allclose(snapshot(dx), 2 * (xx @ ww) @ ww.T, rtol=1e-5)
        np.testing.assert_allclose(snapshot(dw), 2 * xx.T @ (xx @ ww), rtol=1e-5)
        np.testing.assert_allclose(snapshot(ready), np.square(xx @ ww).sum(), rtol=1e-5)
        peak = jt.core.device_pool_stats(0)[2]
        assert peak > 0
        print("OWNER_ORIGINAL_CUDA_ACTIVATION_FREE_RECOMPUTE",
              "native_allocated_peak_bytes=" + str(peak), "fallback=0", flush=True)
    finally:
        jt.core.step_trace_stop()
        lease.close()
        lease.release()


@pytest.mark.parametrize("policy", [1, 2])
def test_owner_lease_preserves_explicit_retention(policy):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync(True, False)
    lease = jt.core._RematerializationLease()
    try:
        original = (x.square() + 1) * 3
        with jt.flag_scope(keep_graph=policy):
            materialize(original)
            assert jt.flags.keep_graph == policy
        address = original._mem_ptr_now
        assert address and not original.is_finished
        assert lease.pin_count() == 0
        lease.close()
        lease.release()
        jt.sync([original], True, False)
        assert not original.is_finished and original._mem_ptr_now == address
        destination = torch.empty_like(original)
        destination.sync(True, False)
        destination._copy_into(original, False)
        np.testing.assert_allclose(snapshot(destination), [3.12, 3.48], rtol=1e-6)
    finally:
        lease.close()
        lease.release()
        original._release_kept()


def test_owner_lease_rejects_wrong_thread_and_lifo():
    import threading

    outer = jt.core._RematerializationLease()
    inner = jt.core._RematerializationLease()
    errors = []
    foreign = []
    try:
        with pytest.raises(RuntimeError, match="LIFO"):
            outer.close()

        def other_thread():
            try:
                inner.close()
            except RuntimeError as error:
                errors.append(str(error))
            value = torch.ones(2, device="cuda") + 2
            jt.sync([value], True, False)
            foreign.append(value)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert len(errors) == 1 and "wrong thread" in errors[0]
        assert foreign[0].is_finished
        np.testing.assert_array_equal(snapshot(foreign[0]), [3., 3.])
    finally:
        inner.close()
        inner.release()
        outer.close()
        outer.release()


def test_owner_callback_masks_construction_and_allows_nested_lease(queued_callback):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    lease = jt.core._RematerializationLease()
    foreign, policies = [], []
    try:
        requested = x * 2

        def callback():
            policies.append(int(jt.flags.keep_graph))
            value = x.square() + 1
            jt.sync([value], True, False)
            foreign.append(value)
            child = jt.core._RematerializationLease(False)
            try:
                nested = x * 3
                materialize(nested)
                assert not nested.is_finished and child.pin_count() > 0
            finally:
                child.close()
                child.release()

        queued_callback(callback)
        materialize(requested)
        assert policies == [0]
        count = lease.pin_count()
        lease.close()
        lease.release()
        assert count > 0 and foreign[0].is_finished
        np.testing.assert_allclose(snapshot(foreign[0]), [1.04, 1.16], rtol=1e-6)
    finally:
        lease.close()
        lease.release()


def test_owner_active_close_rejected_before_mutation(inline_callback):
    lease = jt.core._RematerializationLease()
    observed = []
    try:
        def callback():
            with pytest.raises(RuntimeError, match="active execution"):
                lease.close()
            observed.append(int(jt.flags.keep_graph))
        original = inline_callback(callback)
        materialize(original)
        assert observed == [2] and lease.pin_count() > 0
    finally:
        lease.close()
        lease.release()


def test_owner_error_before_outputs_drops_only_private_bits():
    import gc

    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    foreign = x * 4
    with jt.flag_scope(keep_graph=1):
        jt.sync([foreign], True, False)
    address = foreign._mem_ptr_now
    saved = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        def function():
            lease = jt.core._RematerializationLease()
            try:
                internal = x.square() + 1
                saved.append(internal)
                assert internal.sum().item() == pytest.approx(2.2)
                assert lease.pin_count() > 0
                raise RuntimeError("OWNER_BEFORE_OUTPUT_PRIMARY_ERROR")
            finally:
                lease.close()
                lease.release()
                # Explicit finally cleanup must preserve the primary error.
        with pytest.raises(RuntimeError, match="OWNER_BEFORE_OUTPUT_PRIMARY_ERROR"):
            function()
        assert jt.flags.keep_graph == 0
        jt.sync([saved[0]], True, False)
        assert saved[0].is_finished
        np.testing.assert_allclose(snapshot(saved[0]), [1.04, 1.16], rtol=1e-6)
        assert foreign._mem_ptr_now == address and not foreign.is_finished
        foreign._release_kept()
    finally:
        if enabled:
            gc.enable()


def test_owner_active_destructor_defers_private_cleanup(inline_callback):
    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    handoff = [jt.core._RematerializationLease()]
    internal = x.square() + 1
    materialize(internal)
    assert handoff[0].pin_count() > 0

    policies = []

    def callback():
        policies.append(int(jt.flags.keep_graph))
        if handoff:
            dropped = handoff.pop()
            del dropped
            assert jt.flags.keep_graph == 2
        else:
            # An explicit root reexecutes its provider after private cleanup.
            assert jt.flags.keep_graph == 0

    original = inline_callback(callback)
    materialize(original)
    assert jt.flags.keep_graph == 0
    # This succeeds only after entry-level deferred cleanup removed the bits;
    # a leaked private kept bit leaves the original graph unfinished.
    jt.sync([internal, original], True, False)
    assert internal.is_finished and original.is_finished
    assert policies == [2, 0]
    np.testing.assert_allclose(snapshot(internal), [1.04, 1.16], rtol=1e-6)


@pytest.mark.parametrize("with_view", [False, True])
def test_owner_preserves_foreign_explicit_keep_one_storage(with_view):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    x.sync(True, False)
    foreign = (x.square() + 1) * 3
    with jt.flag_scope(keep_graph=1):
        jt.sync([foreign], True, False)
    address = foreign._mem_ptr_now
    lease = jt.core._RematerializationLease()
    try:
        if with_view:
            with jt.flag_scope(no_fuse=1):
                value = foreign.view(1, 2)
        else:
            value = foreign
        ready, = materialize((value * 2).sum())
        assert lease.pin_count() > 0
        assert foreign._mem_ptr_now == address and not foreign.is_finished
        lease.close()
        lease.release()
        assert foreign._mem_ptr_now == address and not foreign.is_finished
        destination = torch.empty_like(foreign)
        destination.sync(True, False)
        destination._copy_into(foreign, False)
        np.testing.assert_allclose(snapshot(destination), [3.12, 3.48], rtol=1e-6)
        np.testing.assert_allclose(snapshot(ready), 13.2, rtol=1e-6)
    finally:
        lease.close()
        lease.release()
        foreign._release_kept()


def test_owner_foreign_thread_destructor_leaves_no_recording_frame():
    import threading

    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    handoff = [jt.core._RematerializationLease()]
    original = x.square() + 1
    materialize(original)
    assert handoff[0].pin_count() > 0

    def discard():
        dropped = handoff.pop()
        del dropped

    thread = threading.Thread(target=discard)
    thread.start()
    thread.join()
    # Construction prunes the abandoned creator-thread frame. Neither this
    # fresh graph nor the discarded lease's old graph stays privately kept.
    fresh = x * 2
    jt.sync([original, fresh], True, False)
    assert original.is_finished and fresh.is_finished
    np.testing.assert_allclose(snapshot(original), [1.04, 1.16], rtol=1e-6)
    np.testing.assert_allclose(snapshot(fresh), [.4, .8], rtol=1e-6)


def test_owner_custom_backward_and_implicit_module_parameter_update():
    calls = {"forward": 0, "backward": 0}

    class Exponential(jt.Function):
        def execute(self, value):
            calls["forward"] += 1
            self.value = value
            return value.exp()

        def grad(self, incoming):
            calls["backward"] += 1
            return incoming * self.value.exp()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))

        def forward(self, value):
            return Exponential()(value * self.weight).square().sum()

    model = Model()
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    jt.sync([x, model.weight], True, False)
    before = snapshot(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    lease = jt.core._RematerializationLease()
    try:
        original = model(x)
        ready, = materialize(original)
        assert lease.pin_count() > 0
        lease.close()
        lease.release()
        original.backward()
        xx = np.array([.2, .4])
        expected = 2 * xx * np.exp(2 * xx * before)
        np.testing.assert_allclose(snapshot(model.weight.grad), expected, rtol=1e-5)
        np.testing.assert_allclose(snapshot(x.grad),
                                   2 * before * np.exp(2 * xx * before), rtol=1e-5)
        optimizer.step()
        np.testing.assert_allclose(snapshot(model.weight), before - .01 * expected, rtol=1e-5)
        assert calls == {"forward": 1, "backward": 1}
        np.testing.assert_allclose(snapshot(ready), np.exp(2 * xx * before).sum(), rtol=1e-5)
    finally:
        lease.close()
        lease.release()


@pytest.mark.parametrize("close_before_manual", [False, True])
@pytest.mark.parametrize("policy", [1, 2])
@pytest.mark.parametrize("foreign_thread", [False, True])
def test_owner_transfers_later_explicit_retention(close_before_manual, policy, foreign_thread):
    from concurrent.futures import ThreadPoolExecutor

    x = torch.tensor([.2, .4], device="cuda")
    x.sync(True, False)
    lease = jt.core._RematerializationLease()
    value = (x.square() + 1) * 3
    try:
        materialize(value)
        assert lease.pin_count() > 0
        if close_before_manual:
            lease.close()
        def retain_manually():
            with jt.flag_scope(keep_graph=policy):
                jt.sync([value], True, False)

        if foreign_thread:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(retain_manually).result()
        else:
            retain_manually()
        lease.close()
        lease.release()
        jt.sync([value], True, False)
        assert not value.is_finished, "PRIVATE_RELEASE_CLEARED_EXPLICIT_LATER_MANUAL_RETENTION"
        destination = torch.empty_like(value)
        destination.sync(True, False)
        destination._copy_into(value, False)
        np.testing.assert_allclose(snapshot(destination), [3.12, 3.48], rtol=1e-6)
    finally:
        lease.close()
        lease.release()
        value._release_kept()


def _physical_snapshot(value):
    # Copy already-completed bytes without asking a kept producer to run again.
    destination = torch.empty_like(value)
    destination.sync(True, False)
    destination._copy_into(value, False)
    return snapshot(destination)



@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["uniform", "normal"])
@pytest.mark.parametrize("count", [0, 1, 5, 6])
def test_owner_default_cuda_rng_replays_actual_execution(dtype, kind, count):
    torch.manual_seed(1234)
    foreign = jt.rand((7,))
    lease = jt.core._RematerializationLease(True)
    try:
        value = jt.random((count,), dtype, kind)
        # Issue order differs from execution order; capture must be late.
        jt.sync([foreign], True, False)
        start = _curand_position()
        ready, = materialize(value)
        first = snapshot(ready)
        cost = count if kind == "uniform" else (count + 1) // 2
        assert _curand_position() == (start[0], start[1] + cost)
        lease.close()
        outside = jt.rand((11,))
        jt.sync([outside], True, False)
        after_outside = _curand_position()
        jt.sync([value], True, False)
        np.testing.assert_array_equal(_physical_snapshot(value), first)
        assert _curand_position() == after_outside
        assert jt.core.device_pool_stats(0)[2] > 0
    finally:
        lease.close()
        lease.release()


def test_owner_cuda_rng_false_and_uncaptured_keep_legacy_stream():
    torch.manual_seed(1234)
    for captured in [False, True]:
        lease = jt.core._RematerializationLease(False) if captured else None
        value = jt.rand((16,))
        try:
            with jt.flag_scope(keep_graph=2 if lease is None else 0):
                jt.sync([value], True, False)
            first = _physical_snapshot(value)
            initial = _curand_position()
            if lease is not None:
                lease.close()
            jt.sync([value], True, False)
            second = _physical_snapshot(value)
            assert not np.array_equal(first, second)
            assert _curand_position() == (initial[0], initial[1] + 16)
        finally:
            if lease is not None:
                lease.close()
                lease.release()
            else:
                value._release_kept()


def test_owner_default_dropout_gradient_preserves_outside_rng():
    torch.manual_seed(1234)
    x = torch.tensor([.2, .4, .6, .8], device="cuda", requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor([2., 3., 4., 5.], device="cuda"))
    jt.sync([x, weight], True, False)
    lease = jt.core._RematerializationLease(True)
    try:
        activation = torch.nn.functional.dropout(x * weight, p=.5, training=True)
        original = activation.sum()
        forward, ready = materialize(activation, original)
        values = snapshot(forward)
        xx = np.array([.2, .4, .6, .8])
        ww = np.array([2., 3., 4., 5.])
        scaled_mask = values / (xx * ww)
        np.testing.assert_allclose(scaled_mask, np.rint(scaled_mask), atol=1e-6)
        lease.close()
        outside = jt.rand((11,))
        jt.sync([outside], True, False)
        position = _curand_position()
        lease.release()
        dx, dw = jt.grad(original, [x, weight])
        np.testing.assert_allclose(snapshot(dx), ww * scaled_mask, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(snapshot(dw), xx * scaled_mask, rtol=1e-5, atol=1e-6)
        assert _curand_position() == position
        np.testing.assert_allclose(snapshot(ready), values.sum(), rtol=1e-5)
    finally:
        lease.close()
        lease.release()


def test_owner_cpu_rng_state_replay_feeds_real_cuda_gradient():
    torch.manual_seed(1234)
    x = torch.ones(16, device="cuda", requires_grad=True)
    x.sync(True, False)
    lease = jt.core._RematerializationLease(True)
    try:
        host = torch.rand(16, device="cpu")
        gpu = host.to("cuda")
        original = (x * gpu).sum()
        gpu_ready, ready = materialize(gpu, original)
        expected = snapshot(gpu_ready)
        lease.close()
        outside = torch.rand(7, device="cpu").to("cuda")
        jt.sync([outside], True, False)
        position = jt.core.rng_state("cpu")
        jt.sync([host], True, False)
        np.testing.assert_array_equal(_physical_snapshot(host), expected)
        assert jt.core.rng_state("cpu") == position
        lease.release()
        dx = jt.grad(original, x)
        np.testing.assert_allclose(snapshot(dx), expected, rtol=1e-6)
        np.testing.assert_allclose(snapshot(ready), expected.sum(), rtol=1e-6)
        assert jt.core.rng_state("cpu") == position
        assert jt.core.device_pool_stats(0)[2] > 0
    finally:
        lease.close()
        lease.release()


def test_owner_inner_false_overrides_parent_rng_preservation():
    torch.manual_seed(1234)
    outer = jt.core._RematerializationLease(True)
    child = None
    try:
        first_value = jt.rand((16,))
        child = jt.core._RematerializationLease(False)
        second_value = jt.rand((16,))
        first_ready, second_ready = materialize(first_value, second_value)
        first = snapshot(first_ready)
        second = snapshot(second_ready)
        child.close()
        outer.close()
        outside = jt.rand((11,))
        jt.sync([outside], True, False)
        position = _curand_position()
        jt.sync([first_value, second_value], True, False)
        np.testing.assert_array_equal(_physical_snapshot(first_value), first)
        assert not np.array_equal(_physical_snapshot(second_value), second)
        assert _curand_position() == (position[0], position[1] + 16)
    finally:
        if child is not None:
            child.close()
            child.release()
        outer.close()
        outer.release()


def test_owner_explicit_cuda_generator_keeps_its_issue_order_contract():
    torch.manual_seed(1234)
    generator = torch.Generator(device="cuda").manual_seed(19)
    position = _curand_position()
    lease = jt.core._RematerializationLease(True)
    try:
        value = torch.rand(16, device="cuda", generator=generator)
        issued = snapshot(generator.get_state())
        ready, = materialize(value)
        first = snapshot(ready)
        lease.close()
        jt.sync([value], True, False)
        np.testing.assert_array_equal(_physical_snapshot(value), first)
        np.testing.assert_array_equal(snapshot(generator.get_state()), issued)
        assert _curand_position() == position
    finally:
        lease.close()
        lease.release()
