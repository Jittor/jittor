"""Forward scheduling contracts needed by native activation rematerialization."""

import gc
import os

import jittor as jt
import numpy as np
import pytest
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def accelerator():
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
    if device != "cuda":
        pytest.skip("backend: CUDA scheduling contracts")
    if not torch.cuda.is_available():
        pytest.fail("CUDA scheduling acceptance requires a real CUDA device")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, auto_flush_ops=0, auto_graph_replay=0), forbid_backend_fallbacks():
        yield
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def snapshot(value):
    return value.detach().cpu().numpy().copy()


def lazy_exp(value):
    return jt.code(value.shape, value.dtype, [value], cuda_src=r'''
        __global__ static void producer(@ARGS_DEF) {
            @PRECALC
            int i = threadIdx.x;
            if (i < in0_shape0) @out(i) = expf(@in0(i));
        }
        fprintf(stderr, "JT_GRAD_ONLY_PRODUCER\n");
        fflush(stderr);
        producer<<<1, 32>>>(@ARGS);
    ''', cuda_grad_src=[r'''
        __global__ static void derivative(@ARGS_DEF) {
            @PRECALC
            int i = threadIdx.x;
            if (i < in0_shape0) @out(i) = expf(@in0(i)) * @dout(i);
        }
        derivative<<<1, 32>>>(@ARGS);
    '''])


class StaticStraightThrough(jt.Function):
    # Deliberate custom gradient: the forward ignores value, backward reads it.
    # This isolates the grouping edge; it is not a checkpoint implementation.
    def execute(self, value, metadata=None):
        self.value = value
        return jt.ones((2,))

    def grad(self, incoming):
        return incoming * 2 * self.value, None


def input_value():
    value = torch.tensor([0.2, 0.4], device="cuda", requires_grad=True)
    value.sync()
    return value


def producers(capfd):
    return capfd.readouterr().err.count("JT_GRAD_ONLY_PRODUCER")


def test_ignored_input_stays_lazy_through_ordinary_fetch(capfd):
    x = input_value()
    output = StaticStraightThrough()(lazy_exp(x), None)
    np.testing.assert_array_equal(snapshot(output), [1., 1.])
    assert producers(capfd) == 0
    gradient = jt.grad(output.sum(), x)
    np.testing.assert_allclose(snapshot(gradient), 2 * np.exp(2 * np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 1


def test_real_returned_input_path_still_executes(capfd):
    class RealForward(jt.Function):
        def execute(self, value):
            return value * 3

        def grad(self, incoming):
            return incoming * 3

    x = input_value()
    output = RealForward()(lazy_exp(x))
    np.testing.assert_allclose(snapshot(output), 3 * np.exp(np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 1
    gradient = jt.grad(output.sum(), x)
    np.testing.assert_allclose(snapshot(gradient), 3 * np.exp(np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 0


def test_independently_requested_provider_and_repeated_backward(capfd):
    x = input_value()
    value = lazy_exp(x)
    output = StaticStraightThrough()(value, None)
    direct = value.square().sum()
    jt.sync([output, direct], False, False)
    assert producers(capfd) == 1
    loss = output.sum() + direct
    for _ in range(2):
        gradient = jt.grad(loss, x, retain_graph=True)
        np.testing.assert_allclose(snapshot(gradient), 4 * np.exp(2 * np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 0


def test_ignored_host_gradient_input_needs_no_forward_storage(capfd):
    x = input_value()
    host = torch.tensor([2., 3.], device="cpu", requires_grad=True)
    output = StaticStraightThrough()(lazy_exp(x), host)
    np.testing.assert_array_equal(snapshot(output), [1., 1.])
    assert producers(capfd) == 0
    gradient = jt.grad(output.sum(), x)
    np.testing.assert_allclose(snapshot(gradient), 2 * np.exp(2 * np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 1


def test_python_resolved_dynamic_input_is_not_reexecuted(capfd):
    x = input_value()
    indices = jt.ops.where(lazy_exp(x) > 0)[0].float32().start_grad()
    # Returning an unknown-sized Var to Python explicitly resolves its shape.
    assert producers(capfd) == 1
    output = StaticStraightThrough()(indices, None)
    np.testing.assert_array_equal(snapshot(output), [1., 1.])
    assert producers(capfd) == 0
    gradient = jt.grad(output.sum(), indices)
    np.testing.assert_array_equal(snapshot(gradient), [0., 2.])
    assert producers(capfd) == 0


def test_discarded_output_releases_its_ignored_provider(capfd):
    x = input_value()
    value = lazy_exp(x)
    output = StaticStraightThrough()(value, None)
    del output, value
    gc.collect()
    jt.sync_all(True)
    assert producers(capfd) == 0


def test_explicit_saved_input_tape_fetch_still_executes(capfd):
    saved = []

    class ExposedInput(jt.Function):
        def execute(self, value):
            saved.append(value)
            return jt.ones((2,))

        def grad(self, incoming):
            return incoming * 3

    x = input_value()
    output = ExposedInput()(lazy_exp(x))
    np.testing.assert_allclose(snapshot(saved[0]), np.exp(np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 1
    np.testing.assert_array_equal(snapshot(output), [1., 1.])
    gradient = jt.grad(output.sum(), x)
    np.testing.assert_allclose(snapshot(gradient), 3 * np.exp(np.array([.2, .4])), rtol=1e-5)
    assert producers(capfd) == 0
    saved.clear()


def test_native_unresolved_dynamic_input_stays_lazy(capfd):
    # Compose where/cast/group in C++ before any unknown-sized Var crosses
    # the Python return boundary. This exercises the executor's dynamic-input
    # prewalk, rather than Python's intentional eager shape resolution.
    grouped = jt.compile_custom_op(r'''
        #pragma once
        #include "core/op.h"
        namespace jittor {
        struct CompatGradientOnlyDynamicProbeOp : Op {
            CompatGradientOnlyDynamicProbeOp(Var* value);
            const char* name() const override {
                return "compat_gradient_only_dynamic_probe";
            }
        };
        }
    ''', r'''
        #include "compat_gradient_only_dynamic_probe_op.h"
        #include "core/grad.h"
        #include "ops/op_register.h"
        #include "ops/composite/array_op.h"
        namespace jittor {
        CompatGradientOnlyDynamicProbeOp::CompatGradientOnlyDynamicProbeOp(Var* value) {
            auto where = op_constructor<vector<VarPtr>, Var*, NanoString>("where");
            auto cast = op_constructor<VarPtr, Var*, NanoString>("unary");
            auto array = op_constructor<VarPtr, ArrayArgs&&>("array");
            auto tape = op_constructor<VarPtr, Var*>("tape");
            auto indices = where(value, ns_int64);
            auto input = std::make_unique<VarHolder>(cast(indices[0], ns_float32));
            CHECK(input->var->num < 0) << "probe must retain an unresolved native extent";
            input->start_grad();
            auto taped_input = std::make_unique<VarHolder>(tape(input->var));
            float ones[2] = {1.f, 1.f};
            ArrayArgs args{ones, NanoVector{2}, ns_float32};
            auto real_output = std::make_unique<VarHolder>(array(std::move(args)));
            auto taped_output = std::make_unique<VarHolder>(tape(real_output->var));
            GradCallback callback;
            callback.func = [](int nout, Var** douts, int nin, VarPtr* dins) {
                if (nout && nin && douts[0]) dins[0] = VarPtr(douts[0]);
            };
            callback.deleter = [] {};
            tape_together({taped_input.get()}, {taped_output.get()}, std::move(callback));
            forward(taped_output->var);
        }
        }
    ''', "compat_gradient_only_dynamic_probe", warp=False)
    x = input_value()
    output = grouped(lazy_exp(x))
    np.testing.assert_array_equal(snapshot(output), [1., 1.])
    assert producers(capfd) == 0


def test_ignored_provider_error_is_reported_when_backward_reads_it():
    import textwrap
    from _helpers.child_process import run_python_child

    # Isolate the deliberately failed device graph from later tests.
    script = textwrap.dedent(r'''
        import gc
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        jt.flags.use_cuda = 1
        jt.flags.auto_flush_ops = 0
        jt.flags.auto_graph_replay = 0
        class DeferredRead(jt.Function):
            def execute(self, value):
                self.value = value
                return jt.ones((2,))
            def grad(self, incoming):
                return incoming * self.value
        with forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = jt.array([.2, .4])
            x.sync()
            value = jt.code(x.shape, x.dtype, [x],
                cuda_src='USER_CHECK(false) << "JT_GRADONLY_DEFERRED_ERROR";',
                cuda_grad_src=[r"""
                    __global__ static void derivative(@ARGS_DEF) {
                        @PRECALC
                        int i = threadIdx.x;
                        if (i < in0_shape0) @out(i) = @dout(i);
                    }
                    derivative<<<1, 32>>>(@ARGS);
                """])
            output = DeferredRead()(value)
            assert output.numpy().tolist() == [1., 1.]
            gradient = None
            try:
                gradient = jt.grad(output.sum(), x)
                gradient.sync()
            except RuntimeError as error:
                assert 'JT_GRADONLY_DEFERRED_ERROR' in str(error), str(error)
            else:
                raise AssertionError('backward silently discarded its required provider')
            assert jt.core.backend_fallback_count() == before
            del gradient, output, value, x
            gc.collect()
            print('DEFERRED_ERROR_OK', flush=True)
    ''')
    child = run_python_child(["-c", script], env={
        "JITTOR_TORCH_SHIM": "1",
        "JITTOR_HOME": os.environ["JITTOR_HOME"],
        "JITTOR_TORCH_PROJECT_ROOT": os.environ["JITTOR_TORCH_PROJECT_ROOT"],
    }, timeout=240)
    assert child.returncode == 0, child.stdout[-2000:] + child.stderr[-2000:]
    assert "DEFERRED_ERROR_OK" in child.stdout
