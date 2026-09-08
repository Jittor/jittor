"""Real-tensor probes shared by the CPU and CUDA Python dispatch tests."""

from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def registered_kernels(op, backend):
    from jittor._runtime.dispatch import register_kernel, unregister_kernel

    implementations = []

    def register(implementation, **options):
        register_kernel(op, backend, implementation, **options)
        implementations.append(implementation)
        return implementation

    try:
        yield register
    finally:
        for implementation in reversed(implementations):
            unregister_kernel(op, backend, implementation)


def assert_result(jt, value, expected, backend, device_id):
    from jittor._runtime.dispatch import dispatch_context

    value.sync()
    if backend == "cuda":
        assert value.location() == "device"
        assert value.device_id == device_id
    context = dispatch_context(value)
    assert (context.backend, context.device_id) == (backend, device_id)
    np.testing.assert_array_equal(value.numpy(), expected)


def pending_and_materialized(jt, backend, device_id):
    from jittor._runtime.dispatch import dispatch_context, select_kernel

    op = "test.python_dispatch.pending." + backend
    values = np.arange(8, dtype="float32")
    x = jt.array(values)
    pending = x + 1
    calls = []

    def kernel(value):
        calls.append(value)
        return value * 2

    with registered_kernels(op, backend) as register:
        register(kernel, dtypes={"float32"})
        before = (pending.location(), jt.introspection.counters.live_ops, jt.introspection.counters.exec_calls)
        for _ in range(3):
            context = dispatch_context(pending)
            assert (context.backend, context.device_id) == (backend, device_id)
            assert context.dtypes == ("float32",)
            assert select_kernel(op, pending) is kernel
        assert calls == []
        assert (pending.location(), jt.introspection.counters.live_ops, jt.introspection.counters.exec_calls) == before
        assert_result(jt, select_kernel(op, pending)(pending),
                      (values + 1) * 2, backend, device_id)
        assert calls == [pending]
        pending.sync()
        assert select_kernel(op, pending) is kernel
        assert_result(jt, kernel(pending), (values + 1) * 2, backend, device_id)


def all_tensor_dtypes(jt, backend, device_id):
    from jittor._runtime.dispatch import dispatch_context, select_kernel

    op = "test.python_dispatch.dtypes." + backend
    values = np.arange(4, dtype="float32")
    x = jt.array(values)
    integer = jt.array(np.arange(4, dtype="int32"))
    floating = jt.array(values + 1)

    def kernel(value, *, others):
        return value + others["right"][0].float32()

    with registered_kernels(op, backend) as register:
        register(kernel, dtypes={"float32"})
        mixed = {"right": [integer]}
        context = dispatch_context(x, others=mixed)
        assert context.dtypes == ("float32", "int32")
        assert select_kernel(op, x, others=mixed) is None
        same = {"right": [floating]}
        assert dispatch_context(x, others=same).dtypes == ("float32", "float32")
        assert select_kernel(op, x, others=same) is kernel
        assert_result(jt, kernel(x, others=same), values * 2 + 1, backend, device_id)

        register(lambda value, *, others: kernel(value, others=others),
                 dtypes={"float32", "int32"}, priority=-1)
        mixed_kernel = select_kernel(op, x, others=mixed)
        assert mixed_kernel is not None
        assert mixed_kernel is not kernel
        assert_result(jt, mixed_kernel(x, others=mixed), values * 2, backend, device_id)


def shape_priority_and_original_arguments(jt, backend, device_id):
    from jittor._runtime.dispatch import select_kernel

    op = "test.python_dispatch.shape." + backend
    values = np.arange(4, dtype="float32").reshape(2, 2)
    x = jt.array(values)
    y = jt.array(values + 1)
    seen = []

    def supports(left, right, *, scale):
        seen.append((left, right, scale))
        return left.dtype == right.dtype and tuple(left.shape) == (2, 2) and scale == 3

    def preferred(left, right, *, scale):
        return left + right * scale

    def generic(left, right, *, scale):
        return left - right * scale

    with registered_kernels(op, backend) as register:
        register(preferred, supports=supports, priority=10)
        register(generic, priority=0)
        assert select_kernel(op, x, y, scale=3) is preferred
        assert seen[-1][0] is x and seen[-1][1] is y and seen[-1][2] == 3
        assert_result(jt, preferred(x, y, scale=3), values + (values + 1) * 3,
                      backend, device_id)
        assert select_kernel(op, x, y, scale=2) is generic
        assert_result(jt, generic(x, y, scale=2), values - (values + 1) * 2,
                      backend, device_id)
        flat_x, flat_y = x.reshape((-1,)), y.reshape((-1,))
        assert select_kernel(op, flat_x, flat_y, scale=3) is generic
        integer = jt.array(np.arange(4, dtype="int32").reshape(2, 2))
        assert select_kernel(op, x, integer, scale=3) is generic


def optional_query_and_library_error(jt, backend, device_id):
    from jittor._runtime.dispatch import optional_kernel, select_kernel, unregister_kernel

    op = "test.python_dispatch.optional_error." + backend
    values = np.arange(4, dtype="float32")
    x = jt.array(values)
    fallback_calls = []
    failure = RuntimeError("selected library failed")

    def broken(value):
        raise failure

    def fallback(value):
        fallback_calls.append(value)
        return value + 1

    wrapped = optional_kernel(op, backend, priority=10)(broken)
    try:
        with registered_kernels(op, backend) as register:
            register(fallback, priority=0)
            assert select_kernel(op, x) is broken
            with pytest.raises(RuntimeError, match="selected library failed") as caught:
                wrapped(x)
            assert caught.value is failure
            assert fallback_calls == []
            assert_result(jt, x + 2, values + 2, backend, device_id)
    finally:
        unregister_kernel(op, backend, broken)


def predicate_error_is_not_a_miss(jt, backend, device_id):
    from jittor._runtime.dispatch import select_kernel

    op = "test.python_dispatch.predicate_error." + backend
    values = np.arange(4, dtype="float32")
    x = jt.array(values)
    failure = ValueError("capability query failed")

    def bad_predicate(value):
        raise failure

    def kernel(value):
        return value + 1

    with registered_kernels(op, backend) as register:
        register(kernel, supports=bad_predicate)
        with pytest.raises(ValueError, match="capability query failed") as caught:
            select_kernel(op, x)
        assert caught.value is failure
        assert_result(jt, x + 2, values + 2, backend, device_id)


def misses_do_not_hide_late_registration(jt, backend, device_id):
    from jittor._runtime.dispatch import optional_kernel, select_kernel, unregister_kernel

    op = "test.python_dispatch.late." + backend
    values = np.arange(4, dtype="float32")
    x = jt.array(values)
    assert select_kernel(op, x) is None

    def unavailable(value):
        raise AssertionError("unsupported implementation must not execute")

    def late(value):
        return value * 3

    wrapped = optional_kernel(op, backend, supports=lambda value: False)(unavailable)
    try:
        assert wrapped(x) is None
        assert select_kernel(op, x) is None
        with registered_kernels(op, backend) as register:
            register(late)
            assert select_kernel(op, x) is late
            assert_result(jt, wrapped(x), values * 3, backend, device_id)
        assert select_kernel(op, x) is None
        assert wrapped(x) is None
    finally:
        unregister_kernel(op, backend, unavailable)


def other_backend_is_not_selected(jt, backend, device_id):
    from jittor._runtime.dispatch import optional_kernel, select_kernel, unregister_kernel

    op = "test.python_dispatch.other_backend." + backend
    other_backend = "cuda" if backend == "cpu" else "cpu"
    values = np.arange(4, dtype="float32")
    x = jt.array(values)

    def wrong_backend(value):
        raise AssertionError("wrong backend implementation must not execute")

    wrapped = optional_kernel(op, other_backend)(wrong_backend)
    try:
        assert select_kernel(op, x) is None
        assert wrapped(x) is None
        assert_result(jt, x + 1, values + 1, backend, device_id)
    finally:
        unregister_kernel(op, other_backend, wrong_backend)


DISPATCH_PROBES = (
    pending_and_materialized,
    all_tensor_dtypes,
    shape_priority_and_original_arguments,
    optional_query_and_library_error,
    predicate_error_is_not_a_miss,
    misses_do_not_hide_late_registration,
    other_backend_is_not_selected,
)
