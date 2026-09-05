"""The Python routing query observes placement without executing a graph."""

import numpy as np

import jittor as jt


def test_cpu_dispatch_context_is_tuple_and_does_not_materialize():
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0):
        x = jt.array(np.arange(8, dtype=np.float32))
        y = x + 1
        before = (x.location(), y.location(), jt.number_of_lived_ops(),
                  jt.number_of_hold_vars())
        for _ in range(3):
            result = jt.core.dispatch_context([x, y])
            assert type(result) is tuple
            assert result == ("cpu", -1)
            assert jt.core.dispatch_context([]) == ("cpu", -1)
        assert (x.location(), y.location(), jt.number_of_lived_ops(),
                jt.number_of_hold_vars()) == before
        np.testing.assert_array_equal(y.numpy(), np.arange(8, dtype=np.float32) + 1)
