"""CPU execution probes for the dtype-aware Python kernel table."""

import pytest

from _helpers.python_dispatch import DISPATCH_PROBES


@pytest.mark.parametrize("probe", DISPATCH_PROBES, ids=lambda probe: probe.__name__)
def test_cpu_python_dispatch(probe):
    import jittor as jt

    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0):
        probe(jt, "cpu", -1)
