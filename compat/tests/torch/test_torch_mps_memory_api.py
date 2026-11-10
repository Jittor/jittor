"""``torch.mps`` memory queries on a build without an MPS backend.

Native torch always publishes ``torch.mps.current_allocated_memory`` and
``torch.mps.driver_allocated_memory``; calling either on a non-MPS build raises
RuntimeError. DeepSpeed uses the first call as its MPS availability probe and
treats any returned value (even ``0``) as an available accelerator, so the
shim must raise rather than return a number.
"""

import pytest
import torch


_MEMORY_QUERIES = ("current_allocated_memory", "driver_allocated_memory")


def test_mps_is_reported_unavailable():
    assert torch.mps.is_available() is False


@pytest.mark.parametrize("name", _MEMORY_QUERIES)
def test_memory_query_is_published_and_raises(name):
    query = getattr(torch.mps, name)
    assert callable(query)
    with pytest.raises(RuntimeError, match="without MPS backend"):
        query()


@pytest.mark.parametrize("name", _MEMORY_QUERIES)
def test_memory_query_fails_closed_when_probed(name):
    """An availability probe must see an exception, never a byte count."""
    result = None
    try:
        result = getattr(torch.mps, name)()
    except RuntimeError:
        pass
    assert result is None
