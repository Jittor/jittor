from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "python/jittor/src/pyjt/py_ring_buffer.h"
SOURCE = ROOT / "python/jittor/src/pyjt/py_ring_buffer.cc"


def test_pop_remains_unbounded_and_pop_for_is_explicit():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "PyObject* pop();" in header
    assert "PyObject* pop_for(uint64 timeout_ms);" in header
    assert "PyMultiprocessRingBuffer::pop_for" in source
    assert "wait_pop_for(offset + 1, timeout_ms)" in source
    assert "std::rethrow_exception(wait_exception);" in source
