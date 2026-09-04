# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What the device-parity oracle cache must guarantee before it may be trusted.

``test_device_parity`` compares an accelerator against the CPU. The CPU half is
device-independent, so it is cached between runs (0.22). A cache in front of an
oracle is the one optimisation that can make a gate *lie* while staying green:
it answers about the code that produced the entry, not about the code under test.

The properties below are the price of having it, and each one is a way the cache
could go wrong without anything turning red:

* the CPU oracle is reproducible at all -- if it were not, freezing one draw
  would freeze one of several legitimate answers;
* an entry round-trips bit for bit, dtype included (a float32 oracle read back
  as float64 still passes a 2e-4 comparison, so it would hide the defect);
* different operators, dtypes, sample indices, kwargs and input *contents*
  get different keys;
* editing the implementation invalidates every entry;
* an entry whose recorded key material does not match is ignored, not used;
* a partial write is never a load target (writes go through ``os.replace``);
* a corrupt entry degrades to a recompute, never to a gate failure.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest tests/backends/parity/test_reference_cache.py
"""
import os

import numpy as np
import pytest

from _helpers import reference_cache


# ------------------------------------------------------------------ keys

def _material(op="add", dtype="float32", index=0, array=None, kwargs=None):
    if array is None:
        array = np.arange(6, dtype="float32").reshape(2, 3)
    return reference_cache.key_material(
        op, dtype, index, array, {"dim": 0} if kwargs is None else kwargs)


def test_the_same_question_produces_the_same_key():
    assert _material() == _material()


def test_array_content_is_part_of_the_key():
    other = np.arange(6, dtype="float32").reshape(2, 3)
    other[1, 2] = -1.0
    assert _material() != _material(array=other)


def test_array_dtype_and_shape_are_part_of_the_key():
    base = np.arange(6, dtype="float32").reshape(2, 3)
    assert _material() != _material(array=base.astype("float64"))
    assert _material() != _material(array=base.reshape(3, 2))



def test_every_component_separates_the_key():
    for overrides in ({"op": "sub"}, {"dtype": "int8"}, {"index": 1},
                      {"kwargs": {"dim": 1}}, {"kwargs": {}}):
        assert _material() != _material(**overrides), \
            "the key ignores %r" % overrides


def test_key_material_stays_small_for_a_large_input():
    """The material is stored inside the entry, so it must be bounded.

    Arrays enter the key as a content digest, not as a printed value: ``repr``
    of a big array truncates, which would make two different inputs share a key.
    """
    material = reference_cache.key_material(
        "op", "float32", 0, np.zeros((512, 512), dtype="float32"))
    assert len(material) < 400
    assert len(reference_cache.ReferenceCache.digest(material)) == 40


def test_a_var_like_object_enters_the_key_by_value():
    class FakeVar:
        def numpy(self):
            return np.array([1.0, 2.0], dtype="float32")

    assert (reference_cache.value_digest(FakeVar())
            == reference_cache.value_digest(np.array([1.0, 2.0], "float32")))



# ---------------------------------------------------------- fingerprint

def _tree(root):
    os.makedirs(os.path.join(str(root), "src"))
    with open(os.path.join(str(root), "mod.py"), "w") as handle:
        handle.write("value = 1\n")
    with open(os.path.join(str(root), "src", "kernel.cc"), "w") as handle:
        handle.write("int kernel() { return 1; }\n")
    return str(root)


def test_fingerprint_is_stable_within_one_tree():
    assert reference_cache.source_fingerprint() == reference_cache.source_fingerprint()


def test_fingerprint_follows_content_not_mtime(tmp_path):
    root = _tree(tmp_path / "tree")
    first = reference_cache.source_fingerprint(root)
    os.utime(os.path.join(root, "mod.py"), (0, 0))
    assert reference_cache.source_fingerprint(root) == first, (
        "a checkout or rebase rewrites mtimes without changing a number; an "
        "mtime-keyed cache would discard every entry on every branch switch")
    with open(os.path.join(root, "mod.py"), "a") as handle:
        handle.write("\n# one more line\n")
    assert reference_cache.source_fingerprint(root) != first


def test_fingerprint_covers_cpp_as_well_as_python(tmp_path):
    root = _tree(tmp_path / "tree")
    first = reference_cache.source_fingerprint(root)
    with open(os.path.join(root, "src", "kernel.cc"), "a") as handle:
        handle.write("// changed\n")
    assert reference_cache.source_fingerprint(root) != first, (
        "a CPU kernel change moves the numbers the oracle produces")


def test_fingerprint_ignores_build_output(tmp_path):
    root = _tree(tmp_path / "tree")
    first = reference_cache.source_fingerprint(root)
    with open(os.path.join(root, "leftover.so"), "wb") as handle:
        handle.write(b"\0binary")
    assert reference_cache.source_fingerprint(root) == first


def test_key_material_carries_the_fingerprint():
    assert reference_cache.source_fingerprint() in _material()


# ------------------------------------------------------------ round trip

def _cache(tmp_path, name="unit"):
    return reference_cache.ReferenceCache(
        name, root=os.path.join(str(tmp_path), name))


def test_values_round_trip_bit_for_bit(tmp_path):
    cache = _cache(tmp_path)
    values = [
        np.array([1.25, -3.5], dtype="float32"),
        np.array([[1, 2], [3, 4]], dtype="int8"),
        np.array(7.0, dtype="float64"),            # 0-d: a full reduction
    ]
    material = _material()
    assert cache.store(material, values, {"forward_count": 1})
    loaded, extras = cache.load(material)
    assert extras == {"forward_count": 1}
    assert len(loaded) == len(values)
    for before, after in zip(values, loaded):
        assert before.dtype == after.dtype, (
            "dtype must survive: a float32 oracle read back as float64 still "
            "passes a 2e-4 comparison and would hide this")
        assert before.shape == after.shape
        assert before.tobytes() == after.tobytes(), (
            "the oracle must round-trip byte for byte, not approximately")


def test_a_miss_is_reported_as_a_miss(tmp_path):
    cache = _cache(tmp_path)
    assert cache.load(_material()) is None
    assert (cache.hits, cache.misses) == (0, 1)


def test_statistics_say_what_the_run_answered_from_where(tmp_path):
    cache = _cache(tmp_path)
    material = _material()
    cache.store(material, [np.zeros(2, "float32")])
    cache.load(material)
    cache.load(_material(index=1))
    assert (cache.hits, cache.misses, cache.writes) == (1, 1, 1)
    assert "1 reused" in cache.summary()
    assert "1 computed" in cache.summary()


def test_an_entry_with_foreign_key_material_is_not_used(tmp_path):
    """A digest collision, or a file that belongs to another question.

    Every entry re-states its own key material, so the reader can tell. It must
    recompute rather than answer with another operator's oracle.
    """
    cache = _cache(tmp_path)
    mine = _material(op="mine")
    theirs = _material(op="theirs")
    cache.store(theirs, [np.array([1.0], "float32")])
    source = cache._path(cache.digest(theirs))
    target = cache._path(cache.digest(mine))
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(source), str(target))
    assert cache.load(mine) is None
    assert cache.rejected == 1


def test_a_corrupt_entry_degrades_to_a_recompute(tmp_path):
    cache = _cache(tmp_path)
    material = _material()
    cache.store(material, [np.zeros(4, "float32")])
    with open(str(cache._path(cache.digest(material))), "r+b") as handle:
        handle.truncate(17)
    assert cache.load(material) is None, (
        "a corrupt entry is a miss, never a failure: an optimisation must not "
        "be able to fail a gate")
    assert cache.rejected == 1


def test_no_partial_file_is_left_where_a_reader_looks(tmp_path):
    """Atomicity by construction rather than by racing.

    Everything that appears under the final name got there through
    ``os.replace``; a partial write carries a dot-prefixed name that no load
    ever looks for (9.20, 9.22 were non-atomic writes).
    """
    cache = _cache(tmp_path)
    material = _material()
    cache.store(material, [np.zeros(4, "float32")])
    path = cache._path(cache.digest(material))
    assert path.is_file()
    assert [name for name in os.listdir(str(path.parent))
            if name.startswith(".") or "partial" in name] == []



def test_a_disabled_cache_is_usable_and_says_so(tmp_path):
    cache = _cache(tmp_path, "off")
    cache.active = False
    assert cache.load(_material()) is None
    assert cache.store(_material(), [np.zeros(1, "float32")]) is False
    assert "inactive" in cache.summary()


def test_the_environment_switch_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setenv(reference_cache.ENABLE_VARIABLE, "0")
    assert not reference_cache.enabled()
    assert not _cache(tmp_path, "off").active
    monkeypatch.setenv(reference_cache.ENABLE_VARIABLE, "1")
    assert reference_cache.enabled()
    assert _cache(tmp_path, "on").active


# -------------------------------------------------- the oracle's premise

#: Operators covering the shapes the payload has to survive: elementwise,
#: a 0-d full reduction, a normalisation, a matmul and a pooling backward.
_ORACLE_OPERATORS = ("add", "mul", "sum", "softmax", "matmul", "max_pool2d")


def _oracle_cases():
    from opinfo.database import op_db
    for op in op_db:
        if op.full_name in _ORACLE_OPERATORS:
            yield op


@pytest.mark.skipif(os.environ.get("JITTOR_TORCH_SHIM") != "1",
                    reason="the parity battery is a Torch-mode path "
                           "(tests/_helpers/process_modes.py)")
def test_the_cpu_oracle_is_reproducible():
    """The premise of caching the CPU half -- measured, not assumed.

    If the oracle were not a function of its inputs, an entry would freeze one
    of several legitimate answers and the accelerator would be compared against
    a draw. That is worse than being compared slowly.
    """
    from backends.parity import test_device_parity as parity

    checked = 0
    for op in _oracle_cases():
        samples = op.sample_inputs("cpu", "float32", requires_grad=True)
        for index, sample in enumerate(samples[:2]):
            first = parity._run(op, sample, use_cuda=0)
            second = parity._run(op, sample, use_cuda=0)
            _assert_identical(op.full_name, index, first, second)
            checked += 1
    assert checked > 0, "no operator from _ORACLE_OPERATORS is in op_db"


@pytest.mark.skipif(os.environ.get("JITTOR_TORCH_SHIM") != "1",
                    reason="the parity battery is a Torch-mode path "
                           "(tests/_helpers/process_modes.py)")
def test_a_cached_oracle_equals_a_freshly_computed_one(tmp_path):
    """End to end: what the cache hands back is what recomputing hands back."""
    from backends.parity import test_device_parity as parity

    cache = reference_cache.ReferenceCache(
        "unit-oracle", root=os.path.join(str(tmp_path), "oracle"))
    original = parity._REFERENCE_CACHE
    parity._REFERENCE_CACHE = cache
    try:
        checked = 0
        for op in _oracle_cases():
            samples = op.sample_inputs("cpu", "float32", requires_grad=True)
            for index, sample in enumerate(samples[:2]):
                computed = parity._cpu_oracle(op, sample, index, "float32")
                reused = parity._cpu_oracle(op, sample, index, "float32")
                _assert_identical(op.full_name, index, computed, reused)
                checked += 1
        assert checked > 0
        assert cache.hits == checked, cache.summary()
        assert cache.writes == checked, cache.summary()
    finally:
        parity._REFERENCE_CACHE = original



def _assert_identical(name, index, first, second):
    forward_first, grads_first = first
    forward_second, grads_second = second
    assert len(forward_first) == len(forward_second)
    for a, b in zip(forward_first, forward_second):
        assert a.dtype == b.dtype, "%s sample#%d" % (name, index)
        assert np.array_equal(a, b), (
            "%s sample#%d forward is not reproducible on CPU; caching it would "
            "freeze one draw of several" % (name, index))
    assert (grads_first is None) == (grads_second is None)
    for a, b in zip(grads_first or (), grads_second or ()):
        assert a.dtype == b.dtype, "%s sample#%d" % (name, index)
        assert np.array_equal(a, b), (
            "%s sample#%d backward is not reproducible on CPU" % (name, index))
