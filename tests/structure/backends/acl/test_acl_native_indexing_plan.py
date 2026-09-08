"""Execute real ACL copy plans on host bytes and compare with NumPy indexing."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
HEADER = ROOT / "backends/acl/include/aclops/native_indexing_op_acl.h"
HARNESS = r'''
#include "native_indexing_op_acl.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace jittor::acl_indexing;

std::vector<size_t> shape() {
    size_t rank;
    std::cin >> rank;
    std::vector<size_t> result(rank);
    for (auto& dimension : result) std::cin >> dimension;
    return result;
}

std::vector<unsigned char> buffer() {
    size_t count;
    std::cin >> count;
    std::vector<unsigned char> result(count);
    for (auto& byte : result) {
        unsigned int value;
        std::cin >> value;
        byte = static_cast<unsigned char>(value);
    }
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    std::string mode;
    std::cin >> mode;
    if (mode == "intervals") {
        Selection selection;
        CopyPlan plan;
        require(make_selection({2, 3}, {}, 8, selection).empty(), "full selection");
        require(make_get_plan(selection, {2, 3}, plan).empty(), "full get plan");
        require(plan.shape.empty() && plan.block_bytes == 48, "collapse contiguous suffix");
        require(identical_mapping(plan, 4096, 4096), "identity mapping");
        require(!identical_mapping(plan, 4096, 4104), "shifted mapping is not identity");
        require(overlaps(4096, 0, 16, 4096, 8, 24), "overlapping intervals");
        require(!overlaps(4096, 0, 16, 4096, 16, 24), "adjacent intervals");
        require(!overlaps(4096, 0, 16, 8192, 0, 16), "disjoint allocations");
        const uintptr_t top = std::numeric_limits<uintptr_t>::max();
        require(overlaps(top, 0, 8, 4096, 0, 8), "overflow is conservatively overlapping");
        CopyPlan overflow_plan = plan;
        overflow_plan.source_offset = 8;
        require(!identical_mapping(overflow_plan, top, 4096), "overflow cannot prove identity");
        SliceSpec every_other;
        every_other.kind = SliceKind::Range;
        every_other.start = 0;
        every_other.stop = 6;
        every_other.step = 2;
        require(make_selection({6}, {every_other}, 8, selection).empty(), "strided selection");
        require(make_get_plan(selection, {3}, plan).empty(), "strided get plan");
        require(!identical_mapping(plan, 4096, 4096), "strides must participate in identity");
        require(make_set_plan(selection, {}, plan).empty(), "broadcast scalar plan");
        require(!identical_mapping(plan, 4096, 4096), "broadcast is not identity");
        std::cout << "intervals-ok\n";
        return 0;
    }

    const auto input_shape = shape();
    size_t slice_count, element_bytes;
    std::cin >> element_bytes >> slice_count;
    std::vector<SliceSpec> slices(slice_count);
    for (auto& slice : slices) {
        int kind;
        std::cin >> kind >> slice.start >> slice.stop >> slice.step;
        slice.kind = static_cast<SliceKind>(kind);
    }
    const auto other_shape = shape();
    Selection selection;
    auto error = make_selection(input_shape, slices, element_bytes, selection);
    CopyPlan plan;
    if (error.empty()) {
        error = mode == "get" ? make_get_plan(selection, other_shape, plan)
                              : make_set_plan(selection, other_shape, plan);
    }
    if (!error.empty()) {
        std::cout << "error " << error << '\n';
        return 0;
    }

    auto original = buffer();
    require(original.size() == selection.storage_bytes, "input byte count");
    auto source = mode == "get" ? original : buffer();
    std::vector<unsigned char> target = mode == "get"
        ? std::vector<unsigned char>(selection.elements * element_bytes, 0xa5) : original;
    size_t calls = 0;
    for_each_copy(plan, [&](size_t from, size_t to, size_t bytes) {
        require(from <= source.size() && bytes <= source.size() - from, "source copy bounds");
        require(to <= target.size() && bytes <= target.size() - to, "target copy bounds");
        std::memcpy(target.data() + to, source.data() + from, bytes);
        ++calls;
    });
    require(std::cin.good(), "harness input must be complete");
    std::cout << "ok " << calls << ' ' << target.size();
    for (auto byte : target) std::cout << ' ' << static_cast<unsigned int>(byte);
    std::cout << '\n';
}
'''


@pytest.fixture(scope="module")
def indexing_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("acl-native-indexing-host")
    source = directory / "plan.cc"
    executable = directory / "plan"
    source.write_text(HARNESS, encoding="utf-8")
    compiled = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", "-O1", "-Wall", "-Wextra",
         "-I", str(HEADER.parent), str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def _shape_tokens(shape):
    return [len(shape), *shape]


def _slices(shape, index):
    index = index if isinstance(index, tuple) else (index,)
    explicit_axes = sum(item is not None and item is not Ellipsis for item in index)
    axis = 0
    specs = []
    for item in index:
        if item is None:
            specs.append((3, 0, 0, 1))
        elif item is Ellipsis:
            specs.append((4, 0, 0, 1))
            axis += len(shape) - explicit_axes
        elif isinstance(item, slice):
            start, stop, step = item.indices(shape[axis])
            specs.append((2, start, stop, step))
            axis += 1
        else:
            specs.append((1, int(item), 0, 1))
            axis += 1
    return specs


def _byte_tokens(value):
    raw = value.tobytes(order="C")
    return [len(raw), *raw]


def _run(indexing_plan, mode, shape, slices, width, other_shape, arrays=()):
    tokens = [mode, *_shape_tokens(shape), width, len(slices)]
    for spec in slices:
        tokens.extend(spec)
    tokens.extend(_shape_tokens(other_shape))
    for array in arrays:
        tokens.extend(_byte_tokens(array))
    result = subprocess.run(
        [str(indexing_plan)], input=" ".join(map(str, tokens)) + "\n",
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _assert_numpy(indexing_plan, shape, index, value_shape=None, dtype="int64"):
    original = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    selected = np.asarray(original[index])
    arrays = [original]
    if value_shape is None:
        expected = selected
        mode, other_shape = "get", expected.shape
    else:
        value = (np.arange(int(np.prod(value_shape)), dtype=dtype) + 17).reshape(value_shape)
        expected = original.copy()
        expected[index] = value
        arrays.append(value)
        mode, other_shape = "set", value.shape
    output = _run(indexing_plan, mode, shape, _slices(shape, index),
                  original.itemsize, other_shape, arrays)
    assert output.startswith("ok "), output
    fields = output.split()
    calls, byte_count = int(fields[1]), int(fields[2])
    raw = bytes(map(int, fields[3:]))
    assert len(raw) == byte_count == expected.nbytes
    actual = np.frombuffer(raw, dtype=original.dtype).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    assert (calls == 0) == (selected.size == 0)


@pytest.mark.parametrize("shape,index", [
    ((3, 4), (1,)),
    ((3, 4), (-1,)),
    ((3, 4), (-3, slice(1, None, 2))),
    ((3, 4, 2), (slice(None, None, 2), 1)),
    ((3, 4), (None, slice(None), Ellipsis)),
    ((2, 3), (slice(None), None, Ellipsis)),
    ((2, 3, 4), (Ellipsis, -1)),
    ((2, 3, 4), (1, Ellipsis, 1)),
    ((2, 3, 4), (None, Ellipsis, slice(1, 4, 2), None)),
    ((3, 4), (slice(3, 3), Ellipsis)),
    ((2, 3), (slice(2, 2), slice(1, 3))),
    ((0, 4), (Ellipsis, None)),
    ((), ()),
    ((), (None, Ellipsis)),
    ((1,), (0,)),
])
def test_get_plan_matches_numpy(indexing_plan, shape, index):
    _assert_numpy(indexing_plan, shape, index)


@pytest.mark.parametrize("shape,index,value_shape", [
    ((3, 4), (slice(None), slice(1, 4, 2)), ()),
    ((3, 4), (slice(None), slice(None)), (4,)),
    ((3, 4), (slice(None), slice(None)), (3, 1)),
    ((2, 3, 4), (1, Ellipsis), (1, 4)),
    ((2, 3, 4), (Ellipsis,), (1, 4)),
    ((2, 3, 1, 1), (Ellipsis,), (1, 3, 1, 1)),
    ((3, 4), (None, -1, slice(None, None, 2)), (1, 2)),
    ((3, 4), (slice(1, 1), Ellipsis), (4,)),
    ((), (), ()),
])
def test_set_plan_matches_numpy_broadcast(indexing_plan, shape, index, value_shape):
    _assert_numpy(indexing_plan, shape, index, value_shape=value_shape, dtype="float32")


@pytest.mark.parametrize("mode", ("get", "set"))
def test_seeded_small_plans_match_numpy(indexing_plan, mode):
    rng = np.random.default_rng(20260906)
    for _ in range(24):
        shape = tuple(int(value) for value in rng.integers(1, 5, size=int(rng.integers(1, 4))))
        index = []
        for size in shape:
            if rng.random() < 0.3:
                index.append(int(rng.integers(-size, size)))
            else:
                index.append(slice(int(rng.integers(-size - 1, size + 1)),
                                   int(rng.integers(-size - 1, size + 1)),
                                   int(rng.integers(1, 4))))
        if rng.random() < 0.5:
            index.insert(int(rng.integers(0, len(index) + 1)), None)
        index = tuple(index)
        value_shape = None
        if mode == "set":
            selected_shape = np.empty(shape)[index].shape
            start = int(rng.integers(0, len(selected_shape) + 1))
            value_shape = tuple(dim if rng.random() < 0.5 else 1
                                for dim in selected_shape[start:])
        _assert_numpy(indexing_plan, shape, index, value_shape=value_shape, dtype="int8")


@pytest.mark.parametrize("mode,shape,slices,width,other_shape,reason", [
    ("get", (3,), [(1, 3, 0, 1)], 8, (), "out of bounds"),
    ("get", (3,), [(1, -4, 0, 1)], 8, (), "out of bounds"),
    ("get", (3,), [(2, -1, 2, 1)], 8, (3,), "not normalized"),
    ("get", (3,), [(2, 0, 3, 0)], 8, (3,), "slice step"),
    ("get", (3,), [(2, 2, 0, -1)], 8, (3,), "slice step"),
    ("get", (3,), [(4, 0, 0, 1), (4, 0, 0, 1)], 8, (3,), "invalid slice rank"),
    ("get", (3,), [], 0, (3,), "zero dtype width"),
    ("get", (3,), [], 8, (4,), "output shape"),
    ("set", (3, 4), [], 8, (2,), "not broadcastable"),
    ("get", (int(np.iinfo(np.uintp).max), 2), [], 1, (), "overflow"),
    ("set", (3, 4), [], 8, (int(np.iinfo(np.uintp).max), 2), "overflow"),
])
def test_invalid_plans_fail_before_host_copy(indexing_plan, mode, shape, slices,
                                            width, other_shape, reason):
    result = _run(indexing_plan, mode, shape, slices, width, other_shape)
    assert result.startswith("error ") and reason in result, result


def test_copy_identity_overlap_and_address_overflow(indexing_plan):
    result = subprocess.run([str(indexing_plan)], input="intervals\n",
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "intervals-ok"
