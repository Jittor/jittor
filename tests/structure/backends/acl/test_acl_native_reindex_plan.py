"""Execute native ACL affine-reindex byte plans without importing a device runtime."""
import json
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
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace jittor::acl_indexing;

std::vector<size_t> shape() {
    size_t count; std::cin >> count;
    std::vector<size_t> result(count);
    for (auto& value : result) std::cin >> value;
    return result;
}
int main() {
    std::string mode; std::cin >> mode;
    const auto input = shape(), output = shape();
    size_t bytes; std::cin >> bytes;
    CopyPlan plan;
    std::string reason;
    if (mode == "broadcast") {
        uint32_t bcast, keep; std::cin >> bcast >> keep;
        reason = make_broadcast_plan(input, output, bcast, keep, bytes, plan);
    } else {
        size_t count; std::cin >> count;
        std::vector<std::string> expressions(count);
        for (auto& text : expressions) std::cin >> std::quoted(text);
        reason = make_reindex_plan(input, output, expressions, bytes, plan);
    }
    if (!reason.empty()) { std::cout << "error " << reason << '\n'; return 0; }
    size_t input_bytes; std::cin >> input_bytes;
    std::vector<unsigned char> source(input_bytes);
    for (auto& value : source) { unsigned int byte; std::cin >> byte; value = byte; }
    size_t elements;
    if (!product(output, elements)) return 2;
    std::vector<unsigned char> target(elements * bytes, 0xa5);
    size_t calls = 0;
    for_each_copy(plan, [&](size_t from, size_t to, size_t length) {
        if (from > source.size() || length > source.size() - from ||
            to > target.size() || length > target.size() - to)
            throw std::runtime_error("copy outside source/target allocation");
        std::memcpy(target.data() + to, source.data() + from, length);
        ++calls;
    });
    if (!std::cin.good()) return 3;
    std::cout << "ok " << calls << ' ' << target.size();
    for (auto value : target) std::cout << ' ' << static_cast<unsigned int>(value);
    std::cout << '\n';
}
'''


@pytest.fixture(scope="module")
def reindex_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("acl-native-reindex-host")
    source, executable = directory / "plan.cc", directory / "plan"
    source.write_text(HARNESS, encoding="utf8")
    result = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", "-O1", "-Wall", "-Wextra",
         "-I", str(HEADER.parent), str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


def run_plan(executable, input_shape, output_shape, expressions, width, source=None):
    fields = ["reindex", str(len(input_shape)), *map(str, input_shape),
              str(len(output_shape)), *map(str, output_shape),
              str(width), str(len(expressions)), *map(json.dumps, expressions)]
    if source is not None:
        raw = source.tobytes(order="C")
        fields += [str(len(raw)), *map(str, raw)]
    result = subprocess.run([str(executable)], input=" ".join(fields) + "\n",
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def check_plan(executable, source, output_shape, expressions, expected):
    result = run_plan(executable, source.shape, output_shape, expressions, source.itemsize, source)
    assert result.startswith("ok "), result
    fields = result.split()
    calls, length = map(int, fields[1:3])
    raw = bytes(map(int, fields[3:]))
    assert length == len(raw) == expected.nbytes
    assert raw == expected.tobytes(order="C")
    assert (calls == 0) == (expected.size == 0)
    return calls


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "int8", "int16",
                                  "int32", "int64", "float16", "float32", "complex64", "bool"])
def test_identity_preserves_every_storage_byte(reindex_plan, dtype):
    source = np.arange(32).astype(dtype)
    if dtype == "uint16":
        source = np.asarray([0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
                             0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
                             0x7fc1, 0xffc5, 0x7f81, 0xff81] + list(range(14)), dtype=np.uint16)
    elif dtype == "float32":
        source = np.asarray([0,0x80000000,0x7fc00001,0xffc00005,1,0x7f7fffff] +
                            list(range(26)),dtype=np.uint32).view(np.float32)
    elif dtype == "int64":
        source[:4] = [np.iinfo(np.int64).min,np.iinfo(np.int64).max,2**40+3,-2**40-3]
    # A dense identity plan collapses to a real single byte-copy operation.
    assert check_plan(reindex_plan, source, source.shape, ["i0"], source) == 1


@pytest.mark.parametrize("dtype", ["uint16", "int64", "float32"])
def test_permutation_stride_broadcast_and_flatten(reindex_plan, dtype):
    source = np.arange(12,dtype=dtype).reshape(3,4)
    check_plan(reindex_plan,source,(4,3),["i1","i0"],source.T)
    source = np.arange(56,dtype=dtype).reshape(7,8)
    check_plan(reindex_plan,source,(3,3),["2*i0+1","(i1 + 1) * 2"],source[1:7:2,2:8:2])
    source = np.arange(3,dtype=dtype)
    check_plan(reindex_plan,source,(2,3),["i1"],np.broadcast_to(source,(2,3)))
    source = np.arange(12,dtype=dtype)
    check_plan(reindex_plan,source,(3,4),["i0*4 + i1"],source.reshape(3,4))


def test_constant_broadcast_retains_nan_payload_and_negative_zero_bits(reindex_plan):
    for bits in (0x8000,0x7fc1,0xffc5,0x7f81):
        source=np.asarray([bits],dtype=np.uint16)
        check_plan(reindex_plan,source,(2,3),["0"],np.full((2,3),bits,dtype=np.uint16))


@pytest.mark.parametrize("input_shape,output_shape,indexes", [
    ((0,), (0,), ["i0"]),
    ((0,4), (0,4), ["i0","i1"]),
    ((3,4), (0,2), ["i0","i1+1"]),
])
def test_zero_element_outputs_issue_no_copy(reindex_plan,input_shape,output_shape,indexes):
    source=np.empty(input_shape,dtype=np.uint16)
    check_plan(reindex_plan,source,output_shape,indexes,np.empty(output_shape,dtype=np.uint16))


def test_seeded_affine_maps_match_numpy_raw_bytes(reindex_plan):
    rng=np.random.default_rng(20261024)
    for _ in range(40):
        output_shape=tuple(int(x) for x in rng.integers(1,4,size=int(rng.integers(1,4))))
        input_rank=int(rng.integers(1,4))
        coeff=rng.integers(0,3,size=(input_rank,len(output_shape)))
        offsets=rng.integers(0,3,size=input_rank)
        input_shape=tuple(int(offsets[axis]+sum(int(c)*(n-1) for c,n in zip(coeff[axis],output_shape))+1)
                          for axis in range(input_rank))
        source=rng.integers(0,65536,size=input_shape,dtype=np.uint16)
        grid=np.indices(output_shape,dtype=np.int64)
        indices=tuple(offsets[axis]+sum(int(c)*grid[d] for d,c in enumerate(coeff[axis]))
                      for axis in range(input_rank))
        expressions=[str(int(offsets[axis]))+"+"+"+".join(str(int(c))+"*i"+str(d)
                     for d,c in enumerate(coeff[axis])) for axis in range(input_rank)]
        check_plan(reindex_plan,source,output_shape,expressions,source[indices])


@pytest.mark.parametrize("shape,output,indexes,width,reason", [
    ((3,), (3,), ["i0-1"], 2, "nonnegative affine"),
    ((3,), (3,), ["i0/2"], 2, "nonnegative affine"),
    ((3,), (3,), ["i0%2"], 2, "nonnegative affine"),
    ((3,), (3,), ["@e0(i0)"], 2, "nonnegative affine"),
    ((3,), (3,), ["i0*i0"], 2, "not affine"),
    ((3,), (3,), ["i2"], 2, "axis"),
    ((3,), (3,), [""], 2, "nonnegative affine"),
    ((3,), (3,), ["(i0"], 2, "nonnegative affine"),
    ((3,), (3,), ["i0+1"], 2, "out of bounds"),
    ((3,), (3,), ["i0*2"], 2, "out of bounds"),
    ((0,), (1,), ["0"], 2, "out of bounds"),
    ((3,), (), ["0"], 2, "nonempty output rank"),
    ((3,4), (3,), ["i0"], 2, "rank mismatch"),
    ((3,), (3,), ["i0"], 0, "zero dtype width"),
    ((3,), (3,), ["18446744073709551616"], 2, "overflow"),
    ((3,), (3,), ["18446744073709551615+1"], 2, "overflow"),
    ((3,), (3,), ["18446744073709551615*i0"], 2, "overflow"),
    ((3,), (3,), ["i0*18446744073709551615*2"], 2, "overflow"),
    ((3,), (3,), ["xshape0"], 2, "nonnegative affine"),
    ((3,), (3,), ["i0); abort();"], 2, "nonnegative affine"),
    ((3,), (3,), ["("*65+"i0"+")"*65], 2, "nesting"),
    ((int(np.iinfo(np.uintp).max),2), (1,), ["0","0"], 2, "overflow"),
])
def test_unsupported_or_unsafe_maps_reject_before_copy(reindex_plan,shape,output,indexes,width,reason):
    result=run_plan(reindex_plan,shape,output,indexes,width)
    assert result.startswith("error ") and reason in result,result


def test_fused_and_standalone_dispatch_share_the_native_byte_owner():
    source=(ROOT/"backends/acl/src/acl_op_exec.cc").read_text()
    native=(ROOT/"backends/acl/kernels/native/native_indexing_op_acl.cc").read_text()
    assert '{"reindex", exec_native_acl_reindex}' in source
    assert 'exec_native_acl_reindex(current_op)' in source
    assert source.count('acl_reindex_unsupported_reason(op)')==2
    preflight=source[source.index("static string fused_acl_unsupported"):
                     source.index("static void exec_acl_sequence")]
    assert preflight.index('acl_reindex_unsupported_reason(op)') < preflight.index('const auto name = fused_acl_name(op)')
    assert '!op->extras.empty()' in native and '!op->overflow_conditions.empty()' in native
    assert 'make_reindex_plan' in native and 'copy_region(op->x, op->y' in native
    assert 'BackendStream{device, reinterpret_cast<void*>(aclstream)}' in native
    assert 'prepare_reindex(static_cast<ReindexOp*>(op), prepared, false)' in native
    assert 'prepare_reindex(op, prepared, true)' in native


def run_broadcast(executable, input_shape, output_shape, mask, keep, width, source=None):
    fields = ["broadcast", str(len(input_shape)), *map(str,input_shape),
              str(len(output_shape)), *map(str,output_shape), str(width),str(mask),str(keep)]
    if source is not None:
        raw=source.tobytes(order="C")
        fields += [str(len(raw)), *map(str,raw)]
    result=subprocess.run([str(executable)],input=" ".join(fields)+"\n",
                          capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout.strip()


def check_broadcast(executable, source, output_shape, mask, keep, expected):
    result=run_broadcast(executable,source.shape,output_shape,mask,keep,source.itemsize,source)
    assert result.startswith("ok "),result
    fields=result.split()
    calls,length=map(int,fields[1:3])
    raw=bytes(map(int,fields[3:]))
    assert length==len(raw)==expected.nbytes
    assert raw==expected.tobytes(order="C")
    assert (calls==0)==(expected.size==0)


@pytest.mark.parametrize("dtype",["uint16","uint32","int16","complex64"])
@pytest.mark.parametrize("shape,output,mask,keep,aligned",[
    ((1,),(5,),1,1,(1,)),
    ((2,2),(2,3,2),2,0,(2,1,2)),  # explicit dims=[1]
    ((1,2),(3,2),1,1,(1,2)),
    ((2,1),(2,4),2,2,(2,1)),
    ((2,),(3,2,4),5,0,(1,2,1)),  # explicit dims=[0,2]
    ((1,),(2,3),3,2,(1,1)),
    ((),(2,3),3,0,(1,1)),
    ((1,1),(2,3),3,3,(1,1)),
    ((3,4),(3,4),0,0,(3,4)),
    ((1,),tuple([1]*31+[2]),2**32-1,2**31,tuple([1]*32)),
])
def test_byte_broadcast_matches_explicit_and_retained_axes(
        reindex_plan,dtype,shape,output,mask,keep,aligned):
    count=int(np.prod(shape,dtype=np.int64))
    source=(np.arange(count)+1).astype(dtype).reshape(shape)
    expected=np.broadcast_to(source.reshape(aligned),output)
    check_broadcast(reindex_plan,source,output,mask,keep,expected)


def test_scalar_broadcast_preserves_raw_bf16_edge_payloads(reindex_plan):
    for bits in (0,0x8000,1,0x807f,0x80,0x8080,0x3f80,0x7f7f,0xff7f,
                 0x7f80,0xff80,0x7fc1,0xffc5,0x7f81,0xff81):
        source=np.asarray([bits],dtype=np.uint16)
        check_broadcast(reindex_plan,source,(5,),1,1,np.full(5,bits,dtype=np.uint16))
    for dtype,values in (("int16",[-32768,32767]),("uint32",[0,2**32-1]),
                         ("complex64",[complex(1,-2),complex(-3,4)])):
        for value in values:
            source=np.asarray([value],dtype=dtype)
            check_broadcast(reindex_plan,source,(2,3),3,2,np.full((2,3),value,dtype=dtype))


@pytest.mark.parametrize("shape,output,mask,keep,aligned",[
    ((1,),(0,),1,1,(1,)),
    ((0,),(0,),0,0,(0,)),
    ((2,1),(2,0),2,2,(2,1)),
    ((0,2),(0,3,2),2,0,(0,1,2)),
])
def test_byte_broadcast_empty_outputs_need_no_copy(reindex_plan,shape,output,mask,keep,aligned):
    source=np.empty(shape,dtype=np.uint16)
    expected=np.broadcast_to(source.reshape(aligned),output)
    check_broadcast(reindex_plan,source,output,mask,keep,expected)


@pytest.mark.parametrize("shape,output,mask,keep,width,reason",[
    ((1,),(5,),0,1,2,"subset"),
    ((1,),(5,),2,0,2,"exceeds output rank"),
    ((2,),(5,),1,1,2,"one-element input"),
    ((2,),(3,),0,0,2,"nonexpanded dimension"),
    ((2,2),(2,3,2),2,2,2,"one-element input"),
    ((1,),(2,3),3,0,2,"not fully consumed"),
    ((2,), (2,2),0,0,2,"input rank is too small"),
    ((1,),(),0,0,2,"output rank"),
    ((1,),tuple([1]*33),0,0,2,"output rank"),
    ((1,),(5,),1,1,0,"zero dtype width"),
    ((0,),(3,),1,1,2,"one-element input"),
    ((1,1),(int(np.iinfo(np.uintp).max),2),3,3,2,"overflow"),
])
def test_byte_broadcast_rejects_invalid_metadata_before_copy(
        reindex_plan,shape,output,mask,keep,width,reason):
    result=run_broadcast(reindex_plan,shape,output,mask,keep,width)
    assert result.startswith("error ") and reason in result,result


def test_byte_broadcast_dispatch_keeps_sdk_fast_dtypes_and_one_copy_owner():
    source=(ROOT/"backends/acl/src/acl_op_exec.cc").read_text()
    native=(ROOT/"backends/acl/kernels/native/native_indexing_op_acl.cc").read_text()
    header=HEADER.read_text()
    assert "acl_broadcast_uses_byte_copy(op)" in source
    assert "acl_broadcast_unsupported_reason(op)" in source
    assert "acl_broadcast_uses_byte_copy(current_op)" in source
    assert "exec_native_acl_broadcast(current_op)" in source
    assert "AclExecutionRunner<ExpandOpRunner>" in source
    predicate=native[native.index("bool acl_broadcast_uses_byte_copy"):native.index("std::string acl_broadcast_unsupported_reason")]
    for dtype in ("ns_uint16","ns_uint32","ns_int16","ns_complex64"):
        assert dtype in predicate
    for dtype in ("ns_float32","ns_float16","ns_bfloat16","ns_int32","ns_int64","ns_uint8"):
        assert dtype not in predicate
    assert "prepare_broadcast(static_cast<BroadcastToOp*>(op), prepared, false)" in native
    assert "prepare_broadcast(op, prepared, true)" in native
    assert "copy_region(op->x, op->z" in native
    planner=header[header.index("inline std::string make_broadcast_plan"):header.index("inline bool address_end")]
    assert "make_selection(output_shape" in planner and "make_set_plan(selection, aligned, plan)" in planner
