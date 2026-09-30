"""Exercise real HCCL binding generation without loading an NPU runtime.

A multiline HcclReduceOp constructor declaration previously stopped the real
registration generator with ``Wrong op args`` before HCCL could compile. This
CPU gate calls that generator on every checked-in HCCL operator header, then
checks the emitted reduce factory, Python binding and argument forwarding.
It does not establish HCCL compilation or collective execution support.
"""
from pathlib import Path
import re

from _helpers.op_registration_generator import load_op_registration_generator


ROOT = Path(__file__).resolve().parents[4]


def test_hccl_headers_generate_all_bindings_and_preserve_reduce_arguments(tmp_path):
    headers = sorted((ROOT / "backends/comm/hccl/ops").glob("*_op.h"))
    expected_names = [header.stem[:-3] for header in headers]
    assert "hccl_reduce" in expected_names
    assert len(headers) >= 4

    # Run the production generator, using the existing runtime-free loader.
    # An unreadable constructor must fail here, not be hidden by this test's
    # checks on its output. No replacement constructor parser is provided.
    source = load_op_registration_generator(tmp_path)(
        [str(header) for header in headers],
        export="hccl_binding_regression", backend="accelerator")
    registrations = re.findall(
        r'register_op_definition<\w+>\(\{ "([^"]+)"', source)
    assert registrations == expected_names
    assert "PYJT_MODULE_INIT(hccl_binding_regression)" in source
    for line in source.splitlines():
        if "register_op_definition<" in line:
            assert line.strip().endswith(", OpBackendAccelerator);")

    factory = re.search(r"VarPtr make_hccl_reduce\(([^)]*)\)", source)
    binding = re.search(r"VarHolder\* hccl_reduce\(([^)]*)\)", source)
    assert factory is not None
    assert binding is not None
    assert [arg.strip() for arg in factory.group(1).split(",")] == [
        "Var* x", "string reduce_op", "int root", "int group_id"]
    assert [arg.strip() for arg in binding.group(1).split(",")] == [
        "VarHolder* x", 'string reduce_op="sum"', "int root=0", "int group_id=0"]
    assert "new HcclReduceOp(x, reduce_op, root, group_id)" in source
    assert "make_hccl_reduce(x->var, reduce_op, root, group_id)" in source
