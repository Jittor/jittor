"""Expand the real backend indexing prefix on the host, without launching a kernel."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("operation,expected", [
    ("void", "op[iid] = (Ti)dp[did]"),
    ("add", "atomicAdd(&op[iid], (Ti)dp[did])"),
    ("maximum", "cuda_atomic_max_rmw(&op[iid], (Ti)dp[did])"),
    ("minimum", "cuda_atomic_min_rmw(&op[iid], (Ti)dp[did])"),
    ("multiply", "cuda_atomic_mul(&op[iid], (Ti)dp[did])"),
])
def test_cuda_scatter_update_expands_the_selected_atomic_implementation(operation, expected):
    import jittor as jt
    from jittor_utils.backend_resources import backend_root

    prefix = Path(backend_root(jt.compiler.jittor_path, "cuda")) / "kernels/core/setitem_prefix.cc"
    definitions = {"JIT": "1", "JIT_cuda": "1", "OP": operation,
                   "Ti": "float32", "Td": "float32"}
    expanded = jt.core.op_compiler.precompile(
        definitions, prefix.read_text() +
        "\nvoid indexing_probe() { @expand_macro(indexing_backend_update) }\n")
    # Macro declarations contain every atomic spelling even in the broken
    # implementation. Inspect the expanded function, not the prefix or defines.
    body = expanded.split("void indexing_probe()", 1)[1]
    assert expected in body, body
    if operation != "void":
        assert "op[iid] =" not in body, body


def test_cuda_scatter_update_retains_the_general_operator_expansion():
    import jittor as jt
    from jittor_utils.backend_resources import backend_root

    prefix = Path(backend_root(jt.compiler.jittor_path, "cuda")) / "kernels/core/setitem_prefix.cc"
    expanded = jt.core.op_compiler.precompile(
        {"JIT": "1", "JIT_cuda": "1", "OP": "subtract", "Ti": "float32", "Td": "float32"},
        prefix.read_text() + "\nvoid indexing_probe() { @expand_macro(indexing_backend_update) }\n")
    body = expanded.split("void indexing_probe()", 1)[1]
    assert "op[iid] =" in body and "-" in body, body
    assert "atomic" not in body, body
