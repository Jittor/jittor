from pathlib import Path

from jittor.build.codegen import gen_jit_op_maker


def test_hccl_op_headers_generate_bindings():
    op_dir = Path(__file__).resolve().parents[2] / "backends" / "comm" / "hccl" / "ops"
    headers = sorted(op_dir.glob("*_op.h"))
    assert {path.name for path in headers} == {
        "hccl_all_gather_op.h",
        "hccl_all_reduce_op.h",
        "hccl_broadcast_op.h",
        "hccl_reduce_op.h",
    }
    source = gen_jit_op_maker([str(path) for path in headers], backend="accelerator")
    for operator in ("hccl_all_gather", "hccl_all_reduce", "hccl_broadcast", "hccl_reduce"):
        assert "make_" + operator + "(" in source
