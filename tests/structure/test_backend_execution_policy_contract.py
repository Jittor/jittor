"""Backend execution-policy ownership without importing a runtime or device SDK."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python/jittor/src"


def test_acl_descriptor_declares_execution_requirements():
    source = (SRC / "runtime/backends/accelerator.cc").read_text()
    start = source.index("if (ops.id == BackendId::Acl)")
    policy_block = source[start:source.index("\n    }", start)]
    for field, value in (("supports_parallel_compile", "false"),
                         ("requires_pinned_host_storage", "true"),
                         ("preserve_reduction_dtype", "true"),
                         ("native_low_precision_reduction", "true")):
        assert "ops.execution.%s = %s;" % (field, value) in policy_block


def test_backend_compile_constraint_precedes_user_parallel_request():
    source = (SRC / "parallel_compiler.cc").read_text()
    function = source[source.index("void parallel_compile_all_ops("):]
    assert function.index("execution.supports_parallel_compile") < function.index("if (!force_compile)")
    assert "use_parallel_op_compiler =" not in function


def test_array_staging_uses_actual_backend_host_allocator():
    allocator = (SRC / "mem/allocator.cc").read_text()
    assert "execution.requires_pinned_host_storage" in allocator
    assert "return get_allocator(-1, false);" in allocator
    assert "use_pinned_host_memory() ? BackendMemoryKind::Pinned" in allocator
    for path in (SRC / "ops/array_op.cc", SRC / "pyjt/py_array_op.cc"):
        source = path.read_text()
        assert "!save_mem && !use_pinned_host_memory()" in source
        assert "Allocation(get_array_host_allocator(), output->size)" in source
        assert "!save_mem && !use_cuda_host_allocator" not in source


def test_reduction_policy_is_target_scoped_without_changing_amp():
    source = (SRC / "ops/reduce_op.cc").read_text()
    assert source.count("backend_ops(runtime_use_cuda()") == 2
    assert source.count("? accelerator_backend_id() : BackendId::Cpu).execution") == 2
    assert source.count("!policy.native_low_precision_reduction") == 2
    assert source.count("reduce_dtype_infer(ns, x->ns, policy.preserve_reduction_dtype)") == 2
    assert "amp_reg |= " not in source and "amp_reg = " not in source
