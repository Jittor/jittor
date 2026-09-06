from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/jittor/extern/acl/acl_op_exec.cc"


def test_acl_publishes_backend_kernels_without_removing_operator_definitions():
    text = SOURCE.read_text(encoding="utf-8")
    assert "register_backend_implementation_composer(" in text
    assert 'BackendId::Acl, compose_acl_implementation, "acl-native-v1"' in text
    assert "const OpDef &definition, const OpImplementation &original" in text
    assert "registered_op_names()" not in text
    assert "implementation.kernel.fallback_only = !implementation.kernel.native" in text
    assert "unregister_op(" not in text
    assert "do_compile_hook" not in text
    assert 'cuda_src.find("acl")' not in text
    assert 'startswith(name, "cu")' not in text
    for implementation in ("fused", "code", "mapped", "single", "unsupported"):
        assert f"implementation.kernel.compile = compile_acl_{implementation}" in text


def test_acl_code_and_collectives_declare_their_source_backend():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'if (code->backend != "acl")' in text
    assert 'strncmp(op->name(), "hccl"' not in text
    for header in (ROOT / "python/jittor/extern/acl/hccl/ops").glob("*_op.h"):
        assert "kernel.compile = compile_registered_source" in header.read_text()
    compiler = (ROOT / "python/jittor/src/op_compiler.cc").read_text()
    assert "op->implementation().kernel.compile" in compiler
    assert "do_compile_hook" not in compiler
