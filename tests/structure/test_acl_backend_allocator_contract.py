"""ACL BackendOps must publish an ACL-owned allocator, not CUDA pools."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backends/acl/src/backend.cc"


def test_acl_backend_uses_acl_allocator_adapter():
    text = SOURCE.read_text(encoding="utf-8")
    assert "class AclAllocator final : public Allocator" in text
    assert "Allocator* acl_allocator(int device, BackendMemoryKind kind)" in text
    assert "ops.allocator = acl_allocator;" in text
    assert "ops.allocator = accelerator_allocator_for;" not in text
    assert "ACL allocator does not support managed memory" in text


def test_acl_allocator_preserves_device_identity_and_pinned_host_semantics():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("class AclAllocator final")
    end = text.index("Allocator* acl_allocator", start)
    body = text[start:end]
    assert "kind_ == BackendMemoryKind::Pinned ? 0 : Allocator::_cuda" in body
    assert "kind_ == BackendMemoryKind::Pinned ? -1 : device_" in body
    assert "return allocate_memory(device_, kind_, size);" in body
    assert "free_memory(device_, kind_, pointer);" in body
