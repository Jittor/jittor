"""Static contracts for ACL tensor descriptors and temporary workspace ownership."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACL_ROOT = REPO_ROOT / "python" / "jittor" / "extern" / "acl"
UTILS_H = ACL_ROOT / "aclops" / "utils.h"
UTILS_CC = ACL_ROOT / "aclops" / "utils.cc"
ACL_H = ACL_ROOT / "acl_jittor.h"
ACL_CC = ACL_ROOT / "acl_jittor.cc"
GUIDE = REPO_ROOT / "docs" / "guides" / "ascend-910b.md"


def _function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:position]
    raise AssertionError("unterminated function: {}".format(signature))


def test_acl_tensor_creation_returns_a_real_failure_status():
    header = UTILS_H.read_text(encoding="utf-8")
    source = UTILS_CC.read_text(encoding="utf-8")
    assert "aclError CreateAclTensor(" in header
    assert "aclError CreateFakeTransAclTensor(" in header
    for signature in ("aclError CreateAclTensor(", "aclError CreateFakeTransAclTensor("):
        body = _function_body(source, signature)
        assert "*tensor == nullptr" in body
        assert "ACL_ERROR_FAILURE" in body
        assert "ACL_SUCCESS" in body
        assert "return 0;" not in body


def test_acl_workspace_uses_one_retryable_temp_allocation_contract():
    header = ACL_H.read_text(encoding="utf-8")
    source = ACL_CC.read_text(encoding="utf-8")
    assert "void *mallocWorkSpace(uint64_t size);" in header
    body = _function_body(source, "void *mallocWorkSpace(uint64_t size)")
    assert "runtime_executor().temp_allocator" in body
    assert "workspaceAllocator" in body
    assert "workspaceAllocation" in body
    assert "aclrtMalloc" not in body
    assert "LOGf" in body

    reset = body.index("releaseWorkSpace();")
    allocate = body.index("->alloc(")
    commit = body.index("workspaceAddr = new_workspace")
    assert reset < allocate < commit

    release = _function_body(source, "void releaseWorkSpace()")
    for field in ("workspaceAddr", "nowWorkSpaceSize",
                  "workspaceAllocator", "workspaceAllocation"):
        assert field in release
    assert "allocator->free(ptr, size, allocation)" in release
    assert release.index("workspaceAddr = nullptr") < release.index("allocator->free(")


def test_ascend_guide_has_workspace_failure_and_release_checks():
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "ACL workspace allocation failed",
        "workspace requested bytes",
        "workspace allocator",
        "npu-smi info",
        "fallback cpu",
        "before-workspace",
        "after-workspace",
        "process exit",
    ):
        assert required in guide
