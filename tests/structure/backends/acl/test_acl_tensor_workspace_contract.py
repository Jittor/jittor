"""Static contracts for ACL tensor descriptors and temporary workspace ownership."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ACL_ROOT = REPO_ROOT / "backends" / "acl"
UTILS_H = ACL_ROOT / "include" / "aclops" / "utils.h"
UTILS_CC = ACL_ROOT / "kernels" / "native" / "utils.cc"
ACL_H = ACL_ROOT / "include" / "acl_jittor.h"
ACL_CC = ACL_ROOT / "src" / "acl_jittor.cc"
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
    header = (REPO_ROOT / "backends/acl/include/acl_workspace.h").read_text(encoding="utf-8")
    source = (REPO_ROOT / "backends/acl/src/workspace.cc").read_text(encoding="utf-8")
    assert "void* mallocWorkSpace(uint64_t size);" in header
    body = _function_body(source, "void* mallocWorkSpace(uint64_t size)")
    assert "runtime_executor().temp_allocator" in body
    assert "get_allocator(device, true)" in body
    assert "workspace.allocator = allocator" in body
    assert "workspace.allocation = allocation" in body
    assert "aclrtMalloc" not in body
    assert "LOGf" in body

    reset = body.index("release_workspace(workspace);")
    allocate = body.index("->alloc(")
    commit = body.index("workspace.address = address")
    assert reset < allocate < commit

    release = _function_body(source, "void release_workspace(Workspace& workspace)")
    assert release.index("aclrtSynchronizeStream") < release.index("workspace = Workspace()")
    assert release.index("workspace = Workspace()") < release.index("allocator->free(")
    assert "previous.address, previous.size, previous.allocation" in release
    assert "void *workspaceAddr =" not in ACL_CC.read_text(encoding="utf-8")


def test_ascend_guide_has_workspace_failure_and_release_checks():
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "ACL workspace allocation failed",
        "workspace requested bytes",
        "workspace allocator",
        "npu-smi info",
        "forbid_backend_fallbacks()",
        "backend_fallback_count()",
        "backend_fallback=error",
        "before-workspace",
        "after-workspace",
        "process exit",
    ):
        assert required in guide
