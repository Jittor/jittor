from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_HEADER = ROOT / "python/jittor/extern/acl/aclops/base_op.h"
BASE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/base_op_acl.cc"
UNARY_SOURCE = ROOT / "python/jittor/extern/acl/aclops/unary_op_acl.cc"
BINARY_SOURCE = ROOT / "python/jittor/extern/acl/aclops/binary_op_acl.cc"


def test_acl_launcher_tail_has_one_auditable_contract():
    header = BASE_HEADER.read_text()
    source = BASE_SOURCE.read_text()
    assert "using AclExecuteLauncher" in header
    assert "void launch(aclnnStatus workspace_ret" in header
    for token in (
            "checkRet(workspace_ret)", "mallocWorkSpace(workspaceSize)",
            "launcher(", "execute launcher failed", "syncRun()"):
        assert token in source


def test_unary_family_uses_launcher_without_changing_sync_policy():
    source = UNARY_SOURCE.read_text()
    assert "launch(ret, it->second.executeFunc, false);" in source
    assert "CHECK_RET(ret == ACL_SUCCESS" not in source


def test_binary_family_uses_shared_launcher_without_tail_copy():
    source = BINARY_SOURCE.read_text()
    assert "launch(ret, it->second.executeFunc, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source
