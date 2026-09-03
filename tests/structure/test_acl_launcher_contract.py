from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_HEADER = ROOT / "python/jittor/extern/acl/aclops/base_op.h"
BASE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/base_op_acl.cc"
UNARY_SOURCE = ROOT / "python/jittor/extern/acl/aclops/unary_op_acl.cc"
BINARY_SOURCE = ROOT / "python/jittor/extern/acl/aclops/binary_op_acl.cc"
TERNARY_SOURCE = ROOT / "python/jittor/extern/acl/aclops/ternary_op_acl.cc"
REDUCE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/reduce_op_acl.cc"
CUMSUM_SOURCE = ROOT / "python/jittor/extern/acl/aclops/cumsum_op_acl.cc"
MATMUL_SOURCE = ROOT / "python/jittor/extern/acl/aclops/matmul_op_acl.cc"


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


def test_ternary_family_uses_launcher_and_keeps_async_policy():
    source = TERNARY_SOURCE.read_text()
    assert "launch(ret, aclnnSWhere, false);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_reduce_single_step_families_use_launcher_and_prod_stays_special():
    source = REDUCE_SOURCE.read_text()
    for name in ("aclnnReduceSum", "aclnnMean", "aclnnAmax", "aclnnAmin"):
        assert f"launch(ret, {name}, true);" in source
    fixed = source[source.index("case 9:"):source.index("case 13:")]
    assert "mallocWorkSpace(workspaceSize)" not in fixed
    prod = source[source.index("case 13:"):source.index("default:")]
    assert "mallocWorkSpace(workspaceSize)" in prod
    assert "aclrtSynchronizeStream(aclstream)" in prod


def test_cumsum_family_uses_launcher_and_keeps_sync_policy():
    source = CUMSUM_SOURCE.read_text()
    assert "launch(ret, aclnnCumsum, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_matmul_family_uses_launcher_and_keeps_sync_policy():
    source = MATMUL_SOURCE.read_text()
    assert "cube_math_type" in source
    assert "launch(ret, aclnnMatmul, true);" in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source
