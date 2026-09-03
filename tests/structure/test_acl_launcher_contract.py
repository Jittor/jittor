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
EXPAND_SOURCE = ROOT / "python/jittor/extern/acl/aclops/expand_op_acl.cc"
FLOOR_SOURCE = ROOT / "python/jittor/extern/acl/aclops/floor_op_acl.cc"
NANTONUM_SOURCE = ROOT / "python/jittor/extern/acl/aclops/nantonum_op_acl.cc"
TRIU_SOURCE = ROOT / "python/jittor/extern/acl/aclops/triu_op_acl.cc"
SIGMOID_SOURCE = ROOT / "python/jittor/extern/acl/aclops/sigmoid_op_acl.cc"
TRANSPOSE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/transpose_op_acl.cc"
SOFTMAX_SOURCE = ROOT / "python/jittor/extern/acl/aclops/softmax_op_acl.cc"
EMBEDDING_SOURCE = ROOT / "python/jittor/extern/acl/aclops/embedding_op_acl.cc"
ROLL_SOURCE = ROOT / "python/jittor/extern/acl/aclops/roll_op_acl.cc"
CLAMP_SOURCE = ROOT / "python/jittor/extern/acl/aclops/clamp_op_acl.cc"
STACK_SOURCE = ROOT / "python/jittor/extern/acl/aclops/stack_op_acl.cc"
FLIP_SOURCE = ROOT / "python/jittor/extern/acl/aclops/flip_op_acl.cc"
CONCAT_SOURCE = ROOT / "python/jittor/extern/acl/aclops/concat_op_acl.cc"
GATHER_SOURCE = ROOT / "python/jittor/extern/acl/aclops/gather_scatter_op_acl.cc"


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


def test_expand_family_uses_launcher_and_keeps_async_policy():
    source = EXPAND_SOURCE.read_text()
    assert "launch(ret, aclnnExpand, false);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_floor_family_uses_launcher_and_keeps_sync_policy():
    source = FLOOR_SOURCE.read_text()
    assert "launch(ret, aclnnFloor, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_nantonum_family_uses_launcher_and_keeps_sync_policy():
    source = NANTONUM_SOURCE.read_text()
    assert "attr->nan" in source
    assert "attr->posinf" in source
    assert "attr->neginf" in source
    assert "launch(ret, aclnnNanToNum, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_triu_family_uses_launcher_and_keeps_sync_policy():
    source = TRIU_SOURCE.read_text()
    assert "attr->diagonal" in source
    assert "launch(ret, aclnnTriu, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_sigmoid_forward_uses_launcher_and_backward_remains_present():
    source = SIGMOID_SOURCE.read_text()
    forward = source[source.index("void SigmoidOpRunner::executeOp"):source.index("SigmoidBackwardOpRunner::SigmoidBackwardOpRunner")]
    assert "launch(ret, aclnnSigmoid, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void SigmoidBackwardOpRunner::executeOp" in source


def test_transpose_family_uses_launcher_and_keeps_dim_cleanup():
    source = TRANSPOSE_SOURCE.read_text()
    assert "attr->axes" in source
    assert "launch(ret, aclnnPermute, true);" in source
    assert "aclDestroyIntArray(dim);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_softmax_forward_uses_launcher_and_backward_remains_present():
    source = SOFTMAX_SOURCE.read_text()
    forward = source[source.index("void SoftmaxOpRunner::executeOp"):source.index("SoftmaxBackwardOpRunner::SoftmaxBackwardOpRunner")]
    assert "attr->dim" in forward
    assert "launch(ret, aclnnSoftmax, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void SoftmaxBackwardOpRunner::executeOp" in source


def test_embedding_forward_uses_launcher_and_backward_remains_present():
    source = EMBEDDING_SOURCE.read_text()
    forward = source[source.index("void EmbeddingOpRunner::executeOp"):source.index("EmbeddingBackwardOpRunner::EmbeddingBackwardOpRunner")]
    assert "launch(ret, aclnnEmbedding, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void EmbeddingBackwardOpRunner::executeOp" in source


def test_roll_family_uses_launcher_and_keeps_array_cleanup():
    source = ROLL_SOURCE.read_text()
    assert "shifts_array" in source
    assert "dims_array" in source
    assert "launch(ret, aclnnRoll, true);" in source
    assert "aclDestroyIntArray(dims_array);" in source
    assert "aclDestroyIntArray(shifts_array);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_gather_forward_uses_launcher_and_scatter_remains_present():
    source = GATHER_SOURCE.read_text()
    gather = source[source.index("void GatherOpRunner::executeOp"):source.index("ScatterOpRunner::ScatterOpRunner")]
    assert "attr->dim" in gather
    assert "launch(ret, aclnnGather, true);" in gather
    assert "checkRet(ret);" not in gather
    assert "mallocWorkSpace(workspaceSize)" not in gather
    assert "syncRun();" not in gather
    assert "void ScatterOpRunner::executeOp" in source


def test_clamp_tensor_uses_launcher_and_keeps_three_input_query():
    source = CLAMP_SOURCE.read_text()
    assert source.count("inputTensors[") >= 3
    assert "aclnnClampTensorGetWorkspaceSize" in source
    assert "launch(ret, aclnnClampTensor, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_stack_uses_launcher_and_keeps_tensor_list_setup():
    source = STACK_SOURCE.read_text()
    assert "aclCreateTensorList" in source
    assert "attr->dim" in source
    assert "launch(ret, aclnnStack, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_flip_uses_launcher_and_keeps_axes_setup():
    source = FLIP_SOURCE.read_text()
    assert "ReduceAttr" in source
    assert "aclCreateIntArray" in source
    assert "launch(ret, aclnnFlip, true);" in source
    assert "checkRet(ret);" not in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_concat_forward_uses_launcher_and_split_remains_present():
    source = CONCAT_SOURCE.read_text()
    concat = source[source.index("void ConcatOpRunner::executeOp"):source.index("SplitWithSizeOpRunner::SplitWithSizeOpRunner")]
    assert "aclCreateTensorList" in concat
    assert "attr->dim" in concat
    assert "launch(ret, aclnnCat, true);" in concat
    assert "checkRet(ret);" not in concat
    assert "mallocWorkSpace(workspaceSize)" not in concat
    assert "syncRun();" not in concat
    assert "void SplitWithSizeOpRunner::executeOp" in source


def test_scatter_uses_launcher_and_keeps_axis_reduction_query():
    source = GATHER_SOURCE.read_text()
    scatter = source[source.index("void ScatterOpRunner::executeOp"):]
    assert "attr->axis" in scatter
    assert "attr->reduction" in scatter
    assert "launch(ret, aclnnScatter, true);" in scatter
    assert "checkRet(ret);" not in scatter
    assert "mallocWorkSpace(workspaceSize)" not in scatter
    assert "syncRun();" not in scatter
