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
WHERE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/where_op_acl.cc"
RANGE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/index_op_acl.cc"
DROPOUT_SOURCE = ROOT / "python/jittor/extern/acl/aclops/dropout_op_acl.cc"
RELU_SOURCE = ROOT / "python/jittor/extern/acl/aclops/relu_op_acl.cc"
ARG_REDUCE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/arg_reduce_op_acl.cc"
SILU_SOURCE = ROOT / "python/jittor/extern/acl/aclops/silu_op_acl.cc"
BMM_SOURCE = ROOT / "python/jittor/extern/acl/aclops/bmm_op_acl.cc"
TRUTH_REDUCE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/truth_reduce_op_acl.cc"
CONV_SOURCE = ROOT / "python/jittor/extern/acl/aclops/conv_op_acl.cc"
NORMS_SOURCE = ROOT / "python/jittor/extern/acl/aclops/norms_op_acl.cc"
ROPE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/rope_op_acl.cc"
POOL_SOURCE = ROOT / "python/jittor/extern/acl/aclops/pool_op_acl.cc"
RANDOM_SOURCE = ROOT / "python/jittor/extern/acl/aclops/random_op_acl.cc"
UPSAMPLE_SOURCE = ROOT / "python/jittor/extern/acl/aclops/upsample_op_acl.cc"
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


def test_softmax_backward_uses_launcher_and_keeps_dim_query():
    source = SOFTMAX_SOURCE.read_text()
    backward = source[source.index("void SoftmaxBackwardOpRunner::executeOp"):]
    assert "attr->dim" in backward
    assert "launch(ret, aclnnSoftmaxBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_embedding_forward_uses_launcher_and_backward_remains_present():
    source = EMBEDDING_SOURCE.read_text()
    forward = source[source.index("void EmbeddingOpRunner::executeOp"):source.index("EmbeddingBackwardOpRunner::EmbeddingBackwardOpRunner")]
    assert "launch(ret, aclnnEmbedding, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void EmbeddingBackwardOpRunner::executeOp" in source


def test_embedding_backward_uses_launcher_and_keeps_attribute_query():
    source = EMBEDDING_SOURCE.read_text()
    backward = source[source.index("void EmbeddingBackwardOpRunner::executeOp"):]
    assert "numEmbeddings" in backward
    assert "paddingIdx" in backward
    assert "scaleGradByFreq" in backward
    assert "launch(ret, aclnnEmbeddingDenseBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


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


def test_split_with_size_uses_launcher_and_keeps_tensor_list_setup():
    source = CONCAT_SOURCE.read_text()
    split = source[source.index("void SplitWithSizeOpRunner::executeOp"):]
    assert "splitSize" in split
    assert "aclCreateTensorList" in split
    assert "attr->dim" in split
    assert "launch(ret, aclnnSplitWithSize, true);" in split
    assert "checkRet(ret);" not in split
    assert "mallocWorkSpace(workspaceSize)" not in split
    assert "syncRun();" not in split


def test_nonzero_uses_launcher_and_swhere_remains_present():
    source = WHERE_SOURCE.read_text()
    nonzero = source[source.index("void NonzeroOpRunner::executeOp"):]
    assert "launch(ret, aclnnNonzero, true);" in nonzero
    assert "checkRet(ret);" not in nonzero
    assert "mallocWorkSpace(workspaceSize)" not in nonzero
    assert "syncRun();" not in nonzero
    assert "aclnnSWhere" in source


def test_range_uses_launcher_and_keeps_scalar_lifecycle():
    source = RANGE_SOURCE.read_text()
    range_source = source[source.index("void RangeOpRunner::executeOp"):source.index("void IndexOpRunner::") if "void IndexOpRunner::" in source else len(source)]
    assert "aclCreateScalar" in range_source
    assert "launch(ret, aclnnRange, true);" in range_source
    assert "aclDestroyScalar(start);" in range_source
    assert "aclDestroyScalar(end);" in range_source
    assert "aclDestroyScalar(step);" in range_source
    assert "checkRet(ret);" not in range_source
    assert "mallocWorkSpace(workspaceSize)" not in range_source
    assert "syncRun();" not in range_source


def test_dropout_forward_uses_launcher_and_backward_remains_present():
    source = DROPOUT_SOURCE.read_text()
    forward = source[source.index("void DropoutOpRunner::executeOp"):source.index("DropoutBackwardOpRunner::DropoutBackwardOpRunner")]
    assert "attr->p" in forward
    assert "attr->train" in forward
    assert "attr->seed" in forward
    assert "attr->offset" in forward
    assert "launch(ret, aclnnDropout, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void DropoutBackwardOpRunner::executeOp" in source


def test_dropout_backward_uses_launcher_and_keeps_scale_query():
    source = DROPOUT_SOURCE.read_text()
    backward = source[source.index("void DropoutBackwardOpRunner::executeOp"):]
    assert "attr->scale" in backward
    assert "launch(ret, aclnnDropoutBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_leaky_relu_forward_uses_launcher_and_backward_remains_present():
    source = RELU_SOURCE.read_text()
    forward = source[source.index("void LeakyReLUOpRunner::executeOp"):source.index("LeakyReLUBackwardOpRunner::LeakyReLUBackwardOpRunner")]
    assert "negativeSlope" in forward
    assert "launch(ret, aclnnLeakyRelu, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void LeakyReLUBackwardOpRunner::executeOp" in source


def test_leaky_relu_backward_uses_launcher_and_keeps_scalar_cleanup():
    source = RELU_SOURCE.read_text()
    backward = source[source.index("void LeakyReLUBackwardOpRunner::executeOp"):]
    assert "negativeSlope" in backward
    assert "selfIsResult" in backward
    assert "launch(ret, aclnnLeakyReluBackward, true);" in backward
    assert "aclDestroyScalar(negativeSlope);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_arg_reduce_max_min_use_shared_launcher():
    source = ARG_REDUCE_SOURCE.read_text()
    assert "is_max" in source
    assert "keepdims" in source
    assert "aclnnMaxDimGetWorkspaceSize" in source
    assert "aclnnMinDimGetWorkspaceSize" in source
    assert "AclExecuteLauncher launcher = is_max ? aclnnMaxDim : aclnnMinDim;" in source
    assert "launch(ret, launcher, true);" in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_silu_forward_uses_launcher_and_other_owners_remain_present():
    source = SILU_SOURCE.read_text()
    forward = source[source.index("void SiLUOpRunner::executeOp"):source.index("SiLUBackwardOpRunner::SiLUBackwardOpRunner")]
    assert "launch(ret, aclnnSilu, true);" in forward
    assert "checkRet(ret);" not in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void SiLUBackwardOpRunner::executeOp" in source
    assert "void SwishOpRunner::executeOp" in source
    assert "void SwiGluOpRunner::executeOp" in source


def test_silu_backward_uses_launcher_and_forward_remains_present():
    source = SILU_SOURCE.read_text()
    backward = source[source.index("void SiLUBackwardOpRunner::executeOp"):source.index("SwishOpRunner::SwishOpRunner")]
    assert "launch(ret, aclnnSiluBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward
    assert "launch(ret, aclnnSilu, true);" in source


def test_swish_forward_uses_launcher_and_other_owners_remain_present():
    source = SILU_SOURCE.read_text()
    forward = source[source.index("void SwishOpRunner::executeOp"):source.index("SwishBackwardOpRunner::SwishBackwardOpRunner")]
    assert "launch(ret, aclnnSwish, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void SwishBackwardOpRunner::executeOp" in source


def test_swish_backward_uses_launcher_and_forward_remains_present():
    source = SILU_SOURCE.read_text()
    backward = source[source.index("void SwishBackwardOpRunner::executeOp"):source.index("SwiGluOpRunner::SwiGluOpRunner")]
    assert "launch(ret, aclnnSwishBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward
    assert "launch(ret, aclnnSwish, true);" in source


def test_swiglu_uses_launcher_and_keeps_silu_families():
    source = SILU_SOURCE.read_text()
    swiglu = source[source.index("void SwiGluOpRunner::executeOp"):]
    assert "launch(ret, aclnnSwiGlu, true);" in swiglu
    assert "mallocWorkSpace(workspaceSize)" not in swiglu
    assert "syncRun();" not in swiglu
    assert "launch(ret, aclnnSilu, true);" in source


def test_batch_matmul_uses_launcher_and_keeps_cube_math_type():
    source = BMM_SOURCE.read_text()
    assert "cube_math_type" in source
    assert "launch(ret, aclnnBatchMatMul, true);" in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_truth_reduce_all_any_use_shared_launcher_and_keep_raii_axes():
    source = TRUTH_REDUCE_SOURCE.read_text()
    assert "reduce_all" in source
    assert "aclnnAllGetWorkspaceSize" in source
    assert "aclnnAnyGetWorkspaceSize" in source
    assert "AclExecuteLauncher launcher = reduce_all ? aclnnAll : aclnnAny;" in source
    assert "launch(ret, launcher, true);" in source
    assert "unique_ptr<aclIntArray" in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_conv_forward_uses_launcher_and_backward_remains_present():
    source = CONV_SOURCE.read_text()
    forward = source[source.index("void Conv2dOpRunner::executeOp"):source.index("void Conv2dBackwardOpRunner::executeOp")]
    assert "attr->group" in forward
    assert "launch(ret, aclnnConvolution, true);" in forward
    assert "aclDestroyIntArray(strides);" in forward
    assert "aclDestroyIntArray(dilations);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void Conv2dBackwardOpRunner::executeOp" in source


def test_conv_backward_uses_launcher_and_keeps_three_outputs():
    source = CONV_SOURCE.read_text()
    backward = source[source.index("void Conv2dBackwardOpRunner::executeOp"):]
    assert "outputTensors[2]" in backward
    assert "launch(ret, aclnnConvolutionBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_rms_norm_forward_uses_launcher_and_grad_remains_present():
    source = NORMS_SOURCE.read_text()
    forward = source[source.index("void RmsNormOpRunner::executeOp"):source.index("RmsNormGradOpRunner::RmsNormGradOpRunner")]
    assert "attr->eps" in forward
    assert "outputTensors[1]" in forward
    assert "launch(ret, aclnnRmsNorm, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void RmsNormGradOpRunner::executeOp" in source


def test_rms_norm_grad_uses_launcher_and_forward_remains_present():
    source = NORMS_SOURCE.read_text()
    grad = source[source.index("void RmsNormGradOpRunner::executeOp"):]
    assert "aclnnRmsNormGradGetWorkspaceSize" in grad
    assert "launch(ret, aclnnRmsNormGrad, true);" in grad
    assert "mallocWorkSpace(workspaceSize)" not in grad
    assert "syncRun();" not in grad
    assert "launch(ret, aclnnRmsNorm, true);" in source


def test_layer_norm_forward_uses_launcher_and_backward_remains_present():
    source = NORMS_SOURCE.read_text()
    forward = source[source.index("void LayerNormOpRunner::executeOp"):source.index("LayerNormBackwardOpRunner::LayerNormBackwardOpRunner")]
    assert "normalizedShape" in forward
    assert "attr->eps" in forward
    assert "outputTensors[2]" in forward
    assert "launch(ret, aclnnLayerNorm, true);" in forward
    assert "aclDestroyIntArray(normalizedShape);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void LayerNormBackwardOpRunner::executeOp" in source


def test_layer_norm_backward_uses_launcher_and_keeps_descriptor_cleanup():
    source = NORMS_SOURCE.read_text()
    backward = source[source.index("void LayerNormBackwardOpRunner::executeOp"):source.index("GroupNormOpRunner::GroupNormOpRunner")]
    assert "normalizedShape" in backward
    assert "outMask" in backward
    assert "outputTensors[2]" in backward
    assert "launch(ret, aclnnLayerNormBackward, true);" in backward
    assert "aclDestroyIntArray(normalizedShape);" in backward
    assert "aclDestroyBoolArray(outMask);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_group_norm_forward_uses_launcher_and_backward_remains_present():
    source = NORMS_SOURCE.read_text()
    forward = source[source.index("void GroupNormOpRunner::executeOp"):source.index("GroupNormBackwardOpRunner::GroupNormBackwardOpRunner")]
    assert "attr->groups" in forward
    assert "outputTensors[2]" in forward
    assert "launch(ret, aclnnGroupNorm, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void GroupNormBackwardOpRunner::executeOp" in source


def test_group_norm_backward_uses_launcher_and_keeps_output_mask():
    source = NORMS_SOURCE.read_text()
    backward = source[source.index("void GroupNormBackwardOpRunner::executeOp"):source.index("RmsNormOpRunner::RmsNormOpRunner")]
    assert "outputMask" in backward
    assert "attr->groups" in backward
    assert "outputTensors[2]" in backward
    assert "launch(ret, aclnnGroupNormBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_masked_select_uses_launcher_and_keeps_two_inputs():
    source = (ROOT / "python/jittor/extern/acl/aclops/getitem_op_acl.cc").read_text()
    masked = source[source.index("void MaskedSelectOpRunner::executeOp"):source.index("IndexOpRunner::IndexOpRunner")]
    assert masked.count("inputTensors[") >= 2
    assert "launch(ret, aclnnMaskedSelect, true);" in masked
    assert "mallocWorkSpace(workspaceSize)" not in masked
    assert "syncRun();" not in masked


def test_rope_forward_uses_launcher_and_backward_remains_present():
    source = ROPE_SOURCE.read_text()
    forward = source[source.index("void RotaryPositionEmbeddingOpRunner::executeOp"):source.index("RotaryPositionEmbeddingGradOpRunner::RotaryPositionEmbeddingGradOpRunner")]
    assert "inputTensors[0]" in forward
    assert "launch(ret, aclnnRotaryPositionEmbedding, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void RotaryPositionEmbeddingGradOpRunner::executeOp" in source


def test_rope_gradient_uses_launcher_and_keeps_io_query():
    source = ROPE_SOURCE.read_text()
    gradient = source[source.index("void RotaryPositionEmbeddingGradOpRunner::executeOp"):]
    assert "inputTensors[3]" in gradient
    assert "outputTensors[2]" in gradient
    assert "launch(ret, aclnnRotaryPositionEmbeddingGrad, true);" in gradient
    assert "mallocWorkSpace(workspaceSize)" not in gradient
    assert "syncRun();" not in gradient


def test_maxpool_forward_uses_launcher_and_keeps_descriptors():
    source = POOL_SOURCE.read_text()
    forward = source[source.index("void MaxpoolOpRunner::executeOp"):source.index("void AvgpoolOpRunner::executeOp")]
    assert "kernel_size" in forward
    assert "strides" in forward
    assert "pads" in forward
    assert "dilations" in forward
    assert "poolCeil" in forward
    assert "launch(ret, aclnnMaxPool2dWithIndices, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void AvgpoolOpRunner::executeOp" in source


def test_avgpool_forward_uses_launcher_and_maxpool_remains_present():
    source = POOL_SOURCE.read_text()
    forward = source[source.index("void AvgpoolOpRunner::executeOp"):source.index("MaxpoolBackwardOpRunner::MaxpoolBackwardOpRunner")]
    assert "kernel_size" in forward
    assert "poolCeil" in forward
    assert "countIncludePad" in forward
    assert "launch(ret, aclnnAvgPool2d, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void MaxpoolOpRunner::executeOp" in source


def test_avgpool_backward_uses_launcher_and_keeps_descriptor_cleanup():
    source = POOL_SOURCE.read_text()
    backward = source[source.index("void AvgpoolBackwardOpRunner::executeOp"):]
    assert "countIncludePad" in backward
    assert "divisorOverride" in backward
    assert "launch(ret, aclnnAvgPool2dBackward, true);" in backward
    assert "aclDestroyIntArray(strides);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_maxpool_backward_uses_launcher_and_keeps_descriptors():
    source = POOL_SOURCE.read_text()
    backward = source[source.index("void MaxpoolBackwardOpRunner::executeOp"):source.index("AvgpoolBackwardOpRunner::AvgpoolBackwardOpRunner")]
    assert "poolCeil" in backward
    assert "outputTensors[0]" in backward
    assert "launch(ret, aclnnMaxPool2dWithIndicesBackward, true);" in backward
    assert "aclDestroyIntArray(kernel_size);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_random_uniform_normal_share_launcher_and_keep_seed_offset():
    source = RANDOM_SOURCE.read_text()
    assert "RandomUniform" in source
    assert "RandomNormal" in source
    assert "attr->seed" in source
    assert "attr->offset" in source
    assert "launcher = aclnnInplaceUniform;" in source
    assert "launcher = aclnnInplaceNormal;" in source
    assert "launch(ret, launcher, true);" in source
    assert "Not supported random type" in source
    assert "mallocWorkSpace(workspaceSize)" not in source
    assert "syncRun();" not in source


def test_upsample_forward_uses_launcher_and_keeps_output_size_raii():
    source = UPSAMPLE_SOURCE.read_text()
    forward = source[source.index("void UpsampleNearest2dOpRunner::executeOp"):source.index("UpsampleNearest2dBackwardOpRunner::UpsampleNearest2dBackwardOpRunner")]
    assert "outputSize" in forward
    assert "unique_ptr" in forward
    assert "launch(ret, aclnnUpsampleNearest2d, true);" in forward
    assert "mallocWorkSpace(workspaceSize)" not in forward
    assert "syncRun();" not in forward
    assert "void UpsampleNearest2dBackwardOpRunner::executeOp" in source


def test_upsample_backward_uses_launcher_and_keeps_descriptor_raii():
    source = UPSAMPLE_SOURCE.read_text()
    backward = source[source.index("void UpsampleNearest2dBackwardOpRunner::executeOp"):]
    assert "outputSize" in backward
    assert "inputSize" in backward
    assert "unique_ptr" in backward
    assert "launch(ret, aclnnUpsampleNearest2dBackward, true);" in backward
    assert "mallocWorkSpace(workspaceSize)" not in backward
    assert "syncRun();" not in backward


def test_scatter_uses_launcher_and_keeps_axis_reduction_query():
    source = GATHER_SOURCE.read_text()
    scatter = source[source.index("void ScatterOpRunner::executeOp"):]
    assert "attr->axis" in scatter
    assert "attr->reduction" in scatter
    assert "launch(ret, aclnnScatter, true);" in scatter
    assert "checkRet(ret);" not in scatter
    assert "mallocWorkSpace(workspaceSize)" not in scatter
    assert "syncRun();" not in scatter
