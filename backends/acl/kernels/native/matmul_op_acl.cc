#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "matmul_op_acl.h"

namespace jittor
{
    MatMulOpRunner::MatMulOpRunner() : BaseOpRunner("MatMul")
    {
    }
    void MatMulOpRunner::setupInputDesc()
    {
        auto input_num = in_.size();
        for (int input_idx = 0; input_idx < input_num; input_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < in_[input_idx]->shape.size(); j++)
            {
                shape.push_back(in_[input_idx]->shape[j]);
            }
            inputShapes.push_back(shape);
        }
        for (int idx = 0; idx < input_num; idx++)
        {
            inputTensors.push_back(nullptr);
            if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0))
            {
                auto ret = CreateFakeTransAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, in_[idx]);
                if (ret != ACL_SUCCESS) LOGf << name << ": transposed input tensor creation failed. ERROR:" << ret;
            }
            else
            {
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, in_[idx]);
                if (ret != ACL_SUCCESS) LOGf << name << ": input tensor creation failed. ERROR:" << ret;
            }
        }
    }
    void MatMulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {

        ret = aclnnMatmulGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], cube_math_type, &workspaceSize, &executor);
        launch(ret, aclnnMatmul, true);
    }
}
