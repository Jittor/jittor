#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "ops/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/transpose_op.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "executor.h"
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
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
            if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0) )
            {
                auto ret = CreateFakeTransAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }
            else
            {
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }
        }
    }
    void MatMulOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
       
        ret =  aclnnMatmulGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], 1, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMatmul failed. ERROR: %d\n", name.c_str(), ret); return);
        // syncRun();
    }
}