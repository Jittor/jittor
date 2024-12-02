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
#include "reduce_op_acl.h"


namespace jittor
{
    ReduceOpRunner::ReduceOpRunner() : BaseOpRunner("reduce")
    {
    }

    void ReduceOpRunner::setupOutputDesc()
    {
        auto output_num = out_.size();

        for (int output_idx = 0; output_idx < output_num; output_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < out_[output_idx]->shape.size(); j++)
            {
                shape.push_back(out_[output_idx]->shape[j]);
            }
            outputShapes.push_back(shape);
        }

        attr = dynamic_cast<ReduceAttr *>(op_attr.get());
        dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
        keepdims = attr->keepdims;

        if (op_idx < 13)
        {
            if (attr->axes.size() == in_[0]->shape.size())
                outputShapes[0] = {};
        }

        for (int idx = 0; idx < output_num; idx++)
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void ReduceOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        switch (op_idx)
        {
        case 9:
        {
            ret = aclnnReduceSumGetWorkspaceSize(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = aclnnReduceSum(workspaceAddr, workspaceSize, executor, aclstream);
            break;
        }
        case 10:
        {
            ret = aclnnMeanGetWorkspaceSize(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = aclnnMean(workspaceAddr, workspaceSize, executor, aclstream);
            break;
        }
        case 11:
        {
            ret = aclnnAmaxGetWorkspaceSize(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = aclnnAmax(workspaceAddr, workspaceSize, executor, aclstream);
            break;
        }
        case 12:
        {
            ret = aclnnAminGetWorkspaceSize(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = aclnnAmin(workspaceAddr, workspaceSize, executor, aclstream);
            break;
        }
        default:
        {
            LOGir << "no such reduce!!";
            exit(-1);
        }
        }
        syncRun();
        return;
    }
}