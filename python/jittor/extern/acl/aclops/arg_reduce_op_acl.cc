#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "arg_reduce_op_acl.h"
#include "var.h"

namespace jittor
{
    ArgReduceOpRunner::ArgReduceOpRunner(bool is_max, int64_t dim, bool keepdims)
        : BaseOpRunner(is_max ? "MaxDim" : "MinDim"),
          is_max(is_max),
          dim(dim),
          keepdims(keepdims)
    {
        use_nchw = false;
    }

    void ArgReduceOpRunner::setupOutputDesc()
    {
        auto output_num = out_.size();
        for (int output_idx = 0; output_idx < output_num; output_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < out_[output_idx]->shape.size(); j++)
                shape.push_back(out_[output_idx]->shape[j]);

            // Jittor represents scalar reductions as shape (1,), while ACL's
            // non-keepdim 1-D reduction contract requires a scalar descriptor.
            if (!keepdims && in_[0]->shape.size() == 1)
                shape.clear();
            outputShapes.push_back(shape);
        }

        for (int idx = 0; idx < output_num; idx++)
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(
                outputShapes[idx],
                out_[idx]->mem_ptr,
                out_[idx]->size,
                get_dtype(out_[idx]->dtype()),
                &outputTensors[idx],
                use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void ArgReduceOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        if (is_max)
        {
            ret = aclnnMaxDimGetWorkspaceSize(
                inputTensors[0],
                dim,
                keepdims,
                outputTensors[1],
                outputTensors[0],
                &workspaceSize,
                &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMaxDimGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
        }
        else
        {
            ret = aclnnMinDimGetWorkspaceSize(
                inputTensors[0],
                dim,
                keepdims,
                outputTensors[1],
                outputTensors[0],
                &workspaceSize,
                &executor);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMinDimGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
        }

        if (workspaceSize > 0)
            mallocWorkSpace(workspaceSize);
        ret = is_max
            ? aclnnMaxDim(workspaceAddr, workspaceSize, executor, aclstream)
            : aclnnMinDim(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s execution failed. ERROR: %d\n", name.c_str(), ret); return);
        syncRun();
    }
}
