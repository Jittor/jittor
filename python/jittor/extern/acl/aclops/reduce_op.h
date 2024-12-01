#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ReduceOpRunner : public BaseOpRunner
    {
        int op_idx; // Specific to reduce operations

        ReduceOpRunner() : BaseOpRunner("reduce")
        {
        }

    protected:
        bool is_group_op = false;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
        {
            auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
            aclIntArray *dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
            bool keepdims = attr->keepdims;

            if (op_idx < 13)
            {
                if (attr->axes.size() == in_[0]->shape.size())
                    outputShapes[0] = {};
            }

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
            // syncRun();
            return;
        }
    };
}
