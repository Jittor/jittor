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
        ReduceAttr *attr;
        aclIntArray *dim;
        bool keepdims;
        void setupOutputDesc() override
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

        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
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
    };
}
