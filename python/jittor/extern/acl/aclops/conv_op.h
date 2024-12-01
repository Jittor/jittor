#pragma once
#include "utils.h"

namespace jittor
{
    struct ConvOpRunner : public BaseOpRunner
    {
        ConvOpRunner() : BaseOpRunner("Conv2d")
        {
        }

    protected:
        bool use_nchw = true;
        bool is_group_op = false;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override
        {
            // for conv
            aclIntArray *strides = nullptr;
            aclIntArray *pads = nullptr;
            aclIntArray *outPads = nullptr;
            aclIntArray *dilations = nullptr;

            auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
            strides = aclCreateIntArray(attr->convStrides.data(), 2);
            pads = aclCreateIntArray(attr->convPads.data(), 2);
            outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
            dilations = aclCreateIntArray(attr->convDilations.data(), 2);

            aclTensor *bias = nullptr;

            auto input_num = in_.size();
            if (input_num == 3)
                bias = inputTensors[2];

            ret = it->second.getWorkspaceSizeFuncConv(inputTensors[0], inputTensors[1], bias, strides, pads, dilations, false, outPads, attr->group, outputTensors[0], 0, &workspaceSize, &executor);

            checkRet(ret);

            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);
            
            // syncRun();

            aclDestroyIntArray(strides);
            aclDestroyIntArray(pads);
            aclDestroyIntArray(outPads);
            aclDestroyIntArray(dilations);
            return;
        }
    };
}
