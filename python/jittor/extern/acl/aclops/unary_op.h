#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct UnaryOpRunner : public BaseOpRunner
    {
        UnaryOpRunner() : BaseOpRunner("unary")
        {
        }

    protected:
        bool is_group_op = true;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override
        {
            if (name == "Cast")
                ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            else
                ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);

            checkRet(ret);

            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }

            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);
            syncRun();
            return;
        }
    };
}
