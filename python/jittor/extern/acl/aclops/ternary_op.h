#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct TernaryOpRunner : public BaseOpRunner
    {
        TernaryOpRunner() : BaseOpRunner("ternary")
        {
        }

    protected:
        bool is_group_op = false;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override
        {
            ret = aclnnSWhereGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);

            checkRet(ret);

            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }

            ret = aclnnSWhere(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);
            syncRun();
            return;
        }
    };
}
