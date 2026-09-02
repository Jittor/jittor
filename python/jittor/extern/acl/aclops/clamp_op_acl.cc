#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "clamp_op_acl.h"

namespace jittor
{
    ClampTensorOpRunner::ClampTensorOpRunner() : BaseOpRunner("ClampTensor")
    {
    }

    void ClampTensorOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        (void)it;
        ret = aclnnClampTensorGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            outputTensors[0], &workspaceSize, &executor);
        checkRet(ret);

        if (workspaceSize > 0)
            mallocWorkSpace(workspaceSize);

        ret = aclnnClampTensor(
            workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("%s: aclnnClampTensor failed. ERROR: %d\n", name.c_str(), ret); return);
        syncRun();
    }
}
