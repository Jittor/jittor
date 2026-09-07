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
        launch(ret, aclnnClampTensor, true);
    }
}
