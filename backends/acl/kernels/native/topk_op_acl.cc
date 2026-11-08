#include "topk_op_acl.h"
#include "core/var.h"
#include <aclnnop/aclnn_topk.h>

namespace jittor {
TopkOpRunner::TopkOpRunner() : BaseOpRunner("Topk") {}

void TopkOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    USER_CHECK(string(acl_soc_name()).find("Ascend310P") == 0)
        << "ACL Topk inference is validated on Ascend310P only";
    USER_CHECK(inputShapes.size() == 1 && inputShapes[0].size() == 2
            && outputShapes.size() == 2 && outputShapes[0].size() == 2
            && outputShapes[0] == outputShapes[1]
            && outputShapes[0][0] == inputShapes[0][0])
        << "ACL Topk requires rank-two input and matching output geometry";
    int64_t k = outputShapes[0][1];
    USER_CHECK(inputShapes[0][0] > 0 && k > 0 && k <= inputShapes[0][1])
        << "ACL Topk requires nonempty input and positive k within its final dimension";
    auto dtype = in_[0]->dtype();
    USER_CHECK((dtype == ns_float16 || dtype == ns_float32)
            && out_[0]->dtype() == dtype && out_[1]->dtype() == ns_int64)
        << "ACL Topk requires float16/float32 values and int64 indices";
    ret = aclnnTopkGetWorkspaceSize(inputTensors[0], k, -1, true, true,
        outputTensors[0], outputTensors[1], &workspaceSize, &executor);
    launch(ret, aclnnTopk, true);
}
}
