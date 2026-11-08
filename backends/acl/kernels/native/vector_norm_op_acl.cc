#include "vector_norm_op_acl.h"
#include "core/var.h"
#include <aclnnop/aclnn_linalg_vector_norm.h>

namespace jittor {
LinalgVectorNormOpRunner::LinalgVectorNormOpRunner() : BaseOpRunner("LinalgVectorNorm") {}

void LinalgVectorNormOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    USER_CHECK(string(acl_soc_name()).find("Ascend310P") == 0)
        << "ACL vector norm inference is validated on Ascend310P only";
    USER_CHECK(inputShapes.size() == 1 && outputShapes.size() == 1
            && !inputShapes[0].empty() && in_[0]->is_contiguous())
        << "ACL vector norm requires one nonempty dense input and one output";
    auto expected = inputShapes[0];
    for (auto size : expected)
        USER_CHECK(size > 0) << "ACL vector norm requires positive extents";
    bool keepdim = expected.size() == outputShapes[0].size();
    if (keepdim) expected.back() = 1;
    else expected.pop_back();
    USER_CHECK(expected == outputShapes[0])
        << "ACL vector norm output must reduce exactly the final axis";
    auto dtype = in_[0]->dtype();
    USER_CHECK((dtype == ns_float16 || dtype == ns_float32) && out_[0]->dtype() == dtype)
        << "ACL vector norm preserves float16/float32 output dtype";
    float p = 2.0f;
    int64_t axis = -1;
    aclScalar* order = aclCreateScalar(&p, ACL_FLOAT);
    aclIntArray* dims = aclCreateIntArray(&axis, 1);
    ret = aclnnLinalgVectorNormGetWorkspaceSize(inputTensors[0], order, dims, keepdim,
        get_dtype(dtype), outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnLinalgVectorNorm, true);
    aclDestroyIntArray(dims);
    aclDestroyScalar(order);
}
}
