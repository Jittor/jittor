#include "grid_sample_op_acl.h"
#include "core/var.h"
#include <aclnnop/aclnn_grid_sampler2d.h>

namespace jittor {
GridSampler2DOpRunner::GridSampler2DOpRunner()
    : BaseOpRunner("GridSampler2D") {}

void GridSampler2DOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    // Keep both tensors in ND: the input is logical NCHW, but grid is NHW2.
    // This owner is deliberately limited to the measured inference subset.
    USER_CHECK(string(acl_soc_name()).find("Ascend310P") == 0)
        << "ACL grid sampling inference is validated on Ascend310P only";
    USER_CHECK(inputShapes.size() == 2 && outputShapes.size() == 1
            && inputShapes[0].size() == 4 && inputShapes[1].size() == 4)
        << "ACL grid sampling requires rank-four input and grid";
    const auto &x = inputShapes[0];
    const auto &grid = inputShapes[1];
    USER_CHECK(x[0] == grid[0] && grid[3] == 2)
        << "ACL grid sampling requires equal batches and two grid coordinates";
    for (auto size : x) USER_CHECK(size > 0) << "ACL grid sampling requires nonempty input";
    for (auto size : grid) USER_CHECK(size > 0) << "ACL grid sampling requires nonempty grid";
    auto dtype = in_[0]->dtype();
    USER_CHECK((dtype == ns_float16 || dtype == ns_float32)
            && in_[1]->dtype() == dtype && out_[0]->dtype() == dtype)
        << "ACL grid sampling requires matching float16 or float32 tensors";
    USER_CHECK(outputShapes[0] == vector<int64_t>({x[0], x[1], grid[1], grid[2]}))
        << "ACL grid sampling output shape does not match input and grid";
    ret = aclnnGridSampler2DGetWorkspaceSize(
        inputTensors[0], inputTensors[1], 0, 0, false,
        outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnGridSampler2D, true);
}
}
