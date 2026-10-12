#include <acl/acl.h>
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "aclnnop/level2/aclnn_sort.h"
#include "sort_op_acl.h"

namespace jittor {
SortOpRunner::SortOpRunner(bool stable, int64_t dim, bool descending)
    : BaseOpRunner("Sort"), stable(stable), dim(dim), descending(descending) {
    use_nchw = false;
}
void SortOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    ret = aclnnSortGetWorkspaceSize(
        inputTensors[0], stable, dim, descending,
        outputTensors[0], outputTensors[1], &workspaceSize, &executor);
    launch(ret, aclnnSort, true);
}
}
