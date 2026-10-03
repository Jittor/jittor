#include <aclnnop/aclnn_sort.h>
#include "sort_op_acl.h"

namespace jittor {
SortOpRunner::SortOpRunner(int64_t dim, bool descending)
    : BaseOpRunner("Sort"), dim(dim), descending(descending) {
    use_nchw = false;
}

void SortOpRunner::executeOp(AclOpRegistry::const_iterator &) {
    // ArgsortOp outputs (indices, values); aclnnSort takes (values, indices).
    ret = aclnnSortGetWorkspaceSize(inputTensors[0], false, dim, descending,
                                    outputTensors[1], outputTensors[0],
                                    &workspaceSize, &executor);
    launch(ret, aclnnSort, true);
}
}
