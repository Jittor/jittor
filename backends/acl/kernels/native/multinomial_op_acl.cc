#include <acl/acl.h>
#include "acl_jittor.h"
#include "aclnnop/level2/aclnn_multinomial.h"
#include "multinomial_op_acl.h"

namespace jittor {
extern int current_seed;
extern int64 current_offset;

MultinomialOpRunner::MultinomialOpRunner(int64_t count, bool replace)
    : BaseOpRunner("Multinomial"), num_samples(count), replacement(replace) {
    use_nchw = false;
}
void MultinomialOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    ret = aclnnMultinomialGetWorkspaceSize(
        inputTensors[0], num_samples, replacement, current_seed,
        current_offset, outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnMultinomial, true);
    // torch_npu 2.7.1 advances its NPU generator by 12 for one call,
    // independent of the number of categories (verified on 910B3).
    current_offset += 12;
}
}
