#include <acl/acl.h>
#include "acl_jittor.h"
#include "aclnnop/level2/aclnn_multinomial.h"
#include "multinomial_op_acl.h"
#include "core/var.h"
#include "runtime/rng_state.h"

namespace jittor {

MultinomialOpRunner::MultinomialOpRunner(int64_t count, bool replace)
    : BaseOpRunner("Multinomial"), num_samples(count), replacement(replace) {
    use_nchw = false;
}
void MultinomialOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    // torch_npu 2.7.1 advances its NPU generator by 12 for one call,
    // independent of the number of categories (verified on 910B3).
    const auto rng = reserve_acl_random(out_[0]->device_id, 12);
    ret = aclnnMultinomialGetWorkspaceSize(
        inputTensors[0], num_samples, replacement, rng.seed,
        rng.offset, outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnMultinomial, true);
}
}
