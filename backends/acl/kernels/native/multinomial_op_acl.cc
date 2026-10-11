#include <acl/acl.h>
#include "core/var.h"
#include "acl_jittor.h"
#include "aclnnop/level2/aclnn_multinomial.h"
#include "multinomial_op_acl.h"
#include "runtime/rng_state.h"

namespace jittor {

MultinomialOpRunner::MultinomialOpRunner(int64_t count, bool replace)
    : BaseOpRunner("Multinomial"), num_samples(count), replacement(replace) {
    use_nchw = false;
}
void MultinomialOpRunner::executeOp(AclOpRegistry::const_iterator &it) {
    // One CANN draw consumes twelve counters from the output device stream.
    const auto rng = reserve_acl_random(out_[0]->device_id, 12);
    ret = aclnnMultinomialGetWorkspaceSize(
        inputTensors[0], num_samples, replacement, rng.seed,
        rng.offset, outputTensors[0], &workspaceSize, &executor);
    launch(ret, aclnnMultinomial, true);
}
}
