#pragma once
#include "utils.h"
#include "base_op.h"
#include "foreach_op_acl.h"

namespace jittor
{
    // The whole parameter list updated in a fixed number of launches instead of
    // the five elementwise launches per parameter the portable update needs.
    //
    // CANN publishes `aclnnFusedSgd`, which would be one launch, but Ascend950
    // ships no kernel for it: the operator is registered in the opp package for
    // ascend910b and ascend910_93 only, and the query fails with
    // "the JSON configuration file of operator aclnnFusedSgd_0_FusedSgd cannot
    // be found". The foreach family does have Ascend950 kernels, and SGD with
    // momentum is three of them (four with weight decay, six with Nesterov).
    class FusedSgdOpRunner : public ForeachOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FusedSgdOpRunner();
    };
}
