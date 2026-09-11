#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    // One aclnnFusedSgd launch for the whole parameter list, instead of the
    // five elementwise launches per parameter the portable update needs.
    class FusedSgdOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FusedSgdOpRunner();
    };
}
