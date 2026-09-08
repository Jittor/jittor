#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class RollOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        vector<int64_t> shifts;
        vector<int64_t> dims;
        RollOpRunner();
    };
}
