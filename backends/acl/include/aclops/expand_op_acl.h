#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ExpandOpRunner : public BaseOpRunner
    {
        ExpandOpRunner();

    protected:
        bool allowsFlattenedElementwise() const override { return true; }
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
