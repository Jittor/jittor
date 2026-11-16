#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct UnaryOpRunner : public BaseOpRunner
    {
        UnaryOpRunner();
        void setupInputDesc() override;
        void setupOutputDesc() override;

    protected:
        bool stridedHighRankCast() const;
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
