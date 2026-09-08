#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class RotaryPositionEmbeddingOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        RotaryPositionEmbeddingOpRunner();
    };

    class RotaryPositionEmbeddingGradOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        RotaryPositionEmbeddingGradOpRunner();
    };

}
