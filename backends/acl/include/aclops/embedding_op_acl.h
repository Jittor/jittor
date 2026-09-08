#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class EmbeddingOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        EmbeddingOpRunner();
    };

    class EmbeddingBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        EmbeddingBackwardOpRunner();
    };

}
