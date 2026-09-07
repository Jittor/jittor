#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class EmbeddingOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        EmbeddingOpRunner();
    };

    class EmbeddingBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        EmbeddingBackwardOpRunner();
    };

}