#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class FlashAttentionOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        FlashAttentionOpRunner();
    };

    class FlashAttentionBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        FlashAttentionBackwardOpRunner();
    };

    class IncreFlashAttentionOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        IncreFlashAttentionOpRunner();
    };

    class KVCacheMemcpyOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        KVCacheMemcpyOpRunner();
    };

}
