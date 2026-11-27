#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class FlashAttentionOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FlashAttentionOpRunner();
    };

    class FlashAttentionBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FlashAttentionBackwardOpRunner();
    };

    class PromptFlashAttentionOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        PromptFlashAttentionOpRunner();
    };

    class IncreFlashAttentionOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        IncreFlashAttentionOpRunner();
    };

    class KVCacheMemcpyOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        KVCacheMemcpyOpRunner();
        // This runner issues runtime copies, not a registered aclnn launcher.
        void run();
    };

    // Direct C++ inference runner for MappedFlashAttentionOp (BNSD), avoiding
    // CodeOp attribute parsing, dummy input tensors, and per-call dict keys.
    class DirectFlashAttentionOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        double scale = 1.0;
        int64_t head_num = 0;
        int64_t sparse_mode = 0;
        bool has_mask = false;
        bool mask_is_realshift = false;
        DirectFlashAttentionOpRunner();
    };

}
