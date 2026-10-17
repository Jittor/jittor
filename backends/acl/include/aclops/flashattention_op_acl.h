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

    class IncreFlashAttentionOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        IncreFlashAttentionOpRunner();
    };

    // Experimental single-request, cache-free prefill parity with
    // torch_npu.npu_fused_infer_attention_score (TND layout).
    class FusedInferPrefillOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FusedInferPrefillOpRunner();
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

}
