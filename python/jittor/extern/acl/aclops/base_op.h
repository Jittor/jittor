#pragma once
#include "utils.h"
#include "acl_jittor.h"

namespace jittor
{
    extern int sync_run;
    class BaseOpRunner
    {
    protected:
        vector<Var *> in_;
        vector<Var *> out_;

        int ret = -1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        bool is_group_op = false;

        std::vector<std::vector<int64_t>> inputShapes;
        std::vector<std::vector<int64_t>> outputShapes;

        std::vector<aclTensor *> inputTensors;
        std::vector<aclTensor *> outputTensors;

    public:
        string name;
        string jt_name;
        std::unique_ptr<AclOpAttr> op_attr;
        bool use_nchw = false;
        // aclnn cubeMathType for matmul/bmm: 0 = KEEP_DTYPE (full fp32, matches torch,
        // default), 1 = ALLOW_FP32_DOWN_PRECISION (HF32, faster, ~5e-4 off). Set from the
        // op's cuda_src per jt.acl_allow_hf32. Only matmul/bmm executeOp read it.
        int cube_math_type = 0;

        BaseOpRunner(const string &name = "") : name(name) {}
        virtual ~BaseOpRunner() = default;

        // Common functionality for adding input/output variables
        void add(Var *v, bool is_input);

        virtual void setupInputDesc();

        void cleanupDesc();

        virtual void setupOutputDesc();

        virtual void syncRun();

        void checkRet(aclnnStatus ret);

        // Base run method with common operator lookup logic
        void run();

    protected:
        // Virtual method for specific operator execution
        virtual void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) = 0;
        void cleanupAttr();
    };

}
