#pragma once
#include "utils.h"
#include "acl_jittor.h"
#include "runtime/device_state.h"

namespace jittor
{
    using AclExecuteLauncher = std::function<aclnnStatus(
        void *, uint64_t, aclOpExecutor *, aclrtStream)>;

    // A runner is a fresh stack object for every operator, so its six working
    // vectors used to allocate and release their buffers once per launch. The
    // buffers are leased from a per-thread pool instead. A lease is held for
    // the runner's lifetime and the pool hands out a distinct block per lease,
    // so a runner constructed inside another runner's executeOp gets its own.
    struct AclRunnerScratch
    {
        vector<Var *> in_;
        vector<Var *> out_;
        std::vector<std::vector<int64_t>> inputShapes;
        std::vector<std::vector<int64_t>> outputShapes;
        std::vector<aclTensor *> inputTensors;
        std::vector<aclTensor *> outputTensors;
        // Recycled aclTensor descriptors. Survives reset(): it is a pool, not
        // per-operator state.
        std::vector<aclTensor *> descriptors;

        void reset()
        {
            in_.clear();
            out_.clear();
            inputShapes.clear();
            outputShapes.clear();
            inputTensors.clear();
            outputTensors.clear();
        }
    };

    AclRunnerScratch *acl_scratch_acquire();
    void acl_scratch_release(AclRunnerScratch *scratch) noexcept;

    class BaseOpRunner
    {
    protected:
        AclRunnerScratch *scratch;
        vector<Var *> &in_;
        vector<Var *> &out_;

        int ret = -1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        bool is_group_op = false;

        std::vector<std::vector<int64_t>> &inputShapes;
        std::vector<std::vector<int64_t>> &outputShapes;

        std::vector<aclTensor *> &inputTensors;
        std::vector<aclTensor *> &outputTensors;

    public:
        string name;
        string jt_name;
        std::unique_ptr<AclOpAttr> op_attr;
        bool use_nchw = false;

        BaseOpRunner(const string &name = "")
            : scratch(acl_scratch_acquire()),
              in_(scratch->in_), out_(scratch->out_),
              inputShapes(scratch->inputShapes), outputShapes(scratch->outputShapes),
              inputTensors(scratch->inputTensors), outputTensors(scratch->outputTensors),
              name(name) {}
        BaseOpRunner(const BaseOpRunner &) = delete;
        BaseOpRunner &operator=(const BaseOpRunner &) = delete;
        virtual ~BaseOpRunner() { acl_scratch_release(scratch); }

        // Common functionality for adding input/output variables
        void add(Var *v, bool is_input);

        virtual void setupInputDesc();

        void cleanupDesc();

        virtual void setupOutputDesc();

        virtual void syncRun();

        void checkRet(aclnnStatus ret);

        // Shared tail for registry-backed runners. The caller still owns the
        // typed workspace query; this method owns allocation, launch errors,
        // and the optional diagnostic synchronization policy.
        void launch(aclnnStatus workspace_ret,
                    const AclExecuteLauncher &launcher,
                    bool synchronize = true);

        // Base run method with common operator lookup logic
        void run();

    protected:
        // Virtual method for specific operator execution
        virtual void executeOp(AclOpRegistry::const_iterator &it) = 0;
        void cleanupAttr();
    };

}
