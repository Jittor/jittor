#pragma once
#include "core/op.h"

namespace jittor {

// Native CUDA driver kernel launch (CUfunction via cuLaunchKernel) as a
// first-class graph operator: zero Python callback, zero GIL, async-worker
// eligible, and natively capturable into CUDA Graphs on Jittor's compute stream.
struct CudaLaunchOp : Op {
    static constexpr bool accepts_storage_strides = true;

    enum ParamKind : int {
        kInputPtr = 0,
        kOutputPtr = 1,
        kScratchPtr = 2,
        kProfileScratchPtr = 3,
        kNullPtr = 4,
        kScalar32 = 5,
        kScalar64 = 6,
    };

    vector<Var*> read_buffers;
    vector<Var*> write_buffers;
    uint64 function_handle = 0;
    uint64 launcher_handle = 0;
    uint64 ctx_handle = 0;
    uint gx = 1;
    uint gy = 1;
    uint gz = 1;
    uint bx = 1;
    uint by = 1;
    uint bz = 1;
    uint shared_mem = 0;
    uint64 scratch_bytes = 0;
    uint64 scratch_align = 1;
    uint64 profile_scratch_bytes = 0;
    uint64 profile_scratch_align = 1;
    int expected_device = 0;
    vector<int> param_kinds;
    vector<uint64> param_values;
    vector<int> aligned_slots;
    vector<int> copy_dst_outs;
    vector<int> copy_src_ins;
    vector<int> bound_out_indices;
    vector<int> bound_in_indices;
    bool order_bound_outputs = false;

    // @attrs(multiple_outputs)
    CudaLaunchOp(
        vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes,
        vector<Var*>&& inputs,
        vector<uint64>&& launch_config,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& extra_meta);

    // @attrs(multiple_outputs,replace_outputs)
    CudaLaunchOp(
        vector<Var*>&& inputs,
        vector<Var*>&& outputs,
        vector<uint64>&& launch_config,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& extra_meta);

    const char* name() const override { return "cuda_launch"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var*, Var*, Var*, int) override;

private:
    void parse_and_validate(const vector<uint64>& launch_config,
                            const vector<int>& extra_meta);
};

} // namespace jittor
