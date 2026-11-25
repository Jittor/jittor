#pragma once
#include "core/op.h"

namespace jittor {

// Native C++ TVM-FFI C ABI kernel launch operator for TileLang (and any
// TVM-FFI compiled executable): builds stack-allocated DLTensors directly from
// executor Vars, sets TVM-FFI's thread-local stream to Jittor's compute stream,
// and calls TVMFFIFunctionCall with zero Python callback, zero GIL, async-worker
// eligibility, and full CUDA Graph capture compatibility.
struct TvmFfiLaunchOp : Op {
    static constexpr bool accepts_storage_strides = true;

    enum ParamKind : int {
        kInputTensor = 0,
        kOutputTensor = 1,
        kInt = 2,
        kBool = 3,
        kFloat = 4,
        kNone = 5,
    };

    vector<Var*> read_buffers;
    vector<Var*> write_buffers;
    uint64 func_handle = 0;
    uint64 mod_handle = 0;
    uint64 fn_call = 0;
    uint64 fn_set_stream = 0;
    uint64 fn_error_move = 0;
    uint64 fn_incref = 0;
    uint64 fn_decref = 0;
    int expected_device = 0;
    bool handles_owned = false;
    vector<int> param_kinds;
    vector<uint64> param_values;
    vector<int> bound_out_indices;
    vector<int> bound_in_indices;
    bool order_bound_outputs = false;

    // @attrs(multiple_outputs)
    TvmFfiLaunchOp(
        vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes,
        vector<Var*>&& inputs,
        vector<uint64>&& ffi_handles,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values);

    // @attrs(multiple_outputs,replace_outputs)
    TvmFfiLaunchOp(
        vector<Var*>&& inputs,
        vector<Var*>&& outputs,
        vector<uint64>&& ffi_handles,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& bound_pairs);

    ~TvmFfiLaunchOp() override;

    const char* name() const override { return "tvm_ffi_launch"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var*, Var*, Var*, int) override;

private:
    void parse_validate_and_retain(const vector<uint64>& ffi_handles,
                                   const vector<int>& bound_pairs);
};

} // namespace jittor
