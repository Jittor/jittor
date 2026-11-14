// Native construction of the existing CUDA FP32 attention backward.
// Program strings are immutable metadata; every invocation owns fresh graph nodes.

#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "ops/composite/code_op.h"
#include "ops/op_register.h"
#include "core/var.h"
#include "mem/swap.h"
namespace jittor {
namespace {
static auto product = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto sum_axis = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");
static auto transpose_axes = op_constructor<VarPtr, Var*, NanoVector>("transpose");
static auto cast_type = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto code_one = op_constructor<VarPtr, NanoVector, NanoString,
    vector<Var*>&&, string&&, vector<string>&&, string&&, string&&,
    vector<string>&&, string&&, DataMap&&, string&&>("code");
static auto code_many = op_constructor<vector<VarPtr>, vector<NanoVector>&&,
    vector<NanoString>&&, vector<Var*>&&, string&&, vector<string>&&, string&&,
    string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");

struct FusedAttentionF32VjpOp : CodeOp {
    string row_source, backward_source;
    bool sequence_major;
    FusedAttentionF32VjpOp(vector<Var*>&& ins, vector<NanoVector>&& shapes,
                          string&& header, string&& forward, string&& row,
                          string&& backward, bool sequence_major)
        : CodeOp(move(shapes), vector<NanoString>{ns_float32, ns_float32},
                 move(ins), "", {}, "", move(forward),
                 vector<string>{"#error Native attention gradients require their original C++ owner"},
                 move(header), DataMap{{"multi_grad", 1}}, "cuda"),
          row_source(move(row)), backward_source(move(backward)),
          sequence_major(sequence_major) {}

    void grads(Var** douts, VarPtr* dins) override {
        if (!douts[0]) return;
        VarPtr dout(douts[0]);
        if (dout->dtype() != ns_float32) dout = cast_type(dout, ns_float32);
        // The same float32 dO.O reduction and float64 conversion as the
        // Python owner. Keep both the saved output and this VJP's cotangent.
        auto multiplied = product(dout, _outputs[0], ns_multiply);
        auto approximate = sum_axis(multiplied, ns_add, NanoVector{-1}, false);
        if (sequence_major)
            approximate = transpose_axes(approximate, NanoVector{0, 2, 1});
        approximate = cast_type(approximate, ns_float64);
        approximate = contiguous_storage(approximate);
        auto* query = _inputs[0];
        NanoVector row_shape{query->shape[0], query->shape[sequence_major ? 2 : 1],
                             query->shape[sequence_major ? 1 : 2]};
        vector<Var*> row_inputs{query, _inputs[1], _inputs[2], dout,
                                _outputs[1], approximate};
        row_inputs.insert(row_inputs.end(), _inputs.begin() + 3, _inputs.end());
        auto delta = code_one(row_shape, ns_float64, move(row_inputs), "", {}, "",
                              string(row_source), {}, string(cuda_header), {}, "cuda");
        vector<Var*> backward_inputs{query, _inputs[1], _inputs[2], dout, _outputs[1], delta};
        backward_inputs.insert(backward_inputs.end(), _inputs.begin() + 3, _inputs.end());
        vector<NanoVector> shapes{query->shape, _inputs[1]->shape, _inputs[2]->shape};
        auto result = code_many(move(shapes),
            vector<NanoString>{ns_float32, ns_float32, ns_float32}, move(backward_inputs),
            "", {}, "", string(backward_source), {}, string(cuda_header), {}, "cuda");
        for (int i = 0; i < 3; ++i) dins[i] = move(result[i]);
    }
};
} // namespace

// @pyjt(attention_f32_native_vjp_enabled)
bool attention_f32_native_vjp_enabled() {
    return !save_mem;
}

// @pyjt(attention_f32_with_native_vjp)
VarHolder* attention_f32_with_native_vjp(const vector<VarHolder*>& holders,
        vector<NanoVector> shapes, string header, string forward,
        string row, string backward, bool sequence_major) {
    USER_CHECK(holders.size() >= 3 && shapes.size() == 2)
        << "native FP32 attention needs Q/K/V and two output shapes";
    vector<Var*> inputs;
    for (auto* holder : holders) {
        holder->observe_alias_history();
        inputs.push_back(holder->var);
    }
    // Match the generated CodeOp constructor boundary, including strided
    // slices and masks. Keep new contiguous nodes alive until the Op owns
    // their input edges.
    vector<VarPtr> contiguous_owners;
    adapt_storage_input<CodeOp>(inputs, contiguous_owners);
    for (int i = 0; i < 3; ++i)
        USER_CHECK(inputs[i]->shape.size() == 4 && inputs[i]->dtype() == ns_float32)
            << "native FP32 attention needs four-dimensional float32 inputs";
    USER_CHECK(shapes[0] == inputs[0]->shape)
        << "native FP32 attention input shape changed during construction";
    NanoVector normalizer{inputs[0]->shape[0], inputs[0]->shape[sequence_major ? 2 : 1],
                          inputs[0]->shape[sequence_major ? 1 : 2], 2};
    USER_CHECK(shapes[1] == normalizer)
        << "native FP32 attention normalizer shape does not match its input";
    auto* op = new FusedAttentionF32VjpOp(move(inputs), move(shapes), move(header),
                                         move(forward), move(row), move(backward), sequence_major);
    vector<VarPtr> outputs;
    for (auto& held : op->outputs_holder) outputs.push_back(move(held));
    for (auto& value : outputs) value->set_inputs({op});
    op->add_inputs(op->_inputs);
    op->init();
    return new VarHolder(move(outputs[0]));
}
} // namespace jittor
