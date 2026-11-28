#include "bindings/pyjt/py_matmul.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "type/nano_string.h"

namespace jittor {

DECLARE_FLAG(int, amp_reg);
DECLARE_FLAG(bool, no_grad);

namespace {
auto make_cublas_batched_matmul =
    op_constructor<VarPtr, Var*, Var*, bool, bool, float>("cublas_batched_matmul");

struct MatmulPrecisionScope {
    int saved = amp_reg;
    explicit MatmulPrecisionScope(bool half_operands) {
        // Half operands keep their result dtype while the generic reduction
        // accumulates in float32, as in matrix._contraction_scope.
        if (half_operands) amp_reg &= ~(amp_prefer16 | amp_prefer_bfloat16);
        amp_reg |= amp_keep_reduce;
    }
    ~MatmulPrecisionScope() { amp_reg = saved; }
};
} // namespace

PyObject* matmul_contraction_call(PyObject* body, PyObject* fallback,
                                PyObject* a, PyObject* b) {
    if (!PyObject_TypeCheck(a, &PyjtVarHolder.ht_type)
            || !PyObject_TypeCheck(b, &PyjtVarHolder.ht_type)) {
        return PyObject_CallFunctionObjArgs(fallback, a, b, nullptr);
    }
    Var* av = GET_RAW_PTR(VarHolder, a)->var;
    Var* bv = GET_RAW_PTR(VarHolder, b)->var;
    const auto& as = av->shape;
    const auto& bs = bv->shape;
    bool valid = as.size() >= 2 && as.size() == bs.size()
        && as[as.size()-1] == bs[bs.size()-2];
    if (valid) for (uint i = 0; i + 2 < as.size(); ++i) {
        if (as[i] != bs[i] && as[i] != 1 && bs[i] != 1) {
            valid = false;
            break;
        }
    }
    if (!valid)
        return PyObject_CallFunctionObjArgs(fallback, a, b, nullptr);
    auto ad = av->dtype(), bd = bv->dtype();
    MatmulPrecisionScope scope((ad == ns_float16 || ad == ns_bfloat16)
                               && (bd == ns_float16 || bd == ns_bfloat16));
    // RAII also restores the caller's flags when a provider raises or makes
    // a nested matrix product. No result or execution graph is retained.
    return PyObject_CallFunctionObjArgs(body, a, b, nullptr);
}

PyObject* cublas_matmul_matching_batch(PyObject* a, PyObject* b,
                                      bool trans_a, bool trans_b, float alpha) {
    if (!no_grad || accelerator_backend_id() != BackendId::Cuda
            || !has_op("cublas_batched_matmul")
            || !PyObject_TypeCheck(a, &PyjtVarHolder.ht_type)
            || !PyObject_TypeCheck(b, &PyjtVarHolder.ht_type))
        Py_RETURN_NONE;
    auto* left = GET_RAW_PTR(VarHolder, a);
    auto* right = GET_RAW_PTR(VarHolder, b);
    const auto& as = left->var->shape;
    const auto& bs = right->var->shape;
    if (as.size() < 3 || as.size() != bs.size() || left->var->num < 0 || right->var->num < 0)
        Py_RETURN_NONE;
    for (uint i = 0; i + 2 < as.size(); ++i)
        if (as[i] != bs[i]) Py_RETURN_NONE;
    PyObject* values[] = {a, b};
    PyTensorFrontendScope scope(nullptr, values, 2);
    auto* av = left->grad_input();
    auto* bv = right->grad_input();
    return to_py_object<VarHolder*>(
        new VarHolder(make_cublas_batched_matmul(av, bv, trans_a, trans_b, alpha)));
}

} // namespace jittor
