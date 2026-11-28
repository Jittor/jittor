#include "bindings/pyjt/py_matmul.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "core/var_holder.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
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

PyObject* cublas_matmul_frontend(PyObject* a, PyObject* b,
                                PyObject* tensor_type, PyObject* provider) {
    if (!no_grad || amp_reg || graph_capture_active()
            || reinterpret_cast<PyObject*>(Py_TYPE(a)) != tensor_type
            || reinterpret_cast<PyObject*>(Py_TYPE(b)) != tensor_type
            || !PyObject_TypeCheck(a, &PyjtVarHolder.ht_type)
            || accelerator_backend_id() != BackendId::Cuda
            || !has_op("cublas_batched_matmul"))
        Py_RETURN_NONE;
    auto* left = GET_RAW_PTR(VarHolder, a);
    auto* right = GET_RAW_PTR(VarHolder, b);
    Var* av = left->var;
    Var* bv = right->var;
    if (av->dtype() != ns_float32 || bv->dtype() != ns_float32
            || !av->placement.explicit_backend || av->placement.device.backend != BackendId::Cuda
            || av->placement != bv->placement || av->num <= 0 || bv->num <= 0)
        Py_RETURN_NONE;
    const auto& as = av->shape;
    const auto& bs = bv->shape;
    const auto rank = as.size();
    if (rank < 3 || rank != bs.size() || as[rank - 1] != bs[rank - 2]) Py_RETURN_NONE;
    for (int i = 0; i + 2 < rank; ++i)
        if (as[i] != bs[i]) Py_RETURN_NONE;
    if (!kernel_selects_cublas("batched_matmul", provider, ns_float32)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_NONE;
    }

    // Preserve the standard relay's interpretation of last-two-axis views
    // and the frontend's precision and result type.
    PyTensorFrontendScope frontend(Py_TYPE(a));
    MatmulPrecisionScope precision(false);
    unique_ptr<VarHolder> a_base, b_base;
    if (left->is_last2_transpose_view()) a_base.reset(left->transpose_view_base());
    if (right->is_last2_transpose_view()) b_base.reset(right->transpose_view_base());
    av = (a_base ? a_base.get() : left)->grad_input();
    bv = (b_base ? b_base.get() : right)->grad_input();
    return to_py_object<VarHolder*>(new VarHolder(
        make_cublas_batched_matmul(av, bv, bool(a_base), bool(b_base), 1.0f)));
}

} // namespace jittor
