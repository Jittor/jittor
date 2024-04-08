
#include "pyjt/py_obj_holder.h"
#include "utils/str_utils.h"
#include "jtorch_core.h"
#include "graph.h"
#include "grad.h"
#include "ops/op_register.h"

namespace jittor {

void pyjt_def_all(PyObject* m);

EXTERN_LIB void setter_use_cuda(int value);

Device::Device(const string& name, int ordinal) : name(name) {
    if (startswith(name, "cpu"))
        setter_use_cuda(0);
    else
        setter_use_cuda(1);
}

unordered_map<int64, VarPtr> grad_backup;
EXTERN_LIB void (*_var_free_hook)(Var*);
EXTERN_LIB unordered_map<int64, VarPtr>* _grad_backup_ptr;

void jtorch_var_free_hook(Var* v) {
    auto iter = grad_backup.find(v->id);
    if (iter != grad_backup.end()) {
        grad_backup.erase(iter);
    }
}

void jtorch_init() {
    _var_free_hook = &jtorch_var_free_hook;
    _grad_backup_ptr = &grad_backup;
}

inline static VarPtr& get_grad(Var* v) {
    return grad_backup[v->id];
}
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();

inline static void add_grad(VarPtr& a, VarPtr&& b) {
    if (!a) a = move(b);
    else {
        a = make_binary(a, b, ns_add);
    }
}


void grad_set(VarHolder* x, Maybe<VarHolder> v) {
    if (!v) {
        grad_del(x);
        return;
    }
    grad_backup[x->var->id] = v.ptr->var;
}

Maybe<VarHolder> grad_get(VarHolder* x) {
    auto iter = grad_backup.find(x->var->id);
    if (iter != grad_backup.end()) {
        if (!iter->second.ptr) return nullptr;
        return new VarHolder(iter->second.ptr);
    }
    return nullptr;
}

void grad_del(VarHolder* x) {
    auto iter = grad_backup.find(x->var->id);
    if (iter != grad_backup.end())
        grad_backup.erase(iter);
}

void backward(VarHolder* x) {
    vector<Node*> gnodes({x->var});
    bfs_backward(gnodes, [&](Node* node) {
        if (node->is_stop_grad())
            return false;
        return true;
    });
    vector<Var*> targets;
    for (auto* node : gnodes) {
        if (node->is_var() && node->flags.get(NodeFlags::_th_require_grad))
            targets.push_back(node->var());
    }
    auto grads = grad(x->var, targets);
    for (int i=0; i<targets.size(); i++) {
        auto& gptr = get_grad(targets[i]);
        add_grad(gptr, move(grads[i]));
    }
}


}

static void init_module(PyModuleDef* mdef, PyObject* m) {
    jittor::jtorch_init();
    mdef->m_doc = "Inner c++ core of jtorch";
    jittor::pyjt_def_all(m);
}
PYJT_MODULE_INIT(jtorch_core);
