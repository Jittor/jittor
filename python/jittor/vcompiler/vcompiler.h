#include "common.h"
#include "var_holder.h"
#include "pyjt/py_converter.h"
#include "fused_op.h"

namespace jittor {

struct ShapeKey {
    vector<NanoVector> shapes;
};

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey& key) const {
        std::size_t h = 0;
        for (int i=0; i<key.shapes.size(); i++) {
            h = h * 10007 + key.shapes[i].data * 3 + 
                key.shapes[i].offset * 5;
        }
        return h;
    }
};

struct ShapeKeyEqual {
    bool operator()(const ShapeKey& lhs, const ShapeKey& rhs) const {
        return lhs.shapes == rhs.shapes;
    }
};

struct ShapeValue {
    vector<uint64> values;
};

// unordered_map<ShapeKey, ShapeValue, ShapeKeyHash> shape_values;

struct SGraph {
    vector<Var*> outputs;
    vector<Var*> inputs;
    vector<Node*> bfs_q;
    unordered_map<Var*,pair<Var*,uint64>> share_map;
    vector<char> flags;
    
    vector<FusedOp> fused_ops;
    vector<Op*> rid_ops;
    vector<int> v_last_rid;

    std::unordered_map<ShapeKey, ShapeValue, ShapeKeyHash, ShapeKeyEqual> shape_values;
    int shape_value_len;
};

// @pyjt(SGraphPtr)
struct SGraphPtr {
    std::unique_ptr<SGraph> ptr;
};

// SGraphPtr
struct SGraphPtr;
EXTERN_LIB PyTypeObject PyjtSGraphPtr;
DEF_IS(SGraphPtr, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtSGraphPtr;
}
DEF_IS(SGraphPtr*, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtSGraphPtr;
}


DEF_IS(SGraphPtr, PyObject*) to_py_object(T&& a) {
    PyObjHolder obj(_PyObject_New(&PyjtSGraphPtr));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T();
    ptr->ptr = std::move(a.ptr);
    return obj.release();
}

DEF_IS(SGraphPtr, const T&) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(T, obj);
}

DEF_IS(SGraphPtr*, T) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(typename std::remove_pointer<T>::type, obj);
}

// @pyjt(build_sgraph)
SGraphPtr build_sgraph(const vector<VarHolder*>& outputs, const vector<VarHolder*>& inputs);

// @pyjt(prob_sgraph)
bool prob_sgraph(SGraphPtr* sgraph, const vector<VarHolder*>& inputs);

// @pyjt(merge_sgraph)
void merge_sgraph(SGraphPtr* sgraph, SGraphPtr* sgraph2);

// @pyjt(exec_sgraph)
vector<VarHolder*> exec_sgraph(SGraphPtr* sgraph, const vector<VarHolder*>& inputs);

// @pyjt(delay_fetch)
vector<VarHolder*> delay_fetch(const vector<VarHolder*>& inputs);

}