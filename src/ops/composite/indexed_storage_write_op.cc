#include <algorithm>
#include <cstdint>
#include "ops/composite/indexed_storage_write_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include "ops/op_register.h"
#ifdef JIT_cuda
#include "ops/composite/indexed_storage_write_cuda.h"
#endif
namespace jittor {
#ifndef JIT
DECLARE_FLAG(bool, no_grad);
IndexedStorageWriteOp::IndexedStorageWriteOp(Var* storage_root, Var* x, Var* values,
        vector<Var*>&& indices, NanoVector axes)
    : x(x), values(values), storage_root(storage_root), indices(indices), axes(axes) {
    USER_CHECK(construction_target_backend(x) == BackendId::Cuda) << "CUDA storage write required";
    USER_CHECK(no_grad) << "indexed storage writes require no_grad";
    USER_CHECK(x->dtype() == ns_float32 && values->dtype() == ns_float32);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    out = create_output(nullptr, storage_root->dtype());
}

void IndexedStorageWriteOp::infer_shape() {
    int rank=x->shape.size();
    USER_CHECK(rank>0 && rank<=16 && indices.size()>0 && indices.size()==axes.size() && axes[0]==0);
    USER_CHECK(values->dtype()==x->dtype() && values->is_contiguous());
    int nfree=rank-indices.size();
    USER_CHECK(values->shape.size()==nfree+1);
    int64 rows=indices[0]->num;
    USER_CHECK(values->shape[0]==rows);
    vector<bool> indexed(rank,false);
    for (int i=0;i<(int)indices.size();++i) {
        int a=axes[i];
        USER_CHECK(a>=0 && a<rank && !indexed[a] && (i==0 || axes[i-1]<a));
        indexed[a]=true;
        USER_CHECK(indices[i]->shape.size()==1 && indices[i]->num==rows && indices[i]->is_contiguous());
        USER_CHECK(indices[i]->dtype()==ns_int32 || indices[i]->dtype()==ns_int64);
    }
    int v=1;
    vector<std::pair<int64,int64>> layout;
    for (int i=0;i<rank;++i) {
        USER_CHECK(x->storage_stride(i)>=0);
        if (!indexed[i]) USER_CHECK(values->shape[v++]==x->shape[i]);
        if (x->shape[i]>1) layout.emplace_back(x->storage_stride(i),x->shape[i]);
    }
    std::sort(layout.begin(),layout.end());
    int64 extent=1;
    for (auto d:layout) { USER_CHECK(d.first>=extent) << "overlapping destination";extent=d.first*d.second; }
    out->set_shape(storage_root->shape);out->storage_strides=storage_root->storage_strides;out->share_with(storage_root);
    if (!ordered) {ordered=true;order_after_readers(this,{x});}
}
void IndexedStorageWriteOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype() << "«N=" << indices.size();
}
#else
#ifdef JIT_cuda
void IndexedStorageWriteOp::jit_run() {
    IndexedStorageLayout layout{};
    layout.rank=x->shape.size();layout.nindex=indices.size();layout.count=values->num;
    uintptr_t begin=reinterpret_cast<uintptr_t>(x->mem_ptr);
    uintptr_t end=begin+x->storage_span_bytes();
    auto disjoint=[&](Var* value) {
        auto p=reinterpret_cast<uintptr_t>(value->mem_ptr);
        return p+value->storage_span_bytes()<=begin || p>=end;
    };
    USER_CHECK(disjoint(values)) << "indexed storage write requires a non-overlapping RHS";
    for (int i=0;i<layout.rank;++i) {layout.shape[i]=x->shape[i];layout.stride[i]=x->storage_stride(i);}
    for (int i=0;i<layout.nindex;++i) {
        USER_CHECK(disjoint(indices[i])) << "indexed storage write requires non-overlapping indices";
        layout.axes[i]=axes[i];layout.indices[i]=indices[i]->mem_ptr;layout.is64[i]=indices[i]->dtype()==ns_int64;
    }
    layout.row_size=1;
    for (int d=0;d<layout.rank;++d) {
        bool indexed=false;
        for (int i=0;i<layout.nindex;++i) indexed|=layout.axes[i]==d;
        if (!indexed) {layout.free_axes[layout.nfree++]=d;layout.row_size*=x->shape[d];}
    }
    x->note_storage_mutation();
    if (x->autograd_alias) ++*x->autograd_alias->version;
    if (layout.count) ::jittor::indexed_storage_write_cuda<T><<<std::min<int64>((layout.count+255)/256,4096),256>>>(
        x->ptr<T>(),values->ptr<T>(),layout,device_index_fault());
}
#else
void IndexedStorageWriteOp::jit_run() {USER_ERROR << "CUDA storage write required";}
#endif
#endif
}
