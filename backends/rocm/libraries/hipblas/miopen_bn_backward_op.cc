#include "miopen_bn_backward_op.h"
#include "miopen_conv.h"
#include "core/var.h"

namespace jittor {

MiopenBnBackwardOp::MiopenBnBackwardOp(Var* x, Var* weight, Var* dy,
        Var* mean, Var* invvar, double eps)
    : x(x),weight(weight),dy(dy),mean(mean),invvar(invvar),eps(eps) {
    USER_CHECK(x->shape.size()==4 && x->shape==dy->shape && x->dtype()==dy->dtype()
        && (x->dtype()==ns_float32 || x->dtype()==ns_float16)
        && weight->dtype()==ns_float32 && mean->dtype()==ns_float32 && invvar->dtype()==ns_float32
        && weight->shape==NanoVector{x->shape[1]} && mean->shape==weight->shape && invvar->shape==weight->shape)
        << "invalid MIOpen BN backward operands";
    set_flag(OpFlags::_cpu,0);
    set_flag(OpFlags::_cuda,1);
    dx=create_output(nullptr,x->dtype());
    dw=create_output(nullptr,ns_float32);
    db=create_output(nullptr,ns_float32);
}

void MiopenBnBackwardOp::infer_shape() {
    dx->set_shape(x->shape);
    dw->set_shape(weight->shape);
    db->set_shape(weight->shape);
}

void MiopenBnBackwardOp::run() {
    struct Descriptors {
        miopenTensorDescriptor_t data=nullptr,params=nullptr;
        ~Descriptors() {
            if (data) miopenDestroyTensorDescriptor(data);
            if (params) miopenDestroyTensorDescriptor(params);
        }
    } d;
    auto check=[](miopenStatus_t status,const char* operation) {
        CHECK(status==miopenStatusSuccess) << operation << ": " << miopenGetErrorString(status);
    };
    check(miopenCreateTensorDescriptor(&d.data),"MIOpen BN descriptor");
    check(miopenCreateTensorDescriptor(&d.params),"MIOpen BN parameters");
    check(miopenSet4dTensorDescriptor(d.data,x->dtype()==ns_float16 ? miopenHalf : miopenFloat,
        x->shape[0],x->shape[1],x->shape[2],x->shape[3]),"MIOpen BN data");
    check(miopenSet4dTensorDescriptor(d.params,miopenFloat,1,x->shape[1],1,1),"MIOpen BN affine");
    float alpha=1.f,beta=0.f;
    with_miopen_handle(dx->device_id,[&](miopenHandle_t handle) {
        check(miopenBatchNormalizationBackward(handle,miopenBNSpatial,
            &alpha,&beta,&alpha,&beta,d.data,x->mem_ptr,d.data,dy->mem_ptr,
            d.data,dx->mem_ptr,d.params,weight->mem_ptr,dw->mem_ptr,db->mem_ptr,
            eps,mean->mem_ptr,invvar->mem_ptr),"miopenBatchNormalizationBackward");
    });
}

} // namespace jittor
