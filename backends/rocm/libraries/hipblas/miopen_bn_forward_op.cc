#include "miopen_bn_forward_op.h"
#include "miopen_conv.h"
#include "core/var.h"
#include "runtime/backend.h"

namespace jittor {
namespace {
void check_bn(miopenStatus_t status, const char* operation) {
    CHECK(status == miopenStatusSuccess) << operation << ": " << miopenGetErrorString(status);
}
struct BnDescriptors {
    miopenTensorDescriptor_t data=nullptr, params=nullptr;
    ~BnDescriptors() {
        if (data) miopenDestroyTensorDescriptor(data);
        if (params) miopenDestroyTensorDescriptor(params);
    }
    explicit BnDescriptors(Var* x) {
        check_bn(miopenCreateTensorDescriptor(&data),"MIOpen BN descriptor");
        check_bn(miopenCreateTensorDescriptor(&params),"MIOpen BN parameters");
        check_bn(miopenSet4dTensorDescriptor(data,x->dtype()==ns_float16 ? miopenHalf : miopenFloat,
            x->shape[0],x->shape[1],x->shape[2],x->shape[3]),"MIOpen BN data");
        check_bn(miopenSet4dTensorDescriptor(params,miopenFloat,1,x->shape[1],1,1),"MIOpen BN affine");
    }
};
}

MiopenBnForwardOp::MiopenBnForwardOp(Var* x, Var* weight, Var* bias, double eps)
    : x(x),weight(weight),bias(bias),eps(eps) {
    USER_CHECK(x->shape.size()==4 && (x->dtype()==ns_float32 || x->dtype()==ns_float16)
        && weight->dtype()==ns_float32 && bias->dtype()==ns_float32
        && weight->shape==NanoVector{x->shape[1]} && bias->shape==weight->shape)
        << "MIOpen BN training requires NCHW FP32/FP16 data and FP32 affine";
    set_flag(OpFlags::_cpu,0);
    set_flag(OpFlags::_cuda,1);
    y=create_output(nullptr,x->dtype());
    mean=create_output(nullptr,ns_float32);
    invvar=create_output(nullptr,ns_float32);
    running_mean=create_output(nullptr,ns_float32);
    running_var=create_output(nullptr,ns_float32);
}

void MiopenBnForwardOp::infer_shape() {
    y->set_shape(x->shape);
    for (auto* stat : {mean,invvar,running_mean,running_var})
        stat->set_shape(weight->shape);
}

void MiopenBnForwardOp::run() {
    BnDescriptors d(x);
    auto stream=static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(y->device_id));
    CHECK(hipMemsetAsync(running_mean->mem_ptr,0,running_mean->size,stream)==hipSuccess);
    CHECK(hipMemsetAsync(running_var->mem_ptr,0,running_var->size,stream)==hipSuccess);
    float alpha=1.f,beta=0.f;
    with_miopen_handle(y->device_id,[&](miopenHandle_t handle) {
        check_bn(miopenBatchNormalizationForwardTraining(handle,miopenBNSpatial,&alpha,&beta,
            d.data,x->mem_ptr,d.data,y->mem_ptr,d.params,weight->mem_ptr,bias->mem_ptr,
            1.0,running_mean->mem_ptr,running_var->mem_ptr,eps,mean->mem_ptr,invvar->mem_ptr),
            "miopenBatchNormalizationForwardTraining");
    });
}

} // namespace jittor
