#include "hipblas_conv_inference_op.h"
#include "hipblas_wrapper.h"
#include "convolution.h"
#ifdef HAS_ROCM_MIOPEN
#include "miopen_conv.h"
#endif
#include "core/var.h"
#include "core/executor.h"
#include "mem/allocator.h"
#include "runtime/backend.h"
#include <algorithm>
#include <climits>

namespace jittor {
namespace {
constexpr int64 max_workspace_elements = (256LL << 20) / sizeof(float);
}

HipblasConvInferenceOp::HipblasConvInferenceOp(Var* x, Var* weight, vector<Var*> biases,
        NanoVector stride, NanoVector padding, NanoVector dilation)
    : x(x), weight(weight), bias(biases.empty() ? nullptr : biases[0]),
      stride(stride), padding(padding), dilation(dilation) {
    USER_CHECK(biases.size() <= 1) << "hipBLAS convolution takes zero or one bias";
    USER_CHECK(x->dtype() == ns_float32 && weight->dtype() == ns_float32
               && (!bias || bias->dtype() == ns_float32)) << "hipBLAS convolution requires FP32";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HipblasConvInferenceOp::infer_shape() {
    USER_CHECK(x->shape.size() == 4 && weight->shape.size() == 4
               && stride.size() == 2 && padding.size() == 2 && dilation.size() == 2)
        << "hipBLAS convolution requires NCHW, OIHW and paired geometry";
    USER_CHECK(x->shape[1] == weight->shape[1] && x->shape[1] > 0
               && weight->shape[0] > 0) << "hipBLAS convolution channel mismatch";
    USER_CHECK(!bias || (bias->shape.size() == 1 && bias->num == weight->shape[0]))
        << "hipBLAS convolution bias shape mismatch";
    NanoVector shape{x->shape[0], weight->shape[0]};
    for (int d = 0; d < 2; ++d) {
        USER_CHECK(stride[d] > 0 && dilation[d] > 0 && padding[d] >= 0
                   && weight->shape[d+2] > 0) << "invalid hipBLAS convolution geometry";
        // Bound dimensions before geometry products or narrowing for the SDK.
        USER_CHECK(x->shape[d+2] <= INT_MAX && weight->shape[d+2] <= INT_MAX
                   && stride[d] <= INT_MAX && padding[d] <= INT_MAX && dilation[d] <= INT_MAX)
            << "hipBLAS convolution geometry exceeds int32";
        int64 numerator = x->shape[d+2] + 2*padding[d]
                      - dilation[d]*(weight->shape[d+2]-1) - 1;
        USER_CHECK(numerator >= 0) << "hipBLAS convolution output is non-positive";
        int64 size = numerator / stride[d] + 1;
        USER_CHECK(size > 0 && size <= INT_MAX) << "hipBLAS convolution output is non-positive or too large";
        shape.push_back(size);
    }
    const int64 spatial = shape[2]*shape[3];
    USER_CHECK(spatial <= INT_MAX && x->shape[0] >= 0 && x->shape[0] <= INT_MAX
               && weight->shape[0] <= INT_MAX && weight->num/weight->shape[0] <= INT_MAX)
        << "hipBLAS convolution GEMM exceeds int32";
    const int64 inner = weight->num/weight->shape[0];
    USER_CHECK(!x->shape[0] || inner <= max_workspace_elements)
        << "hipBLAS convolution one-column workspace exceeds 256 MiB";
    y->set_shape(shape);
}

void HipblasConvInferenceOp::run() {
    if (!y->num) return;
    const int batch=x->shape[0], channels=x->shape[1], oc=weight->shape[0];
    const int kh=weight->shape[2], kw=weight->shape[3], inner=channels*kh*kw;
    const int oh=y->shape[2], ow=y->shape[3], spatial=oh*ow;
    const int64 plane=(int64)inner*spatial;
    const bool pointwise = kh == 1 && kw == 1 && stride[0] == 1 && stride[1] == 1
        && padding[0] == 0 && padding[1] == 0;
    auto stream=static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(y->device_id));
    bool completed = false;
#ifdef HAS_ROCM_MIOPEN
    if (!pointwise && plane > max_workspace_elements / batch)
        completed = try_miopen_conv_inference(x,weight,y,stride,padding,dilation);
#endif
    if (!completed) {
        auto fill_columns = [&](const float* input, float* columns, int64 count, int tile, int offset) {
            auto status=hip_conv_columns(input,columns,count,channels,x->shape[2],x->shape[3],
                kh,kw,tile,ow,offset,stride[0],stride[1],padding[0],padding[1],
                dilation[0],dilation[1],stream);
            CHECK(status == hipSuccess) << hipGetErrorString(status);
        };
        auto handle=hipblas_bind_stream(y->device_id);
        const float alpha=1.f,beta=0.f;
        if (pointwise || plane <= max_workspace_elements / batch) {
            Allocation workspace(runtime_executor().temp_allocator, pointwise ? 0 : plane*batch*sizeof(float));
            const float* columns=x->ptr<float>();
            if (!pointwise) {
                auto buf=static_cast<float*>(workspace.ptr);
                fill_columns(x->ptr<float>(),buf,plane*batch,spatial,0);
                columns=buf;
            }
            // Independent batches write NCHW directly. The weights have batch
            // stride zero, with no duplicated weights or output transpose.
            if (batch == 1)
                check_hipblas(hipblasSgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,
                    spatial,oc,inner,&alpha,columns,spatial,weight->ptr<float>(),inner,
                    &beta,y->ptr<float>(),spatial),"hipblasSgemm convolution");
            else
                check_hipblas(hipblasSgemmStridedBatched(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,
                    spatial,oc,inner,&alpha,columns,spatial,plane,weight->ptr<float>(),inner,0,
                    &beta,y->ptr<float>(),spatial,(int64)oc*spatial,batch),"hipblasSgemmStridedBatched convolution");
        } else {
            // Tile the output spatial positions without exceeding the workspace
            // budget. Each tile writes directly into the NCHW output with the full
            // spatial leading dimension. The same stream orders workspace reuse.
            const int tile_size=std::min<int64>(spatial,max_workspace_elements/inner);
            Allocation workspace(runtime_executor().temp_allocator,(int64)inner*tile_size*sizeof(float));
            auto columns=static_cast<float*>(workspace.ptr);
            const int64 input_batch_num=x->num/batch;
            for (int n=0; n<batch; ++n) {
                const float* input=x->ptr<float>()+n*input_batch_num;
                float* output=y->ptr<float>()+(int64)n*oc*spatial;
                for (int offset=0; offset<spatial;) {
                    int tile=std::min(tile_size,spatial-offset);
                    fill_columns(input,columns,(int64)inner*tile,tile,offset);
                    check_hipblas(hipblasSgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,
                        tile,oc,inner,&alpha,columns,tile,weight->ptr<float>(),inner,
                        &beta,output+offset,spatial),"hipblasSgemm tiled convolution");
                    offset+=tile;
                }
            }
        }
    }
    if (bias) {
        auto status=hip_conv_bias(y->ptr<float>(),bias->ptr<float>(),y->num,oc,spatial,stream);
        CHECK(status == hipSuccess) << hipGetErrorString(status);
    }
}
} // namespace jittor
