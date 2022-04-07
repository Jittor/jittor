import jittor as jt
import numpy as np
from jittor import nn
import time
import sys
import os

cutlass_path = os.environ.get('cutlass_include_path')
def depthwise_src_backward(x, weights):
    cuda_header = '''
    #undef out
    #include <cutlass/cutlass.h>

    #include <cutlass/numeric_types.h>
    #include <cutlass/gemm/device/gemm.h>

    #include <cutlass/util/host_tensor.h>

    #include <cutlass/gemm/device/gemm.h>
    #include <cutlass/convolution/device/convolution.h>

    #include <cutlass/util/command_line.h>
    #include <cutlass/util/host_tensor.h>
    #include <cutlass/util/tensor_view_io.h>
    #include <cutlass/util/reference/device/gemm.h>
    #include <cutlass/util/reference/host/tensor_compare.h>
    #include <cutlass/util/reference/host/tensor_copy.h>
    #include <cutlass/util/reference/host/tensor_fill.h>
    #include <cutlass/util/reference/host/convolution.h>
    #include <cutlass/util/tensor_view_io.h>

    #include "executor.h"

    #include <iostream>


    // The code section below describes datatype for input, output tensors and
    // computation between elements 
    using ElementAccumulator = float;  // Data type of accumulator
    using ElementComputeEpilogue = float;                // Data type of epilogue computation (alpha, beta)
    using ElementSrc = float;     // Data type of elements in src tensor
    using ElementFilter = float;  // Data type of elements in filter tensor
    using ElementDst = float;     // Data type of elements in output tensor

    using LayoutSrc = cutlass::layout::TensorNCHW;
    using LayoutFilter = cutlass::layout::TensorNCHW;
    using LayoutDst = cutlass::layout::TensorNCHW;

    // This code section describes whether you want to use tensor cores or regular
    // SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassSimt;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75; 

    // This code section describes the tile size a thread block will compute
    using ThreadblockShape =
            cutlass::gemm::GemmShape<32, 32, 8>;  // Threadblock tile shape

    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;  // Warp tile shape

    // This code section describes the size of MMA op
    using InstructionShape =
            cutlass::gemm::GemmShape<1, 1, 1>;  // TensorCore instruction shape 

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock =
            cutlass::conv::threadblock::DepthwiseConvolutionDgradThreadblockSwizzle;

    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    // This code section describes the epilogue part of the kernel, we use default
    // value
    using EpilogueOp = cutlass::epilogue::thread::BiasAddLinearCombination<
            ElementDst,               // Data type of output matrix.
            1, ElementAccumulator,    // Data type of accumulator
            ElementDst,               // Data type of bias
            ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                    // combination
    using Convolution = cutlass::conv::device::Deconvolution<
            ElementSrc, LayoutSrc, ElementFilter, LayoutFilter, ElementDst,
            LayoutDst, ElementDst, LayoutDst, ElementDst,
            cutlass::conv::ConvType::kDepthwiseConvolution, MMAOp, SmArch,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
            SwizzleThreadBlock, NumStages, 1, 1,
            cutlass::conv::SpecialOptimizeDesc::NONE, cutlass::arch::OpMultiplyAdd,
            cutlass::conv::ImplicitGemmMode::GEMM_TN>;

    struct Options {
        bool help;
        cutlass::Tensor4DCoord input_size;
        cutlass::Tensor4DCoord filter_size;
        cutlass::Tensor4DCoord padding;
        cutlass::MatrixCoord conv_stride;
        cutlass::MatrixCoord dilation;
        bool reference_check;
        bool measure_performance;
        int iterations;
        bool save_workspace;
        ElementComputeEpilogue alpha;
        ElementComputeEpilogue beta;
        bool benchmark;
        std::string tag;

        Options()
                : help(false),
                input_size(1, 32, 32, 32),
                filter_size(32, 3, 3, 1),
                padding(1, 1, 1, 1),
                conv_stride(1, 1),
                dilation(1, 1),
                reference_check(false),
                measure_performance(true),
                iterations(1000),
                save_workspace(false),
                alpha(1),
                beta(0),
                benchmark(false) {}

            // Verify the problem size is compatible with the CUTLASS Convolution
        // implementation.
        bool valid() {
            int const kAlignment = 1;

            if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
                // misaligned tensors
                return false;
            }

            // Invalid padding
            if ((padding.h() != filter_size.h() / 2) ||
                (padding.w() != filter_size.w() / 2)) {
                return false;
            }

            return true;
        }

        /// Updates input and filter sizes
        void update(cutlass::Tensor4DCoord input_size,
                    cutlass::Tensor4DCoord filter_size) {
            this->input_size = input_size;
            this->filter_size = filter_size;

            padding.n() = filter_size.h() / 2;
            padding.h() = filter_size.h() / 2;
            padding.w() = filter_size.w() / 2;
            padding.c() = filter_size.w() / 2;
        }

        /// Computes the output tensor size (NPQK)
        cutlass::Tensor4DCoord output_size() const {
            return cutlass::Tensor4DCoord(
                    input_size.n(),
                    (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                    conv_stride.row() +
                            1,
                    (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                    conv_stride.column() +
                            1,
                    filter_size.n());
        }

        /// Compute performance in GFLOP/s
        double gflops(double runtime_s) const {
            // Number of multiply-adds = NPQK * CRS / K
            int64_t fmas =
                    output_size().product() *
                    int64_t(filter_size.h() * filter_size.w() * filter_size.c()) /
                    output_size().c();

            // Two flops per multiply-add
            return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
        }

    };

    #define CUTLASS_CHECK(status)                                                 \
        {                                                                         \
            cutlass::Status error = status;                                       \
            if (error != cutlass::Status::kSuccess) {                             \
                std::cerr << "Got cutlass error: "                                \
                        << cutlassGetStatusString(error) << " at: " << __LINE__ \
                        << std::endl;                                           \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }

    #define CUDA_CHECK(status)                                                    \
        {                                                                         \
            cudaError_t error = status;                                           \
            if (error != cudaSuccess) {                                           \
                std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                        << " at line: " << __LINE__ << std::endl;               \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }

    #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
    #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

    '''

    cuda_src = '''
    @alias(weights, in1)      
    @alias(x, in0)
    @alias(dst, out0)
    bool notSupported = false;
    Options options = Options(); 
    options.update(  
        {x_shape0, x_shape2, x_shape3, x_shape1},
        {weights_shape0, weights_shape2, weights_shape3, 1}); 

    cutlass::TensorRef<ElementSrc, LayoutSrc> d_src((ElementSrc*)x_p, 
        LayoutSrc().packed({options.input_size}));
    cutlass::TensorRef<ElementFilter, LayoutFilter> d_filter((ElementFilter*)weights_p,
        LayoutFilter().packed(options.filter_size));
    cutlass::TensorRef<typename Convolution::ElementDst,
                    typename Convolution::LayoutDst> d_dst((ElementDst*)dst_p, LayoutDst().packed(options.output_size()));
    cutlass::TensorRef<typename Convolution::ElementDst,
                    typename Convolution::LayoutDst> d_bias = {nullptr, Convolution::LayoutDst()};
    cutlass::TensorRef<typename Convolution::ElementDst,
                    typename Convolution::LayoutDst> d_z = {nullptr, Convolution::LayoutDst()};
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    int split_k_slices = 1;    
    typename Convolution::Arguments arguments{
            {options.input_size, options.filter_size, options.padding,
            options.conv_stride, options.dilation, options.output_size(), mode,
            split_k_slices, options.filter_size.n()},
            d_src,      // tensor_src.device_ref(),
            d_filter,   // tensor_filter.device_ref(),
            d_bias,       // tensor_bias.device_ref(),
            d_z,       // tensor_z.device_ref(),
            d_dst,      // tensor_dst.device_ref(),
            {options.alpha, 0, options.beta}};

    Convolution conv_op;  

    size_t workspace_size = conv_op.get_workspace_size(arguments);

    // Allocate workspace memory  
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(conv_op.initialize(arguments, workspace.get()));
    // auto temp1 = exe.alloc_temp(workspace_size);
    // CUTLASS_CHECK(conv_op.initialize(arguments, temp1.ptr)); 
    {
        //static SimpleProfiler _("aa");
        //SimpleProfilerGuard __(_);        

    //ccccc   c 
    CUTLASS_CHECK(conv_op());      
    // kernel<<<1,1>>>();      
    }
    '''
    output = jt.zeros(x.shape)
    output = jt.code([x, weights], [output], cuda_header=cuda_header, cuda_src=cuda_src)[0]
    output.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I{cutlass_path}/include -I{cutlass_path}/tools/util/include ": 1}
    return output

def depthwise_filter_backward(x, weights, diff):
    cuda_header = '''
    #undef out
    #include <cutlass/cutlass.h>
    #include <cutlass/numeric_types.h>
    #include <cutlass/gemm/device/gemm.h>
    #include <cutlass/util/host_tensor.h>
    #include <cutlass/gemm/device/gemm.h>
    #include <cutlass/convolution/device/convolution.h>
    #include <cutlass/util/command_line.h>
    #include <cutlass/util/host_tensor.h>
    #include <cutlass/util/tensor_view_io.h>
    #include <cutlass/util/reference/device/gemm.h>
    #include <cutlass/util/reference/host/tensor_compare.h>
    #include <cutlass/util/reference/host/tensor_copy.h>
    #include <cutlass/util/reference/host/tensor_fill.h>
    #include <cutlass/util/reference/host/convolution.h>
    #include <cutlass/util/tensor_view_io.h>

    #include "executor.h"

    #include <iostream>

    // The code section below describes datatype for input, output tensors and
    // computation between elements
    using ElementAccumulator = float;  // Data type of accumulator
    using ElementComputeEpilogue = float;                // Data type of epilogue computation (alpha, beta)
    using ElementSrc = float;     // Data type of elements in src tensor
    using ElementFilter = float;  // Data type of elements in filter tensor
    using ElementDst = float;     // Data type of elements in output tensor

    using LayoutSrc = cutlass::layout::TensorNCHW;
    using LayoutFilter = cutlass::layout::TensorNCHW;
    using LayoutDst = cutlass::layout::TensorNCHW;
    using LayoutGrad = cutlass::layout::TensorNCHW;

    // This code section describes whether you want to use tensor cores or regular
    // SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassSimt;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using ThreadblockShape =
            cutlass::gemm::GemmShape<32, 32, 8>;  // Threadblock tile shape

    // This code section describes tile size a warp will compute 
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;  // Warp tile shape 

    // This code section describes the size of MMA op
    using InstructionShape =
            cutlass::gemm::GemmShape<1, 1, 1>;  // TensorCore instruction shape

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock =
            cutlass::conv::threadblock::DepthwiseConvolutionWgradThreadblockSwizzle;

    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    // This code section describes the epilogue part of the kernel, we use default
    // value
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementFilter,               // Data type of output matrix.
            1, ElementAccumulator,    // Data type of accumulator
            ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                    // combination
    using Convolution = cutlass::conv::device::ConvolutionBackwardFilter<
            ElementSrc, LayoutSrc, ElementDst, LayoutDst, ElementFilter,
            LayoutFilter, ElementFilter,
            cutlass::conv::ConvType::kDepthwiseConvolution, MMAOp, SmArch,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
            SwizzleThreadBlock, NumStages, 1, 1,
            cutlass::conv::SpecialOptimizeDesc::NONE, cutlass::arch::OpMultiplyAdd,
            cutlass::conv::ImplicitGemmMode::GEMM_NT>;

    struct Options {
        bool help;
        cutlass::Tensor4DCoord input_size;
        cutlass::Tensor4DCoord filter_size;
        cutlass::Tensor4DCoord padding;
        cutlass::MatrixCoord conv_stride;
        cutlass::MatrixCoord dilation;
        bool reference_check;
        bool measure_performance;
        int iterations;
        bool save_workspace;
        ElementComputeEpilogue alpha;
        ElementComputeEpilogue beta;
        bool benchmark;
        std::string tag;

        Options()
                : help(false),
                input_size(1, 32, 32, 32),
                filter_size(32, 3, 3, 1),
                padding(1, 1, 1, 1),
                conv_stride(1, 1),
                dilation(1, 1),
                reference_check(false),
                measure_performance(true),
                iterations(1000),
                save_workspace(false),
                alpha(1),
                beta(0),
                benchmark(false) {}

            // Verify the problem size is compatible with the CUTLASS Convolution
        // implementation.
        bool valid() {
            int const kAlignment = 1;

            if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
                // misaligned tensors
                return false;
            }

            // Invalid padding
            if ((padding.h() != filter_size.h() / 2) ||
                (padding.w() != filter_size.w() / 2)) {
                return false;
            }

            return true;
        }

        /// Updates input and filter sizes
        void update(cutlass::Tensor4DCoord input_size,
                    cutlass::Tensor4DCoord filter_size) {
            this->input_size = input_size;
            this->filter_size = filter_size;

            padding.n() = filter_size.h() / 2;
            padding.h() = filter_size.h() / 2;
            padding.w() = filter_size.w() / 2;
            padding.c() = filter_size.w() / 2;
        }

        /// Computes the output tensor size (NPQK)
        cutlass::Tensor4DCoord output_size() const {
            return cutlass::Tensor4DCoord(
                    input_size.n(),
                    (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                    conv_stride.row() +
                            1,
                    (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                    conv_stride.column() +
                            1,
                    filter_size.n());
        }

        /// Compute performance in GFLOP/s
        double gflops(double runtime_s) const {
            // Number of multiply-adds = NPQK * CRS / K
            int64_t fmas =
                    output_size().product() *
                    int64_t(filter_size.h() * filter_size.w() * filter_size.c()) /
                    output_size().c();

            // Two flops per multiply-add
            return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
        }

    };

    #define CUTLASS_CHECK(status)                                                 \
        {                                                                         \
            cutlass::Status error = status;                                       \
            if (error != cutlass::Status::kSuccess) {                             \
                std::cerr << "Got cutlass error: "                                \
                        << cutlassGetStatusString(error) << " at: " << __LINE__ \
                        << std::endl;                                           \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }

    #define CUDA_CHECK(status)                                                    \
        {                                                                         \
            cudaError_t error = status;                                           \
            if (error != cudaSuccess) {                                           \
                std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                        << " at line: " << __LINE__ << std::endl;               \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }

    #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
    #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
    '''

    cuda_src = '''
    @alias(grad, in2)
    @alias(weights, in1)      
    @alias(x, in0)
    @alias(dst, out0)
    bool notSupported = false;
    Options options = Options(); 
    options.update(  
        {x_shape0, x_shape2, x_shape3, x_shape1},
        {weights_shape0, weights_shape2, weights_shape3, 1}); 

    cutlass::TensorRef<ElementSrc, LayoutSrc> d_src((ElementSrc*)x_p, 
        LayoutSrc().packed({options.input_size}));
    cutlass::TensorRef<ElementDst, LayoutDst> d_diff((ElementFilter*)grad_p,
        LayoutDst().packed(options.output_size()));
    cutlass::TensorRef<typename Convolution::ElementGrad,
                typename Convolution::LayoutGrad> d_filter((ElementFilter*)dst_p, LayoutFilter().packed(options.filter_size));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    int split_k_slices = 1;    
    typename Convolution::Arguments arguments{
            {options.input_size, options.filter_size, options.padding,
            options.conv_stride, options.dilation, options.output_size(), mode,
            split_k_slices, options.filter_size.n()},
            d_src,      // tensor_src.device_ref(),
            d_diff,   // tensor_filter.device_ref(),
            d_filter,
            {options.alpha}};

    Convolution conv_op;  

    size_t workspace_size = conv_op.get_workspace_size(arguments);

    // Allocate workspace memory  
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(conv_op.initialize(arguments, workspace.get()));
    // auto temp1 = exe.alloc_temp(workspace_size);
    // CUTLASS_CHECK(conv_op.initialize(arguments, temp1.ptr)); 
    {
        //static SimpleProfiler _("aa");
        //SimpleProfilerGuard __(_);        

    //ccccc   c 
    CUTLASS_CHECK(conv_op());      
    // kernel<<<1,1>>>();      
    }
    '''
    output = jt.zeros(weights.shape)
    output = jt.code([x, weights, diff], [output], cuda_header=cuda_header, cuda_src=cuda_src)[0]
    output.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I{cutlass_path}/include -I{cutlass_path}/tools/util/include ": 1}
    return output

class DepthwiseConv(jt.Function):
    def __init__(self, stride=1, padding=0, dilation=1):
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.x = None
        self.weights = None

    def execute(self, x, weights):
        self.x = x
        self.weights = weights
        cuda_header = '''
        #undef out
        #include <iostream> 
        #include <sstream> 
        #include <device_launch_parameters.h>
        #include <cutlass/cutlass.h>

        #include <cutlass/numeric_types.h>
        #include <cutlass/gemm/device/gemm.h>

        #include <cutlass/util/host_tensor.h>

        #include <cutlass/gemm/device/gemm.h>
        #include <cutlass/convolution/device/convolution.h>

        #include <cutlass/util/command_line.h>
        #include <cutlass/util/host_tensor.h>
        #include <cutlass/util/tensor_view_io.h>
        #include <cutlass/util/reference/device/gemm.h>
        #include <cutlass/util/reference/host/tensor_compare.h>
        #include <cutlass/util/reference/host/tensor_copy.h>
        #include <cutlass/util/reference/host/tensor_fill.h>
        #include <cutlass/util/reference/host/convolution.h>
        #include <cutlass/util/tensor_view_io.h>
        #include "executor.h"

        // The code section below describes datatype for input, output tensors and
        // computation between elements
        using ElementAccumulator = float;  // Data type of accumulator
        using ElementComputeEpilogue = float;                // Data type of epilogue computation (alpha, beta)
        using ElementSrc = float;     // Data type of elements in src tensor
        using ElementFilter = float;  // Data type of elements in filter tensor
        using ElementDst = float;     // Data type of elements in output tensor

        using LayoutSrc = cutlass::layout::TensorNCHW;
        using LayoutFilter = cutlass::layout::TensorNCHW;
        using LayoutDst = cutlass::layout::TensorNCHW;

        // This code section describes whether you want to use tensor cores or regular
        // SIMT cores on GPU SM
        using MMAOp = cutlass::arch::OpClassSimt;

        // This code section describes CUDA SM architecture number
        using SmArch = cutlass::arch::Sm75;

        // This code section describes the tile size a thread block will compute
        using ThreadblockShape = 
                cutlass::gemm::GemmShape<32, 32, 8>;  // Threadblock tile shape

        // This code section describes tile size a warp will compute   
        using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;  // Warp tile shape

        // This code section describes the size of MMA op 
        using InstructionShape =   
                cutlass::gemm::GemmShape<1, 1, 1>;  // TensorCore instruction shape

        // This code section describes how threadblocks are scheduled on GPU
        using SwizzleThreadBlock =
                cutlass::conv::threadblock::DepthwiseConvolutionFpropThreadblockSwizzle;
            
        // Number of pipelines you want to use
        constexpr int NumStages = 1; 

        // This code section describes the epilogue part of the kernel, we use default   
        // value 
        using EpilogueOp = cutlass::epilogue::thread::BiasAddLinearCombination<
                ElementDst,               // Data type of output matrix.
                1, ElementAccumulator,    // Data type of accumulator
                ElementDst,               // Data type of bias
                ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                        // combination
        using Convolution = cutlass::conv::device::Convolution< 
                ElementSrc, LayoutSrc, ElementFilter, LayoutFilter, ElementDst,
                LayoutDst, ElementDst, LayoutDst, ElementDst,
                cutlass::conv::ConvType::kDepthwiseConvolution, MMAOp, SmArch,
                ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
                SwizzleThreadBlock, NumStages, 1, 1,
                cutlass::conv::SpecialOptimizeDesc::NONE, cutlass::arch::OpMultiplyAdd,  
                cutlass::conv::ImplicitGemmMode::GEMM_TN>; 

        struct Options {
            bool help; 
            cutlass::Tensor4DCoord input_size;   
            cutlass::Tensor4DCoord filter_size;
            cutlass::Tensor4DCoord padding;
            cutlass::MatrixCoord conv_stride;
            cutlass::MatrixCoord dilation;
            bool reference_check;
            bool measure_performance;
            int iterations;
            bool save_workspace; 
            ElementComputeEpilogue alpha;
            ElementComputeEpilogue beta;
            bool benchmark;
            std::string tag;

            Options()
                    : help(false),
                    input_size(1, 32, 32, 32),
                    filter_size(32, 3, 3, 1),
                    padding(1, 1, 1, 1),
                    conv_stride(1, 1),
                    dilation(1, 1),
                    reference_check(false),
                    measure_performance(false),
                    iterations(1000),
                    save_workspace(false),
                    alpha(1),
                    beta(0),
                    benchmark(false) {}

                // Verify the problem size is compatible with the CUTLASS Convolution
            // implementation.
            bool valid() {  
                int const kAlignment = 1;   

                if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
                    // misaligned tensors
                    return false;
                }

                // Invalid padding
                if ((padding.h() != filter_size.h() / 2) ||
                    (padding.w() != filter_size.w() / 2)) {
                    return false;
                }

                return true;
            }  

            /// Updates   input  and   filter  sizes      
            void update(cutlass::Tensor4DCoord input_size,
                        cutlass::Tensor4DCoord filter_size) {
                this->input_size = input_size;
                this->filter_size = filter_size;

                padding.n() = filter_size.h() / 2; 
                padding.h() = filter_size.h() / 2;  
                padding.w() = filter_size.w() / 2; 
                padding.c() = filter_size.w() / 2;   
            }  

            /// Computes the output tensor size (NPQK)    
            cutlass::Tensor4DCoord output_size() const {
                return cutlass::Tensor4DCoord(
                        input_size.n(),
                        (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                        conv_stride.row() +
                                1,
                        (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                        conv_stride.column() +
                                1,
                        filter_size.n()); 
            }

            /// Compute performance in GFLOP/s 
            double gflops(double runtime_s) const {
                // Number of multiply-adds = NPQK * CRS / K
                int64_t fmas =
                        output_size().product() *
                        int64_t(filter_size.h() * filter_size.w() * filter_size.c()) /
                        output_size().c();

                // Two flops per multiply-add
                return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
            }

        };

        #define CUTLASS_CHECK(status)                                                 \
            {                                                                         \
                cutlass::Status error = status;                                       \
                if (error != cutlass::Status::kSuccess) {                             \
                    std::cerr << "Got cutlass error: "                                \
                            << cutlassGetStatusString(error) << " at: " << __LINE__ \
                            << std::endl;                                           \
                    exit(EXIT_FAILURE);                                               \
                }                                                                     \
            }

        #define CUDA_CHECK(status)                                                    \
            {                                                                         \
                cudaError_t error = status;                                           \
                if (error != cudaSuccess) {                                           \
                    std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                            << " at line: " << __LINE__ << std::endl;               \
                    exit(EXIT_FAILURE);                                               \
                }                                                                     \
            }

        '''
        cuda_src = '''
            // __global__ void kernel() {}
            @alias(weights, in1)      
            @alias(x, in0)
            @alias(dst, out0)
            bool notSupported = false;
            Options options = Options(); 
            options.update(  
                {x_shape0, x_shape2, x_shape3, x_shape1},
                {weights_shape0, weights_shape2, weights_shape3, 1}); 

            cutlass::TensorRef<ElementSrc, LayoutSrc> d_src((ElementSrc*)x_p, 
                LayoutSrc().packed({options.input_size}));
            cutlass::TensorRef<ElementFilter, LayoutFilter> d_filter((ElementFilter*)weights_p,
                LayoutFilter().packed(options.filter_size));
            cutlass::TensorRef<typename Convolution::ElementDst,
                            typename Convolution::LayoutDst> d_dst((ElementDst*)dst_p, LayoutDst().packed(options.output_size()));
            cutlass::TensorRef<typename Convolution::ElementDst,
                            typename Convolution::LayoutDst> d_bias = {nullptr, Convolution::LayoutDst()};
            cutlass::TensorRef<typename Convolution::ElementDst,
                            typename Convolution::LayoutDst> d_z = {nullptr, Convolution::LayoutDst()};
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

            int split_k_slices = 1;    
            typename Convolution::Arguments arguments{
                    {options.input_size, options.filter_size, options.padding,
                    options.conv_stride, options.dilation, options.output_size(), mode,
                    split_k_slices, options.filter_size.n()},
                    d_src,      // tensor_src.device_ref(),
                    d_filter,   // tensor_filter.device_ref(),
                    d_bias,       // tensor_bias.device_ref(),
                    d_z,       // tensor_z.device_ref(),
                    d_dst,      // tensor_dst.device_ref(),
                    {options.alpha, 0, options.beta}};
            Convolution conv_op;  

            size_t workspace_size = conv_op.get_workspace_size(arguments); 

            // Allocate workspace memory  
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

            CUTLASS_CHECK(conv_op.initialize(arguments, workspace.get()));
            // auto temp1 = exe.alloc_temp(workspace_size);
            // CUTLASS_CHECK(conv_op.initialize(arguments, temp1.ptr)); 
            {
                //static SimpleProfiler _("aa"); 
                //SimpleProfilerGuard __(_);         

            //ccccc   c 
            CUTLASS_CHECK(conv_op());       
            // kernel<<<1,1>>>();      
            }
        '''
        output = jt.zeros(x.shape)
        output = jt.code([x, weights], [output], cuda_header=cuda_header, cuda_src=cuda_src)[0]
        output.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I{cutlass_path}/include -I{cutlass_path}/tools/util/include ": 1}
        return output
    
    def grad(self, g):
        return depthwise_src_backward(g, self.weights), depthwise_filter_backward(self.x, self.weights, g)