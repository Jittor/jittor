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

def backward_header():
    return '''
        #pragma once
        #undef out
        #include <cutlass/cutlass.h>

        #include <cutlass/array.h>
        #include <cutlass/functional.h>
        #include <cutlass/gemm/device/gemm.h>
        #include <cutlass/gemm/device/gemm_splitk_parallel.h>
        #include <cutlass/numeric_conversion.h>
        #include <cutlass/numeric_types.h>
        #include <cutlass/util/host_tensor.h>
        #include <cutlass/util/reference/device/gemm.h>
        #include <iostream>
        #include <map>
        #include <type_traits>
        #include <stdint.h>
        #include <algorithm>

        // implement temp GPUMatrix and GPUDynamicMatrix here.
        // #define RM MatrixLayout::kRowMajor
        // #define CM MatrixLayout::kColumnMajor
        // enum class MatrixLayout {
        //   kColumnMajor,
        //   kRowMajor
        // };

        enum class Activation {
            ReLU,
            Exponential,
            Sine,
            Sigmoid,
            Squareplus,
            Softplus,
            None,
        };

        #define CUTLASS_CHECK(status)                                                                      \
        {                                                                                                  \
            cutlass::Status error = status;                                                                \
            if (error != cutlass::Status::kSuccess) {                                                      \
                std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                        << std::endl;                                                                    \
                exit(EXIT_FAILURE);                                                                        \
            }                                                                                              \
        }

        #define CUDA_CHECK(status)                                                \
        {                                                                         \
            cudaError_t error = status;                                           \
            if (error != cudaSuccess) {                                           \
                std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                        << " at line: " << __LINE__ << std::endl;               \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }

        using SmArch = cutlass::arch::Sm80;

        using TypeAccumulator = float;
        using TypeCompute = float;
        using ElementComputeEpilogue = TypeAccumulator;  // <- data type of epilogue operations
        using MMAOp = cutlass::arch::OpClassTensorOp;

        // using ShapeMMAOp = typename std::conditional<
        // 	std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value,
        // 	typename std::conditional<
        // 		std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
        // 		cutlass::gemm::GemmShape<16, 8, 8>,
        // 		cutlass::gemm::GemmShape<8, 8, 4>
        // 	>::type,
        // 	cutlass::gemm::GemmShape<1, 1, 1>
        // >::type;

        // using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;
        template <typename thread_block, typename warp>
        struct LayerConfig {
            using k_thread_block = thread_block;
            using k_warp = warp;
        };

        using FullLayerK = LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>;
        using LastLayerK = LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>;

        // using FullLayer = typename std::conditional<
        // 	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
        // 	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
        // 	LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>
        // >::type;

        // using FullLayerPreReLU = typename std::conditional<
        // 	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
        // 	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8, true>, cutlass::gemm::GemmShape<32, 64, 8, true>>,
        // 	LayerConfig<cutlass::gemm::GemmShape<128, 128, 32, true>, cutlass::gemm::GemmShape<64, 64, 32, true>>
        // >::type;

        // using LastLayer = typename std::conditional<
        // 	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
        // 	LayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
        // 	typename std::conditional<
        // 		std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
        // 		LayerConfig<cutlass::gemm::GemmShape<128, 32, 32>, cutlass::gemm::GemmShape<32, 32, 32>>,
        // 		LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>
        // 	>::type
        // >::type;

        // This code section describes how threadblocks are scheduled on GPU
        using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

        // warp activation defined here
        template <typename T, typename fragment_t>
        __host__ __device__ void warp_activation(Activation activation, const fragment_t& frag, fragment_t& result) {
            switch (activation) {
                case Activation::ReLU:
                    CUTLASS_PRAGMA_UNROLL
                    for (int t=0; t < result.num_elements; t++) {
                        result.x[t] = frag.x[t] * (T)((T)frag.x[t] > (T)0.0f);
                    }
                    return;
                case Activation::None: result = frag; return;
                default:
                    // Unsupported activation
                    // assert(false); // Commented out due to isolated strange side-effects on Windows
                    return;
            }
        }

        template <typename T, typename fragment_t>
        __host__ __device__ fragment_t warp_activation(Activation activation, const fragment_t& frag) {
            fragment_t result;
            warp_activation<T>(activation, frag, result);
            return result;
        }


        template <typename T, typename fragment_t, typename forward_fragment_t>
        __host__ __device__ void warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag, fragment_t& result) {
            switch (activation) {
                case Activation::ReLU:
                    CUTLASS_PRAGMA_UNROLL
                    for (int t=0; t < result.num_elements; t++) {
                        result.x[t] = frag.x[t] * (T)(forward_frag.x[t] > (T)0.0f);
                    }
                    return;
                case Activation::None: result = frag; return;
                default:
                    // Unsupported activation
                    // assert(false); // Commented out due to isolated strange side-effects on Windows
                    return;
            }
        }

        template <typename T, typename fragment_t, typename forward_fragment_t>
        __host__ __device__ fragment_t warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag) {
            fragment_t result;
            warp_activation_backward<T>(activation, frag, forward_frag, result);
            return result;
        }

        // // This code section describes the epilogue part of the kernel

        template <typename V>
        struct CutlassFragmentWrapper {
            static const uint32_t num_elements = V::kElements;
            V x;
        };

        template <
            typename ElementOutput_,                             ///< Data type used to load and store tensors
            int Count,                                           ///< Number of elements computed per operation
            typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
            typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
            cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
        >
        class ActivationEpilogue {
        public:
            using ElementOutput = ElementOutput_;
            using ElementAccumulator = ElementAccumulator_;
            using ElementCompute = ElementCompute_;

            static int const kCount = Count;

            using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
            using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
            using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

            static cutlass::FloatRoundStyle const kRound = Round;

            struct Params {
                Activation activation;
                bool sum_source;
            };

        public:
            CUTLASS_HOST_DEVICE
            ActivationEpilogue(Params const &params) : m_activation{params.activation}, m_sum_source{params.sum_source} { }

            CUTLASS_HOST_DEVICE
            bool is_source_needed() const {
                return m_sum_source;
            }

            /// Functionally required for serial reduction in the epilogue
            CUTLASS_HOST_DEVICE
            void set_k_partition(int k_partition, int k_partition_count) { }

            CUTLASS_HOST_DEVICE
            FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
                cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

                auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
                intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

                cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
                return destination_converter(intermediate.x);
            }

            CUTLASS_HOST_DEVICE
            FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentOutput const &source) const {
                cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
                cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

                cutlass::plus<ComputeFragment> plus_op;
                auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
                if (m_sum_source) {
                    intermediate.x = plus_op(intermediate.x, source_converter(source));
                }
                intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

                cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
                return destination_converter(intermediate.x);
            }

        private:
            Activation m_activation;
            bool m_sum_source;
        };

        template <
            typename ElementOutput_,                             ///< Data type used to load and store tensors
            int Count,                                           ///< Number of elements computed per operation
            typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
            typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
            cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
        >
        class ActivationTransferEpilogue {
        public:
            using ElementOutput = ElementOutput_;
            using ElementAccumulator = ElementAccumulator_;
            using ElementCompute = ElementCompute_;

            static int const kCount = Count;

            using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
            using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
            using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

            static cutlass::FloatRoundStyle const kRound = Round;

            /// Host-constructable parameters structure
            struct Params {
                Activation activation;
            };

        public:
            /// Constructs the function object, possibly loading from pointers in host memory
            CUTLASS_HOST_DEVICE
            ActivationTransferEpilogue(Params const &params) : m_activation{params.activation} { }

            /// Returns true if source is needed
            CUTLASS_HOST_DEVICE
            bool is_source_needed() const {
                return true;
            }

            /// Functionally required for serial reduction in the epilogue
            CUTLASS_HOST_DEVICE
            void set_k_partition(int k_partition, int k_partition_count) { }

            CUTLASS_HOST_DEVICE
            FragmentOutput operator()(
                FragmentAccumulator const &accumulator,
                FragmentOutput const &source) const {

                cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
                cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

                auto converted_source = CutlassFragmentWrapper<ComputeFragment>{source_converter(source)};
                auto intermediate = CutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};

                intermediate = warp_activation_backward<ElementCompute>(m_activation, intermediate, converted_source);

                cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
                return destination_converter(intermediate.x);
            }

            CUTLASS_HOST_DEVICE
            FragmentOutput operator()(
                FragmentAccumulator const &accumulator) const {

                cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

                ComputeFragment converted_accumulator = accumulator_converter(accumulator);

                cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

                return destination_converter(converted_accumulator);
            }

        private:
            Activation m_activation;
        };


        // template <typename T>
        // static constexpr int n_vectorized_elements = std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value ? (128 / cutlass::sizeof_bits<T>::value) : 1;

        // template <typename T>
        // using SumOp = cutlass::epilogue::thread::LinearCombination<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

        // template <typename T>
        // using IntermediateActivationOp = ActivationEpilogue<T, 4, TypeAccumulator, TypeCompute>;

        // template <typename T>
        // using IntermediateActivationTransferOp = ActivationTransferEpilogue<T, 4, TypeAccumulator, TypeCompute>;

        // template <typename T>
        // using ActivationOp = ActivationEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

        // template <typename T>
        // using ActivationTransferOp = ActivationTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

        using OurGemm = cutlass::gemm::device::Gemm<
            float,
            cutlass::layout::RowMajor,
            float,
            cutlass::layout::ColumnMajor,
            float,
            cutlass::layout::RowMajor,
            TypeAccumulator,
            MMAOp,
            SmArch,
            cutlass::gemm::GemmShape<128, 128, 32>, 
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>,
            ActivationEpilogue<float, 4, TypeAccumulator, TypeCompute>,
            SwizzleThreadBlock,
            2
        >;

        using OurGemmTransfer = cutlass::gemm::device::Gemm<
            float,
            cutlass::layout::RowMajor,
            float,
            cutlass::layout::ColumnMajor,
            float,
            cutlass::layout::RowMajor,
            TypeAccumulator,
            MMAOp,
            SmArch,
            cutlass::gemm::GemmShape<128, 128, 32>, 
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>,
            ActivationTransferEpilogue<float, 4, TypeAccumulator, TypeCompute>,
            SwizzleThreadBlock,
            2
        >;

        // using epi = cutlass::epilogue::thread::LinearCombination<float, 4, TypeAccumulator, TypeCompute>;
        // using SplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
        // 	float,
        // 	cutlass::layout::ColumnMajor,
        // 	float,
        // 	cutlass::layout::RowMajor,
        // 	float,
        // 	cutlass::layout::RowMajor,
        // 	TypeAccumulator,
        // 	MMAOp,
        // 	SmArch,
        // 	cutlass::gemm::GemmShape<128, 128, 32>, 
        // 	cutlass::gemm::GemmShape<64, 32, 32>,
        // 	cutlass::gemm::GemmShape<16, 8, 8>,
        // 	epi
        // >;

        using OurGemmW = cutlass::gemm::device::Gemm<
            float,
            cutlass::layout::ColumnMajor,
            float,
            cutlass::layout::RowMajor,
            float,
            cutlass::layout::RowMajor,
            TypeAccumulator,
            MMAOp,
            SmArch,
            cutlass::gemm::GemmShape<128, 128, 32>, 
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>,
            ActivationEpilogue<float, 4, TypeAccumulator, TypeCompute>,
            SwizzleThreadBlock,
            2
        >;

        void backward(float* input, float* grad, float* output, float* output_grad, int input_m, int input_n, int grad_m, int grad_n, int output_m, int output_n) { // grad * weight.T
            using Gemm = OurGemmTransfer;
            const int lda = grad_n;
            const int ldb = input_n;
            const int ldc = output_n;
            const int ldd = output_n;
            typename Gemm::Arguments arguments{
                {grad_m, input_m, grad_n}, // TODO
                {grad, lda},
                {input, ldb},
                {output, ldc},
                {output_grad, ldc},
                {Activation::ReLU},
                1
            };
            size_t workspace_size = Gemm::get_workspace_size(arguments);

            // Instantiate CUTLASS kernel depending on templates
            Gemm gemm_op;

            // Initialize CUTLASS kernel with arguments and workspace pointer
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
            cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
            CUTLASS_CHECK(status);

            // Launch initialized CUTLASS kernel
            status = gemm_op(0);
            CUTLASS_CHECK(status);
        }

        void last_inp_backward(float* input, float* grad, float* output, int input_m, int input_n, int grad_m, int grad_n, int output_m, int output_n) { // output * weight.T
            using Gemm = OurGemm;
            const int lda = grad_n;
            const int ldb = input_n;
            const int ldc = output_n;
            const int ldd = output_n;
            typename Gemm::Arguments arguments{
                {grad_m, input_m, grad_n}, // TODO
                {grad, lda},
                {input, ldb},
                {output, ldc},
                {output, ldc},
                {Activation::None, false},
                1
            };
            size_t workspace_size = Gemm::get_workspace_size(arguments);

            // Instantiate CUTLASS kernel depending on templates
            Gemm gemm_op;

            // Initialize CUTLASS kernel with arguments and workspace pointer
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
            cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
            CUTLASS_CHECK(status);

            // Launch initialized CUTLASS kernel
            status = gemm_op();
            CUTLASS_CHECK(status);
        }

        void weight_backward(float* input, float* grad, float* weight_grad, int input_m, int input_n, int grad_m, int grad_n, int weight_grad_m, int weight_grad_n) { // A.T * GRAD

            int batch_size = grad_n;

            using Gemm = OurGemmW;
            const int lda = input_n;
            const int ldb = grad_n;
            const int ldc = weight_grad_n;
            const int ldd = weight_grad_n;
            typename Gemm::Arguments arguments{
                {input_n, grad_n, grad_m}, // TODO
                {input, lda},
                {grad, ldb},
                {weight_grad, ldc},
                {weight_grad, ldc},
                {Activation::None, false},
                1
            };
            size_t workspace_size = Gemm::get_workspace_size(arguments);

            // Instantiate CUTLASS kernel depending on templates
            Gemm gemm_op;

            // Initialize CUTLASS kernel with arguments and workspace pointer
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
            cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
            CUTLASS_CHECK(status);

            // Launch initialized CUTLASS kernel
            status = gemm_op();
            CUTLASS_CHECK(status);
        }
    '''

class FullyFusedMlp(jt.Function):
    def __init__(self):
        self.input = None
        self.outputs = []
        self.shapes = []
        self.dtypes = []
        self.weights_grad = []
        self.max_dim = 0
        self.weights = None

    def single_forward(self, a, b):
        cuda_header = '''
            #undef out
            #include <algorithm>
            #include <iostream>

            #include <cutlass/cutlass.h>
            #include <cutlass/gemm/device/gemm.h>
            #include <cutlass/epilogue/thread/linear_combination_relu.h>
            #include <cutlass/epilogue/thread/linear_combination_drelu.h>
            #include <cutlass/util/host_tensor.h>
            #include <cutlass/util/reference/device/gemm.h>
            #include <cutlass/util/reference/host/tensor_compare.h>
            #include <cutlass/util/reference/host/tensor_copy.h>
            #include <cutlass/util/reference/host/tensor_fill.h>
            #include <cutlass/util/tensor_view_io.h>
            #include "executor.h"

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

            // The code section below describes datatype for input, output matrices and computation between
            // elements in input matrices.
            using ElementAccumulator = float;                   // <- data type of accumulator
            using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
            using ElementInputA = float;              // <- data type of elements in input matrix A
            using ElementInputB = float;              // <- data type of elements in input matrix B
            using ElementOutput = float;                        // <- data type of elements in output matrix D

            // The code section below describes matrix layout of input and output matrices.
            // Column Major for Matrix A, B and C.
            //
            // Note this example only works for ColumnMajor output because
            //   1) we only have row major epilogue.
            //   2) we swap A and B if the output is column major then we can still use the
            //      row major epilogue.
            //   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
            //   4) we can use the existing OutputIterator to load 1xM bias vector.

            using LayoutInputA = cutlass::layout::RowMajor;
            using LayoutInputB = cutlass::layout::RowMajor;
            using LayoutOutput = cutlass::layout::RowMajor;

            // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
            using MMAOp = cutlass::arch::OpClassTensorOp;

            // This code section describes CUDA SM architecture number
            using SmArch = cutlass::arch::Sm75;

            // This code section describes the tile size a thread block will compute
            using ShapeMMAThreadBlock =
                cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
            // This code section describes tile size a warp will compute
            using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
            // This code section describes the size of MMA op
            using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4

            // This code section describes how threadblocks are scheduled on GPU
            using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

            // Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
            //
            //    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
            //
            using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
                ElementOutput,                                        // <- data type of output matrix
                128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                                    // vectorized memory access. For half
                                                                    // precision, it's 8 elements. This becomes
                                                                    // the vector width of math instructions in
                                                                    // epilogue too
                ElementAccumulator,                                   // <- data type of accumulator
                ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
                cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

            // Number of pipelines you want to use
            constexpr int NumStages = 2;

            using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                    LayoutInputA,
                                                    ElementInputB,
                                                    LayoutInputB,
                                                    ElementOutput,
                                                    LayoutOutput,
                                                    ElementAccumulator,
                                                    MMAOp,
                                                    SmArch,
                                                    ShapeMMAThreadBlock,
                                                    ShapeMMAWarp,
                                                    ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages>;
        '''
        cuda_src = '''
            @alias(b, in1)
            @alias(a, in0)
            @alias(c, out0)
            const int length_m = a_shape0;
            const int length_n = b_shape1;
            const int length_k = a_shape1;

            // Create a tuple of problem size for matrix multiplication
            cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

            // Initialize tensors using CUTLASS helper functions
            cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a((ElementInputA*)a_p,
                LayoutInputA().packed(problem_size.mk()));
            cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b((ElementInputB*)b_p,
                LayoutInputB().packed(problem_size.kn()));
            cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d((ElementOutput*)c_p,
                LayoutOutput().packed(problem_size.mn()));

            // Initialize alpha for dot product computation
            ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

            // Split K dimension into 1 partitions
            int split_k_slices = 1;

            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename Gemm::Arguments arguments{
                problem_size,                       // <- problem size of matrix multiplication
                tensor_a,              // <- reference to matrix A on device
                tensor_b,              // <- reference to matrix B on device

                {NULL, 0},   // <- the C matrix is treated as the bias vector. We can enable the GEMM
                                                    //    to project away the N dimension by setting the stride to zero.

                tensor_d,              // <- reference to matrix D on device,
                {alpha},                              // <- alpha
                split_k_slices};                    // <- k-dimension split factor

            // Using the arguments, query for extra workspace required for matrix multiplication computation
            size_t workspace_size = Gemm::get_workspace_size(arguments);


            // Allocate workspace memory
            auto temp1 = exe.alloc_temp(workspace_size);

            // Allocate workspace memory
            // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

            // Instantiate CUTLASS kernel depending on templates
            Gemm gemm_op;

            // Check the problem size is supported or not 
            cutlass::Status status = gemm_op.can_implement(arguments);
            CUTLASS_CHECK(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = gemm_op.initialize(arguments, temp1.ptr);
            CUTLASS_CHECK(status);

            // Launch initialized CUTLASS kernel
            status = gemm_op();
            CUTLASS_CHECK(status);  
        '''
        output = jt.code((a.shape[0], b.shape[1]), a.dtype, [a, b], cuda_header=cuda_header, cuda_src=cuda_src)
        output.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I/home/penghy/jittor-dev/depth/cutlass/include -I/home/penghy/jittor-dev/depth/cutlass/tools/util/include ": 1}
        # print(output)
        return output
    
    def execute(self, a, *args):
        self.shapes = []
        self.dtypes = []
        self.max_dim = 0
        self.weights = list(args)
        self.input = a
        weights = args
        for i in range(len(weights)):
            self.outputs.append(self.single_forward(a, weights[i]))
            a = self.outputs[-1]
        # print(self.outputs)
        return self.outputs[-1]

    def backward(self, grad, weight, output):
        cuda_header = backward_header()
        cuda_src = '''
        @alias(input, in0)
        @alias(grad, in1)
        @alias(weight, in2)
        @alias(weight_grad, out0)  
        @alias(inp_grad, out1)
        weight_backward(input_p, grad_p, weight_grad_p, input_shape0, input_shape1, grad_shape0, grad_shape1, weight_shape0, weight_shape1); 
        backward(weight_p, grad_p, input_p, inp_grad_p, weight_shape0, weight_shape1, grad_shape0, grad_shape1, input_shape0, input_shape1);
        '''
        weight_grad, out_grad = jt.code([weight.shape, output.shape], [weight.dtype, output.dtype], [output, grad, weight], cuda_header=cuda_header, cuda_src=cuda_src)
        weight_grad.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I{cutlass_path}/include -I{cutlass_path}/tools/util/include ": 1}
        return out_grad, weight_grad

    def last_backward(self, grad, weight, output):
        cuda_header = backward_header()
        cuda_src = '''
        @alias(input, in0)
        @alias(grad, in1)  
        @alias(weight, in2) 
        @alias(weight_grad, out0) 
        @alias(out_grad, out1)
        weight_backward(input_p, grad_p, weight_grad_p, input_shape0, input_shape1, grad_shape0, grad_shape1, weight_shape0, weight_shape1);
        last_inp_backward(weight_p, grad_p, out_grad_p, weight_shape0, weight_shape1, grad_shape0, grad_shape1, input_shape0, input_shape1); 
        '''
        weight_grad, out_grad = jt.code([weight.shape, output.shape], [weight.dtype, output.dtype], [output, grad, weight], cuda_header=cuda_header, cuda_src=cuda_src)
        weight_grad.compile_options = {f"FLAGS: --expt-relaxed-constexpr -I{cutlass_path}/include -I{cutlass_path}/tools/util/include ": 1}
        return out_grad, weight_grad

    def grad(self, grads):
        self.weights_grad = []
        output = self.outputs[-1]
        grads[output == 0] = 0
        num_hidden = len(self.weights)-1
        for idx in range(num_hidden, -1, -1):
            if idx == 0:
                grads, weight_grad = self.last_backward(grads, self.weights[0], self.input)
                self.weights_grad.insert(0, weight_grad)
            else:
                grads, weight_grad = self.backward(grads, self.weights[idx], self.outputs[idx - 1])
                self.weights_grad.insert(0, weight_grad)
        return (grads, *self.weights_grad)