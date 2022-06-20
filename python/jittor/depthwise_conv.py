# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers:
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import init
from jittor import nn
from jittor import Function

class DepthwiseConv(Function):
    def __init__(self, stride=1, padding=0, dilation=1):
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def execute(self, x, weight):
        self.save_vars = x, weight
        N,C,H,W = x.shape
        o,i,Kh,Kw = weight.shape
        assert(o == C)
        oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
        ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
        filter_height, filter_width = Kh, Kw
        self.Khw = Kh, Kw
        assert oh>0 and ow>0
        output = jt.code(
            [N, C, oh, ow],
            x.dtype,
            [x, weight],
            cuda_header = """
        template <typename T, 
            int filter_height,
            int filter_width, 
            int stride_height,
            int stride_width>
        __global__ void KernelDepthwiseConv(
            const T *const input_data, const T *const filter_data, const int batch_size,
            const int output_channels, const int output_height,
            const int output_width, const int input_channels,  
            const int input_height, const int input_width,     
            const int padding_height, const int padding_width, 
            const int dilate_height, const int dilate_width, T *const output_data) {
            const int kWeghtSize = filter_height * filter_width;
            T r_weight[kWeghtSize];
            const int batch = blockIdx.y;
            const int c_out = blockIdx.x;
            const T* weight = filter_data + c_out * filter_height * filter_width;
            for (int i = 0; i < filter_height * filter_width; i++) r_weight[i] = weight[i];

            for (int w_out = threadIdx.x; w_out < output_width; w_out += blockDim.x) {
                for (int h_out = threadIdx.y; h_out < output_height; h_out += blockDim.y) {
                    const int batch = blockIdx.y;
                    const int c_out = blockIdx.x;

                    const int c_in = c_out;
                    T value = 0;
                    const int h_in_start = -padding_height + h_out * stride_height;
                    const int w_in_start = -padding_width + w_out * stride_width;
                    const int h_in_end = h_in_start + filter_height * dilate_height;
                    const int w_in_end = w_in_start + filter_width * dilate_width;

                    const int in_offset =
                        ((batch * input_channels + c_in) * input_height) * input_width;

                    const int h_end = h_in_end < input_height ? h_in_end : input_height;
                    const int w_end = w_in_end < input_width ? w_in_end : input_width;
                    const int h_start = h_in_start > 0 ? h_in_start : 0;
                    const int w_start = w_in_start > 0 ? w_in_start : 0;

                    for (int h_in = h_in_start, h_f = 0; h_f < filter_height;
                        h_in += dilate_height, h_f++) {
                        for (int w_in = w_in_start, w_f = 0; w_f < filter_width;
                            w_in += dilate_width, w_f++) {
                            if (h_in >= 0 && h_in < input_height && w_in >= 0 &&
                                w_in < input_width) {
                                const int offset = in_offset + h_in * input_width + w_in;
                                value += r_weight[h_f * filter_width + w_f] * input_data[offset];
                            }
                        }
                    }
                    int index =
                        ((batch * gridDim.x + c_out) * output_height + h_out) * output_width +
                        w_out;
                    output_data[index] = value;
                }
            }
        }
        """,
        cuda_src=f"""
            @alias(input, in0)
            @alias(filter, in1)
            @alias(output, out)
            
            const int batch_size = input_shape0;
            const int input_channels = input_shape1;
            const int input_height = input_shape2;
            const int input_width = input_shape3;
            const int output_channels = output_shape1;
            const int output_height = output_shape2;
            const int output_width = output_shape3;
            const int ksize_height = {Kh};
            const int ksize_width = {Kw};
            const int stride_height = {self.stride[0]};
            const int stride_width = {self.stride[1]};
            const int padding_height = {self.padding[0]};
            const int padding_width = {self.padding[1]};
            const int dilate_height = {self.dilation[0]};
            const int dilate_width = {self.dilation[1]};

            int thread = 512;
            if (output_width > 1024 && output_width <= 2048)
                thread = (output_width - 1) / 2 + 1;
            else if (output_width > 512 && output_width <= 1024)
                thread = output_width;
            int blocks = std::min(std::max(thread / output_width, 1), output_height);
            dim3 threads(std::min(output_width, thread), blocks, 1);
            dim3 grid(output_channels, batch_size, 1);
            KernelDepthwiseConv<
                input_type, ksize_height, ksize_width, 
                stride_height, stride_width>
            <<<grid, threads>>>( 
                input_p, filter_p, batch_size, output_channels, output_height,
                output_width, input_channels, input_height, input_width,
                padding_height, padding_width, dilate_height,
                dilate_width, output_p);
        """
        )
        return output

    def grad(self, grad):
        x, weight = self.save_vars
        Kh, Kw = self.Khw
        return jt.code([x.shape, weight.shape], [x.dtype, weight.dtype], [x, weight, grad],
        cuda_header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"+"""
    template <typename T>
    __device__ __inline__ void CudaAtomicAddWithWarp(T* sum, T value) {
    typedef cub::WarpReduce<T> WarpReduce;
    typename WarpReduce::TempStorage temp_storage;
    value = WarpReduce(temp_storage).Sum(value);
    if (cub::LaneId() == 0) 
        atomicAdd(sum, value);
    }
    
    // CUDA kernel to compute the depthwise convolution backprop w.r.t input.
    template <typename T, 
        int filter_height,
        int filter_width, 
        int stride_height,
        int stride_width>
    __global__ void KernelDepthwiseConvInputGradCFilter(
        const T *const input_data, const T *const output_grad_data,
        const T *const filter_data, const int batch_size,   
        const int output_channels, const int output_height, 
        const int output_width, const int input_channels,   
        const int input_height, const int input_width,      
        const int padding_height, const int padding_width,  
        const int dilate_height, const int dilate_width,    
        T *const input_grad_data) {
        const int kWeghtSize = filter_height * filter_width + 1;
        T r_weight[kWeghtSize];
        const int batch = blockIdx.y;
        const int c_in = blockIdx.x;

        const T* weight = filter_data + c_in * filter_height * filter_width;
        for (int i = 0; i < filter_height * filter_width; i++)
            r_weight[i] =
                weight[filter_height * filter_width - i - 1];

        for (int w_in = threadIdx.x; w_in < input_width; w_in += blockDim.x) {
            for (int h_in = threadIdx.y; h_in < input_height; h_in += blockDim.y) {
                const int batch = blockIdx.y;
                const int c_in = blockIdx.x;

                int h_out_start = h_in - (filter_height - 1) * dilate_height + padding_height;

                int w_out_start = w_in - (filter_width - 1) * dilate_width + padding_width;

                T value = 0;
                int index =
                    ((batch * gridDim.x + c_in) * input_height + h_in) * input_width +
                    w_in;

                for (int h_out = h_out_start, h_f = 0; h_f < filter_height;
                    h_out += dilate_height, h_f++) {
                    for (int w_out = w_out_start, w_f = 0; w_f < filter_width;
                        w_out += dilate_width, w_f++) {
                        int s_h_out = h_out / stride_height;
                        int s_w_out = w_out / stride_width;
                        if (h_out % stride_height == 0 && w_out % stride_width == 0 &&
                            s_h_out >= 0 && s_h_out < output_height && s_w_out >= 0 &&
                            s_w_out < output_width) {
                        const int output_grad_offset =
                            ((batch * output_channels + c_in) * output_height +
                            s_h_out) *
                                output_width +
                            s_w_out;
                        value +=
                            output_grad_data[output_grad_offset] *
                            r_weight[h_f * filter_width + w_f];
                        }
                    }
                }
                input_grad_data[index] = value;
            }
        }
    }

    // Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
    template <typename T>
    __global__ void KernelDepthwiseConvFilterGrad(
        const T* output_grad_data, const T* input_data, const int num,
        const int output_channels, const int output_height, const int output_width,
        const int input_channels, const int input_height, const int input_width,
        const int filter_height,
        const int filter_width, const int stride_height, const int stride_width,
        const int padding_height, const int padding_width, const int dilate_height,
        const int dilate_width, T* filter_grad_data) {
        T s = 0;

        int gbid = (((blockIdx.z * blockDim.z + threadIdx.z) * gridDim.y) + blockIdx.y) * gridDim.x + blockIdx.x;

        for (int image_w = threadIdx.x; image_w < output_width;
            image_w += blockDim.x) {
            for (int bid = 0; bid < num; bid++) {
            //for (int bid = threadIdx.z; bid < num; bid+=blockDim.z) {
                for (int image_h = threadIdx.y; image_h < output_height;
                    image_h += blockDim.y) {
                    int kernel_id = blockIdx.z;
                    int kernel_h = blockIdx.y * dilate_height - padding_height;
                    int kernel_w = blockIdx.x * dilate_width - padding_width;

                    int image_hk = image_h * stride_height + kernel_h;
                    int image_wk = image_w * stride_width + kernel_w;
                    if (image_hk < 0 || image_hk >= input_height) continue;
                    if (image_wk < 0 || image_wk >= input_width) continue;
                    #define gaid(N, C, H, W) \
                    ((((N)*gridDim.z + (C)) * output_height + (H)) * output_width + (W))
                            int input_id = ((bid * gridDim.z +
                                            kernel_id) *
                                                input_height +
                                            image_hk) *
                                            input_width +
                                        image_wk;
                            s += output_grad_data[gaid(bid, kernel_id, image_h, image_w)] *
                                input_data[input_id];

                    #undef gaid
                }
            }
        }
        CudaAtomicAddWithWarp(&filter_grad_data[gbid], s);
    }
        """,
    cuda_src=f"""
    // source for backward to data
        @alias(input, in0)
        @alias(filter, in1)
        @alias(output_grad, in2)
        @alias(input_grad, out0)
        @alias(filter_grad, out1)

        const int batch_size = input_shape0;
        const int input_channels = input_shape1;
        const int input_height = input_shape2;
        const int input_width = input_shape3;
        const int output_channels = output_grad_shape1;
        const int output_height = output_grad_shape2;
        const int output_width = output_grad_shape3;
        const int ksize_height = {Kh};
        const int ksize_width = {Kw};
        const int stride_height = {self.stride[0]};
        const int stride_width = {self.stride[1]};
        const int padding_height = {self.padding[0]};
        const int padding_width = {self.padding[1]};
        const int dilate_height = {self.dilation[0]};
        const int dilate_width = {self.dilation[1]};

        int thread = 512;
        if (input_width > 1024 && input_width <= 2048)
        thread = (input_width - 1) / 2 + 1;
        else if (input_width > 512 && input_width <= 1024)
        thread = input_width;
        int blocks = std::min(std::max(thread / input_width, 1), input_height);
        dim3 threads(std::min(input_width, thread), blocks, 1);
        dim3 grid(input_channels, batch_size, 1);
        KernelDepthwiseConvInputGradCFilter<
            input_type, ksize_height, ksize_width
            , stride_height, stride_width>
            <<<grid, threads, 0>>>( 
            input_p, output_grad_p, filter_p, batch_size,          
            output_channels, output_height, output_width, input_channels,   
            input_height, input_width, padding_height,       
            padding_width, dilate_height, dilate_width, input_grad_p);   

    // source for backward to filter
    
        int block_size = 512;
        if (output_width > 1024 && output_width <= 2048)
        block_size = (output_width - 1) / 2 + 1;
        else if (output_width > 512 && output_width <= 1024)
        block_size = output_width;
        int crop_output_height =
            std::min(std::max(block_size / output_width, 1), output_height);

        grid = dim3(ksize_width, ksize_height, output_channels);
        threads = dim3(std::min(output_width, block_size), crop_output_height, 1);
        cudaMemsetAsync(filter_grad_p, 0, filter_grad->size);

        KernelDepthwiseConvFilterGrad<                                         
            input_type><<<grid, threads, 0>>>(      
            output_grad_p, input_p, batch_size, output_channels,           
            output_height, output_width, input_channels, input_height,           
            input_width, ksize_height, ksize_width,           
            stride_height, stride_width, padding_height, padding_width,          
            dilate_height, dilate_width, filter_grad_p);                      
    """
    )