"""CUDA fallback kernels for native complex64/real-pair views."""

COMPLEX64_TO_REAL2_CUDA_SOURCE = """
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) { @out(i,0) = @in0(i).real; @out(i,1) = @in0(i).imag; }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);"""

REAL2_TO_COMPLEX64_CUDA_SOURCE = """
        __global__ void k(@ARGS_DEF) {
            @PRECALC
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if (i < in0_shape0) {
                @out(i) = complex64(float(@in0(i,0)), float(@in0(i,1)));
            }
        }
        int n = in0_shape0; k<<<(n+63)/64, 64>>>(@ARGS);"""
