"""CUDA special-function kernels; shared CPU mathematics stays with math_util."""

from pathlib import Path

from jittor._runtime.dispatch import register_kernel


LGAMMA_CUDA_HEADER = '''
        __global__ void lgamma_cuda(float* __restrict__ x,
                                float* out,
                                int batch_shape)
        {
            int tidx = threadIdx.x;
            int start = batch_shape / blockDim.x * tidx;
            int end = threadIdx.x == blockDim.x - 1 ? batch_shape : start + batch_shape / blockDim.x;
            float* bx = x+batch_shape*blockIdx.x;
            float* bout = out + batch_shape * blockIdx.x;
            for(int i=start;i<end;i++) bout[i] = ::lgamma(bx[i]);
        }
        '''


LGAMMA_CUDA_SRC = '''
        @alias(x, in0)
        @alias(lx ,out0)
        int batch_size = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_shape0 * x_stride0 / batch_size;
        lgamma_cuda<<<batch_size, 16>>>(x_p, lx_p, batch_shape);
        '''


def polygamma_cuda_header(cpu_header):
    return "#define C10_HOST_DEVICE __host__ __device__\n" + cpu_header + '''
        __global__ void polygamma_cuda(float* __restrict__ x,
                        float* out,
                        int n,
                        int batch_shape)
        {
            int tidx = threadIdx.x;
            int start = batch_shape / blockDim.x * tidx;
            int end = threadIdx.x == blockDim.x - 1 ? batch_shape : start + batch_shape / blockDim.x;
            float* bx = x+batch_shape*blockIdx.x;
            float* bout = out + batch_shape * blockIdx.x;
            for(int i=start;i<end;i++)
                bout[i] = ((n % 2) ? 1.0 : -1.0) * ::exp(::lgamma(static_cast<scalar_t>(n) + 1.0)) *
                zeta<scalar_t>(static_cast<scalar_t>(n + 1), bx[i]);
        }
        '''


DIGAMMA_CUDA_HEADER = '''
        #define C10_HOST_DEVICE __host__ __device__

        template <typename T>
        C10_HOST_DEVICE static inline T polevl(const T x, const T A[], size_t len) {
        T result = 0;
        for (size_t i = 0; i <= len; i++) {
            result = result * x + A[i];
        }
        return result;
        }

        __device__ static inline float calc_digamma(float x) {
        // See [C++ Standard Reference: Gamma Function]
        static float PSI_10 = 2.25175258906672110764f;
        if (x == 0) {
            // As per C++ standard for gamma related functions and SciPy,
            // If the argument is ±0, ±∞ is returned
            return std::copysign(INFINITY, -x);
        }

        bool x_is_integer = x == truncf(x);
        if (x < 0) {
            if (x_is_integer) {
            // As per C++ standard for gamma related functions and SciPy,
            // If the argument is a negative integer, NaN is returned
            return std::numeric_limits<float>::quiet_NaN();
            }
            // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
            // accurate than tan(pi * x). While these operations are mathematically equivalent
            // since both x and r are in radians and tan() has a periodicity of pi, in practice
            // the computation of pi * x is a source of error (when |x| > 1).
            double q, r;
            r = std::modf(x, &q);
            float pi_over_tan_pi_x = (float)(M_PI / tan(M_PI * r));
            return calc_digamma(1 - x) - pi_over_tan_pi_x;
        }

        // Push x to be >= 10
        float result = 0;
        while (x < 10) {
            result -= 1 / x;
            x += 1;
        }
        if (x == 10) {
            return result + PSI_10;
        }

        // Compute asymptotic digamma
        static const float A[] = {
            8.33333333333333333333E-2f,
            -2.10927960927960927961E-2f,
            7.57575757575757575758E-3f,
            -4.16666666666666666667E-3f,
            3.96825396825396825397E-3f,
            -8.33333333333333333333E-3f,
            8.33333333333333333333E-2f,
        };

        float y = 0;
        if (x < 1.0e17f) {
            float z = 1 / (x * x);
            y = z * polevl(z, A, 6);
        }
        return result + logf(x) - (0.5f / x) - y;
        }

        __global__ void digamma_cuda(float* __restrict__ x,
                                float* out,
                                int batch_shape)
        {
            int tidx = threadIdx.x;
            int start = batch_shape / blockDim.x * tidx;
            int end = threadIdx.x == blockDim.x - 1 ? batch_shape : start + batch_shape / blockDim.x;
            float* bx = x+batch_shape*blockIdx.x;
            float* bout = out + batch_shape * blockIdx.x;
            for(int i=start;i<end;i++) bout[i] = calc_digamma(bx[i]);
        }
        '''


DIGAMMA_CUDA_SRC = '''
        @alias(x, in0)
        @alias(di_x, out0)
        int block_num = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_stride0 == 1 ? x_shape0: x_stride0;
        digamma_cuda<<<block_num, 16>>>(x_p, di_x_p, batch_shape);
        '''


def _gamma_cuda(owner, x):
    import jittor as jt
    return jt.code(x.shape, x.dtype, [x],
                   cuda_header=owner.cuda_header, cuda_src=owner.cuda_src)


def _digamma_cuda(owner, x):
    result = _gamma_cuda(owner, x)
    result.compile_options = {"FLAGS: --expt-relaxed-constexpr": 1}
    return result


def _polygamma_cuda(owner, x, n):
    import jittor as jt
    source = f'''
        @alias(x, in0)
        @alias(px ,out0)
        int batch_size = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_shape0 * x_stride0 / batch_size;
        polygamma_cuda<<<batch_size, 16>>>(x_p, px_p, {n}, batch_shape);
    '''
    return jt.code(x.shape, x.dtype, [x], cuda_header=owner.cuda_header, cuda_src=source)


def gamma_grad(x, alpha):
    import jittor as jt
    cuda_header = (Path(__file__).parent / "src" / "gamma_grad.h").read_text(encoding="utf8")
    cuda_src = '''
    @alias(x, in0)
    @alias(di_x, out0)
    int block_num = x_stride0 == 1 ? 1 : x_shape0;
    int batch_shape = x_stride0 == 1 ? x_shape0: x_stride0;
    float alpha = data["alpha"];
    gamma_grad_kenrel<<<block_num, 16>>>(x_p, di_x_p, alpha, batch_shape);
    '''
    grad = jt.code(x.shape, x.dtype, [x], cuda_header=cuda_header, cuda_src=cuda_src, data={"alpha":alpha})
    return grad


for _backend in ("cuda", "rocm_legacy", "corex_legacy"):
    register_kernel("math.lgamma", _backend, _gamma_cuda, dtypes={"float32"})
    register_kernel("math.digamma", _backend, _digamma_cuda, dtypes={"float32"})
    register_kernel("math.polygamma", _backend, _polygamma_cuda, dtypes={"float32"})
del _backend
