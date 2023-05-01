import numpy as np
import jittor as jt
from jittor import nn

class lgamma(jt.Function):
    def __init__(self):
        self.cpu_src = '''
        @alias(x, in0)
        @alias(di_x, out0)
        int numel = x_shape0 * x_stride0;
        for(int i=0;i<numel;i++)
            di_x_p[i] = ::lgamma(x_p[i]);
        '''
        self.cuda_header = '''
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
        self.cuda_src = '''
        @alias(x, in0)
        @alias(lx ,out0)
        int batch_size = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_shape0 * x_stride0 / batch_size;
        lgamma_cuda<<<batch_size, 16>>>(x_p, lx_p, batch_shape);
        '''

    def execute(self, x):
        if jt.flags.use_cuda:
            return jt.code(x.shape, x.dtype, [x], cuda_header=self.cuda_header, cuda_src=self.cuda_src)
        else:
            return jt.code(x.shape, x.dtype, [x], cpu_src=self.cpu_src)

class polygamma(jt.Function):
    def __init__(self):
        self.cpu_header = '''
        #ifdef __CUDACC__
        #define C10_HOST_DEVICE __host__ __device__
        #else
        #define C10_HOST_DEVICE
        #endif

        template <typename scalar_t> C10_HOST_DEVICE static inline scalar_t zeta(scalar_t x, scalar_t q) {
            using acc_t = float;
            const acc_t MACHEP = acc_t{1.11022302462515654042E-16};
            constexpr acc_t zero = acc_t{0.0};
            constexpr acc_t half = acc_t{0.5};
            constexpr acc_t one = acc_t{1.0};
            static const acc_t A[] = {
                12.0,
                -720.0,
                30240.0,
                -1209600.0,
                47900160.0,
                -1.8924375803183791606e9, /*1.307674368e12/691*/
                7.47242496e10,
                -2.950130727918164224e12, /*1.067062284288e16/3617*/
                1.1646782814350067249e14, /*5.109094217170944e18/43867*/
                -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
                1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
                -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
            };

            int i = 0;
            acc_t a, b, k, s, t, w;
            if (x == one) {
                return std::numeric_limits<scalar_t>::infinity();
            }

            if (x < one) {
                return std::numeric_limits<scalar_t>::quiet_NaN();
            }

            if (q <= zero) {
                if (q == ::floor(q)) {
                return std::numeric_limits<scalar_t>::infinity();
                }
                if (x != ::floor(x)) {
                return std::numeric_limits<scalar_t>::quiet_NaN();
                }
            }

            s = ::pow(q, -x);
            a = q;
            i = 0;
            b = zero;
            while ((i < 9) || (a <= acc_t{9.0})) {
                i += 1;
                a += one;
                b = ::pow(a, -x);
                s += b;
                if ((-MACHEP * s < b) && (b < MACHEP * s)) {
                return static_cast<scalar_t>(s);
                }
            };

            w = a;
            s += b * w / (x - one);
            s -= half * b;
            a = one;
            k = zero;
            for (int i = 0; i < 12; i++) {
                a *= x + k;
                b /= w;
                t = a * b / A[i];
                s = s + t;
                t = ::fabs(t / s);
                if (t < MACHEP) {
                return static_cast<scalar_t>(s);
                }
                k += one;
                a *= x + k;
                b /= w;
                k += one;
            }
            return static_cast<scalar_t>(s);
        }
        using scalar_t = float;
        '''
        self.cuda_header = self.cpu_header + '''
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

    def execute(self, x, n):
        if jt.flags.use_cuda:
            self.cuda_src = f'''
                @alias(x, in0)
                @alias(px ,out0)
                int batch_size = x_stride0 == 1 ? 1 : x_shape0;
                int batch_shape = x_shape0 * x_stride0 / batch_size;
                polygamma_cuda<<<batch_size, 16>>>(x_p, px_p, {n}, batch_shape);
            '''
            return jt.code(x.shape, x.dtype, [x], cuda_header=self.cuda_header, cuda_src=self.cuda_src)
        else:
            self.cpu_src = f'''
                @alias(x, in0)
                @alias(px, out0)
                int numel = x_shape0 * x_stride0;
                for(int i=0;i<numel;i++) {{
                px_p[i] = (({n} % 2) ? 1.0 : -1.0) * ::exp(::lgamma(static_cast<scalar_t>({n}) + 1.0)) *
                zeta<scalar_t>(static_cast<scalar_t>({n} + 1), x_p[i]);
                }}
            '''
            return jt.code(x.shape, x.dtype, [x], cpu_header=self.cpu_header, cpu_src=self.cpu_src)

class digamma(jt.Function):
    '''
    digamma(x) = psi(x) = d/dx[ln(gamma(x))]
    '''
    def __init__(self):
        self.cpu_header = '''
        #include <cmath>
        #define C10_HOST_DEVICE
        template <typename T>
        C10_HOST_DEVICE static inline T polevl(const T x, const T A[], size_t len) {
        T result = 0;
        for (size_t i = 0; i <= len; i++) {
            result = result * x + A[i];
        }
        return result;
        }

        static inline float calc_digamma(float x) {
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
        '''
        self.cpu_src = '''
        @alias(x, in0)
        @alias(di_x, out0)
        int numel = x_shape0 * x_stride0;
        for(int i=0;i<numel;i++)
            di_x_p[i] = calc_digamma(x_p[i]);
        '''
        self.cuda_header = '''
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
        self.cuda_src = '''
        @alias(x, in0)
        @alias(di_x, out0)
        int block_num = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_stride0 == 1 ? x_shape0: x_stride0;
        digamma_cuda<<<block_num, 16>>>(x_p, di_x_p, batch_shape);
        '''
    
    def execute(self, x):
        self.input = x
        if jt.flags.use_cuda:
            dx = jt.code(x.shape, x.dtype, [x], cuda_header=self.cuda_header, cuda_src=self.cuda_src)
            dx.compile_options = {"FLAGS: --expt-relaxed-constexpr":1}
            return dx
        else:
            return jt.code(x.shape, x.dtype, [x], cpu_header=self.cpu_header, cpu_src=self.cpu_src)
    
    def grad(self, grad_d):
        return grad_d * polygamma.apply(self.input, 1)

def gamma_grad(x, alpha):
    cuda_header = open(os.path.join(os.path.realpath(os.path.dirname(__file__)), "src", "gamma_grad.h"), "r").read()
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

def sample_gamma(alpha, shape):
    cuda_header = '''
    #include <curand_kernel.h>

    template<typename scalar_t, typename accscalar_t>
    __device__ float sample_gamma(float alpha, curandState& state) {
        accscalar_t scale = 1.0f;

        // Boost alpha for higher acceptance probability.
        if (alpha < 1.0f) {
            if (alpha == 0.f) return 0.f;
            scale *= pow(1 - curand_uniform(&state), 1.0f / alpha);
            alpha += 1.0f;
        }

        // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
        // doi:10.1145/358407.358414
        const accscalar_t d = alpha - 1.0f / 3.0f;
        const accscalar_t c = 1.0f / sqrt(9.0f * d + 1e-8);
        for (;;) {
            accscalar_t x, y;
            do {
            x = curand_normal(&state);
            y = 1.0f + c * x;
            } while (y <= 0);
            const accscalar_t v = y * y * y;
            const accscalar_t u = 1 - curand_uniform(&state);
            const accscalar_t xx = x * x;
            if (u < 1.0f - 0.0331f * xx * xx)
                return static_cast<scalar_t>(scale * d * v);
            if (log(u) < 0.5f * xx + d * (1.0f - v + log(v)))
                return static_cast<scalar_t>(scale * d * v);
        }
    }

    __global__ void sample_gamma_kernel(float* out,
                            float alpha,
                            int seed,
                            int batch_shape) 
    {
        int tidx = threadIdx.x;
        int start = batch_shape / blockDim.x * tidx;
        int end = threadIdx.x == blockDim.x - 1 ? batch_shape : start + batch_shape / blockDim.x;
        if(start > end) 
            return;
        float* bout = out + batch_shape * blockIdx.x;
        curandState state;
        curand_init(clock64(), threadIdx.x, 0, &state);
        for(int i=start;i<end;i++) bout[i] = sample_gamma<float, float>(alpha, state);
    }
    '''
    cuda_src = '''
    @alias(lx ,out0)
    int batch_size = lx_stride0 == 1 ? 1 : lx_shape0;
    int batch_shape = lx_shape0 * lx_stride0 / batch_size;
    float alpha = data["alpha"];
    sample_gamma_kernel<<<batch_size, 16>>>(lx_p, alpha, time(NULL), batch_shape);
    '''
    samples = jt.code(shape, jt.float32, [], cuda_header=cuda_header, cuda_src=cuda_src, data={"alpha":alpha})
    return samples
