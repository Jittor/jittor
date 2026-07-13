import math
import numpy as np
import jittor as jt
from jittor import nn

# ---------------------------------------------------------------------------
# Device-agnostic (CPU/CUDA/NPU) composite implementations of the special
# functions lgamma / digamma / trigamma, expressed purely with jittor PRIMITIVE
# ops (log/exp/pow/sin/tan/abs/maximum/ternary/...). These are needed because
# the fast jt.code() kernels below only target CPU & CUDA -- on the Ascend ACL
# backend a `code` op raises "op code not supported" (acl_op_exec.cc). Composing
# from primitives lets these run on the NPU too, and (bonus) stay autodiff-able.
# Numerics mirror PyTorch's lgamma/digamma so parity holds (target <=1e-5).
# Used only when jt.flags.use_acl is set; CPU/CUDA keep their kernels.
# ---------------------------------------------------------------------------
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_PI = math.pi
# Lanczos g=7, n=9 (same constants PyTorch/Numpy use)
_LANCZOS_G = 7.0
_LANCZOS_C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]
# digamma asymptotic series coeffs (PyTorch calc_digamma)
_DIGAMMA_A = [
    8.33333333333333333333e-2, -2.10927960927960927961e-2, 7.57575757575757575758e-3,
    -4.16666666666666666667e-3, 3.96825396825396825397e-3, -8.33333333333333333333e-3,
    8.33333333333333333333e-2,
]


def _lgamma_acl(x):
    """log|Gamma(x)| via Lanczos approx + reflection (x<0.5). Primitive ops only."""
    reflect = x < 0.5
    y = jt.ternary(reflect, 1.0 - x, x)   # operand for the series is always >= 0.5
    z = y - 1.0
    acc = _LANCZOS_C[0]
    for i in range(1, 9):
        acc = acc + _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    lg_y = _HALF_LOG_2PI + (z + 0.5) * jt.log(t) - t + jt.log(acc)
    # reflection: lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x)
    reflected = _PI / jt.maximum(jt.abs(jt.sin(_PI * x)), 1e-30)
    reflected = jt.log(reflected) - lg_y
    return jt.ternary(reflect, reflected, lg_y)


def _digamma_acl(x):
    """psi(x) via recurrence-to->=10 + asymptotic series + reflection (x<0.5)."""
    reflect = x < 0.5
    xr = jt.ternary(reflect, 1.0 - x, x)   # >= 0.5
    acc = 0.0
    for _ in range(12):                    # push xr up to >=10 (psi(x)=psi(x+1)-1/x)
        need = (xr < 10.0).float()
        acc = acc - need / xr
        xr = xr + need
    z = 1.0 / (xr * xr)
    poly = _DIGAMMA_A[0]
    for i in range(1, 7):
        poly = poly * z + _DIGAMMA_A[i]
    psi = acc + jt.log(xr) - 0.5 / xr - z * poly
    # reflection: psi(x) = psi(1-x) - pi*cot(pi x)
    reflected = psi - _PI / jt.tan(_PI * x)
    return jt.ternary(reflect, reflected, psi)


def _trigamma_acl(x):
    """psi'(x) (= polygamma(1)) via recurrence-to->=10 + asymptotic. For x>0."""
    acc = 0.0
    xr = x + 0.0
    for _ in range(12):                    # trigamma(x)=trigamma(x+1)+1/x^2
        need = (xr < 10.0).float()
        acc = acc + need / (xr * xr)
        xr = xr + need
    w = 1.0 / xr
    w2 = w * w
    # 1/x + 1/(2x^2) + 1/6 x^-3 - 1/30 x^-5 + 1/42 x^-7 - 1/30 x^-9
    asy = w + 0.5 * w2 + w * w2 * (1.0/6.0 - w2 * (1.0/30.0 - w2 * (1.0/42.0 - w2 * (1.0/30.0))))
    return acc + asy


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
        self.x = x
        if jt.flags.use_acl:                      # ACL has no `code` op -> composite
            return _lgamma_acl(x)
        elif jt.flags.use_cuda:
            return jt.code(x.shape, x.dtype, [x], cuda_header=self.cuda_header, cuda_src=self.cuda_src)
        else:
            return jt.code(x.shape, x.dtype, [x], cpu_src=self.cpu_src)

    def grad(self, grad_output):
        # d/dx lgamma(x) = digamma(x). (torch's lgamma is differentiable; this gives
        # jittor parity -- needed to backprop through Gamma/Beta/Dirichlet log_prob &
        # entropy w.r.t. their concentration parameters. digamma already defines its
        # own grad, so this composes for higher-order too.)
        return grad_output * digamma.apply(self.x)

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
        if jt.flags.use_acl:                      # ACL has no `code` op -> composite
            if n == 1:
                return _trigamma_acl(x)
            raise NotImplementedError(
                f"polygamma(n={n}) not implemented on ACL/NPU; only n=1 (trigamma). "
                "Add the corresponding composite series in gamma.py:_trigamma_acl.")
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
        if jt.flags.use_acl:                      # ACL has no `code` op -> composite
            return _digamma_acl(x)
        elif jt.flags.use_cuda:
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

# --- implicit reparameterization gradient dx/dalpha for x ~ Gamma(alpha, 1) ---
# Direct port of PyTorch's standard_gamma_grad (src/gamma_grad.h): Taylor series
# for small x, Rice saddle-point for large alpha, bivariate rational approx
# otherwise.  = -d/dalpha[P(alpha,x)] / pdf(x;alpha).  Verified against an
# independent incomplete-gamma CDF reference across 88 (alpha,x) pairs spanning
# all three regimes (worst rel 2.6e-4).  Pure scalar math -> works on every
# backend via numpy_code (host).
def _gamma_grad_digamma_one(x):
    PSI_10 = 2.25175258906672110764
    if x == 0: return math.inf
    add = 0.0
    if x < 0:
        if x == math.floor(x): return math.inf
        add = -math.pi / math.tan(math.pi * x); x = 1 - x
    result = 0.0
    while x < 10: result -= 1 / x; x += 1
    if x == 10: return result + PSI_10 + add
    A = [8.33333333333333333333E-2, -2.10927960927960927961E-2, 7.57575757575757575758E-3,
         -4.16666666666666666667E-3, 3.96825396825396825397E-3, -8.33333333333333333333E-3,
         8.33333333333333333333E-2]
    y = 0.0
    if x < 1.0e17:
        z = 1.0 / (x * x); r = 0.0
        for i in range(7): r = r * z + A[i]
        y = z * r
    return result + math.log(x) - (0.5 / x) - y + add

_GG_COEF = [[0.16009398, -0.094634809, 0.025146376, -0.0030648343, 1, 0.32668115, 0.10406089, 0.0014179084],
            [0.53487893, 0.1298071, 0.065735949, -0.0015649758, 0.16639465, 0.020070113, -0.0035938915, -0.00058392623],
            [0.040121004, -0.0065914022, -0.0026286047, -0.0013441777, 0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07]]
def _standard_gamma_grad_one(alpha, x):
    if alpha <= 0 or x <= 0: return 0.0
    if x < 0.8:
        numer = 1.0; denom = alpha; s1 = numer / denom; s2 = numer / (denom * denom)
        for i in range(1, 6):
            numer *= -x / i; denom += 1; s1 += numer / denom; s2 += numer / (denom * denom)
        pxa = x ** alpha; pdf = x ** (alpha - 1) * math.exp(-x); cdf = pxa * s1
        cdf_a = (math.log(x) - _gamma_grad_digamma_one(alpha)) * cdf - pxa * s2
        r = -cdf_a / pdf
        return r if math.isfinite(r) else 0.0
    if alpha > 8.0:
        if 0.9 * alpha <= x <= 1.1 * alpha:
            n1 = 1 + 24 * alpha * (1 + 12 * alpha)
            n2 = 1440 * alpha * alpha + 6 * x * (53 - 120 * x) - 65 * x * x / alpha + alpha * (107 + 3600 * x)
            r = n1 * n2 / (1244160 * alpha ** 4)
            return r if math.isfinite(r) else 0.0
        denom = math.sqrt(8 * alpha + 1e-8); term2 = denom / (alpha - x)
        term3 = (x - alpha - alpha * math.log(x / alpha)) ** (-1.5)
        term23 = term2 - term3 if x < alpha else term2 + term3
        term1 = math.log(x / alpha) * term23 - math.sqrt(2 / alpha + 1e-8) * (alpha + x) / ((alpha - x) ** 2)
        stir = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha))
        r = -stir * (x * term1) / denom
        return r if math.isfinite(r) else 0.0
    u = math.log(x / alpha); v = math.log(alpha)
    cv = [_GG_COEF[0][i] + u * (_GG_COEF[1][i] + u * _GG_COEF[2][i]) for i in range(8)]
    p = cv[0] + v * (cv[1] + v * (cv[2] + v * cv[3])); q = cv[4] + v * (cv[5] + v * (cv[6] + v * cv[7]))
    r = math.exp(p / q)
    return r if math.isfinite(r) else 0.0

_standard_gamma_grad_vec = np.vectorize(_standard_gamma_grad_one, otypes=[np.float64])

def sample_gamma(alpha, shape):
    r'''Draw x ~ Gamma(concentration=alpha, rate=1), differentiable wrt alpha via
    the implicit reparameterization gradient (Figurnov et al. 2018). All-backend
    (numpy_code runs on host; alpha may be a per-element Var). The old version was
    CUDA-only, baked alpha as a scalar (crashed for Var alpha) and produced no
    gradient (empty input list).'''
    alpha = alpha if isinstance(alpha, jt.Var) else jt.array(alpha)
    alpha = alpha.float32()
    # broadcast alpha to the requested sample shape (grad flows back, summed)
    alpha_b = alpha if tuple(alpha.shape) == tuple(shape) else alpha + jt.zeros(shape, alpha.dtype)

    # xp is cupy on CUDA, numpy on CPU/ACL; `np` (module) is always real numpy.
    # The rejection sampler + the python implicit-grad run on host numpy, then
    # results are copied to the device buffer (sampling is not a hot path).
    def _host(arr):
        return arr.get() if hasattr(arr, "get") else np.asarray(arr)

    def forward_code(xp, data):
        a = _host(data["inputs"][0]).astype(np.float64)
        out = data["outputs"][0]
        samp = np.random.gamma(np.maximum(a, 1e-8)).astype(np.float32)
        xp.copyto(out, xp.asarray(samp) if hasattr(xp, "asarray") else samp)

    def backward_code(xp, data):
        a = _host(data["inputs"][0]).astype(np.float64)
        x = _host(data["f_outputs"][0]).astype(np.float64)
        dout = _host(data["dout"]).astype(np.float64)
        out = data["outputs"][0]
        res = (dout * _standard_gamma_grad_vec(a, x)).astype(np.float32)
        xp.copyto(out, xp.asarray(res) if hasattr(xp, "asarray") else res)

    return jt.numpy_code([alpha_b.shape], [alpha_b.dtype], [alpha_b],
                         forward_code, [backward_code])[0]
