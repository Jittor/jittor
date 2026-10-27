"""CUDA BF16 training fast path for exact GELU."""


_gelu_training_cuda = None


def _install_gelu_training_cuda(jt, output_requires_grad, optional_kernel):
    """Register the kernel without adding this module to Jittor's import cycle."""

    def supports(x, approximate="none"):
        return (
            approximate == "none"
            and isinstance(x, jt.Var)
            and output_requires_grad(x)
            and x.numel() > 0
            and x._storage_is_contiguous()
        )

    class GeluTrainingCUDA(jt.Function):
        def execute(self, x):
            self.x = x
            return jt.code(
                x.shape,
                x.dtype,
                [x],
                cuda_src=r"""
                __global__ static void gelu_forward(
                        const in0_type* x, out0_type* y, long long n) {
                    long long start =
                        (long long)blockIdx.x * blockDim.x + threadIdx.x;
                    long long stride = (long long)blockDim.x * gridDim.x;
                    for (long long i = start; i < n; i += stride) {
                        float v = static_cast<float>(x[i]);
                        y[i] = out0_type(
                            (0.5f * v)
                            * (1.0f + erff(v * 0.7071067811865476f)));
                    }
                }
                long long n = in0->num;
                long long blocks = (n + 255) / 256;
                if (blocks > 65536) blocks = 65536;
                gelu_forward<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
                """,
            )

        def grad(self, grad_y):
            x = self.x
            if not grad_y._storage_is_contiguous():
                grad_y = grad_y.contiguous()
            return jt.code(
                x.shape,
                x.dtype,
                [grad_y, x],
                cuda_src=r"""
                __global__ static void gelu_backward(
                        const in0_type* grad_y, const in1_type* x,
                        out0_type* grad_x, long long n) {
                    long long start =
                        (long long)blockIdx.x * blockDim.x + threadIdx.x;
                    long long stride = (long long)blockDim.x * gridDim.x;
                    for (long long i = start; i < n; i += stride) {
                        float v = static_cast<float>(x[i]);
                        float cdf =
                            0.5f * (1.0f + erff(v * 0.7071067811865476f));
                        float pdf =
                            0.3989422804014327f * expf(-0.5f * v * v);
                        grad_x[i] = out0_type(
                            static_cast<float>(grad_y[i])
                            * (cdf + v * pdf));
                    }
                }
                long long n = in0->num;
                long long blocks = (n + 255) / 256;
                if (blocks > 65536) blocks = 65536;
                gelu_backward<<<(unsigned)blocks, 256>>>(
                    in0_p, in1_p, out0_p, n);
                """,
            )

    @optional_kernel(
        "nn.gelu.training",
        "cuda",
        dtypes=("bfloat16",),
        supports=supports,
    )
    def gelu_training_cuda(x, approximate="none"):
        return GeluTrainingCUDA.apply(x)

    global _gelu_training_cuda
    _gelu_training_cuda = gelu_training_cuda
    return gelu_training_cuda
