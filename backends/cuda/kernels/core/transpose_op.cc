#include "ops/transpose_op.h"
#include "var.h"
#include "executor.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace jittor {
#ifdef JIT_cuda
__global__ static void transpose_kernel(
    const Tx* __restrict__ xp,
    Tx* __restrict__ yp,
    index_t num,
    @for(i, 0, DIM, 1, index_t yshape@i, )
    @for(i, 0, DIM, 1, index_t ystride@i, )
    @for(i, 0, DIM, 1, index_t xstride@i, )
    int dummy
) {
    index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    index_t step = blockDim.x * gridDim.x;
    for (index_t yid = tid; yid < num; yid += step) {
        index_t t = yid;
        @for(i, 0, DIM,
            index_t yi@i = t / ystride@i;
            t -= yi@i * ystride@i;
        )
        index_t xid = @for(d, 0, DIM, + yi@{AXES@d} * xstride@d);
        yp[yid] = xp[xid];
    }
}

void TransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    index_t num = y->num;
    if (num == 0)
        return;
    @for(i, 0, DIM, index_t yshape@i = y->shape[@i];)
    index_t ystride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    @for(i, 0, DIM, index_t xshape@i = yshape@{AXES@i};)
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    int block = 256;
    int grid = (num + block - 1) / block;
    if (grid > 65535)
        grid = 65535;
    transpose_kernel<<<grid, block>>>(
        xp, yp, num,
        @for(i, 0, DIM, 1, yshape@i, )
        @for(i, 0, DIM, 1, ystride@i, )
        @for(i, 0, DIM, 1, xstride@i, )
        0
    );
}
#endif
} // namespace jittor
