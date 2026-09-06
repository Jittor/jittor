# Author: Zheng-Ning Liu
#
# CUDA nearest-neighbor kernel for Chamfer loss.
# The CPU kernel and shared loss mathematics live in jittor.loss3d.chamfer.
# The implementation does no use extra NxM matrix to store distances, and thus
# supports large point clouds.

cuda_src = '''
    __global__ void chamfer_loss_min_idx_kernel(@ARGS_DEF) {
        @PRECALC
        int bs = blockIdx.x;
        int n = in0_shape1;
        int m = in1_shape1;

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float min_dis = (@in0(bs, i, 0) - @in1(bs, 0, 0)) * (@in0(bs, i, 0) - @in1(bs, 0, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, 0, 1)) * (@in0(bs, i, 1) - @in1(bs, 0, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, 0, 2)) * (@in0(bs, i, 2) - @in1(bs, 0, 2));
            @out(bs, i) = 0;
            for (int j = 1; j < m; ++j) {
                float dis = (@in0(bs, i, 0) - @in1(bs, j, 0)) * (@in0(bs, i, 0) - @in1(bs, j, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, j, 1)) * (@in0(bs, i, 1) - @in1(bs, j, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, j, 2)) * (@in0(bs, i, 2) - @in1(bs, j, 2));
                if (dis < min_dis) {
                    min_dis = dis;
                    @out(bs, i) = j;
                }
            }
        }
    }

    chamfer_loss_min_idx_kernel<<<in0_shape0, 512>>>(@ARGS);
'''


def build_sources():
    return {"cuda_src": cuda_src}
