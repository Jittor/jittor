# Author: Zheng-Ning Liu 
# 
# The gpu implementation is original provided by Haoqiang Fan and Kaichun Mo,
# <https://github.com/daerduoCarey/PyTorchEMD>.

import jittor as jt
from jittor import Function

EMD_gpu_header = '''
namespace jittor {
__device__ inline out_type dist2(out_type x1, out_type y1, out_type z1,
        out_type x2, out_type y2, out_type z2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
}
}
'''

approxmatch_gpu_src = '''
    __global__ void approxmatch_gpu_kernel(@ARGS_DEF) {
        @PRECALC
        @alias(xyz1, in0)
        @alias(xyz2, in1)
        @alias(match, out)

        int b = in0_shape0;
        int n = in0_shape1;
        int m = in1_shape1;

        out_type *remainL = in2_p + blockIdx.x * (n + m) * 2;
        out_type *remainR = remainL + n;
        out_type *ratioL = remainR + m;
        out_type *ratioR = ratioL + n;

        const int Block = 1024;
        __shared__ out_type buf[Block * 4];

        for (int i = blockIdx.x; i < b; i += gridDim.x) {
            for (int j = threadIdx.x; j < n * m; j += blockDim.x)
                match_p[i * n * m + j] = 0;
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                remainL[j] = n >= m ? 1 : m / n;
            for (int j = threadIdx.x; j < m; j += blockDim.x)
                remainR[j] = n >= m ? n / m : 1;
            __syncthreads();

            for (int j = 7; j >= -2; j--) {
                out_type level = j > -2 ? -powf(4.0f, j) : 0;

                for (int k0 = 0; k0 < n; k0 += blockDim.x) {
                    int k = k0 + threadIdx.x;
                    out_type x1 = 0, y1 = 0, z1 = 0;
                    if (k < n) {
                        x1 = @xyz1(i, k, 0);
                        y1 = @xyz1(i, k, 1);
                        z1 = @xyz1(i, k, 2);
                    }

                    out_type suml = 1e-9f;
                    for (int l0 = 0; l0 < m; l0 += Block){
                        int lend = min(m, l0 + Block) - l0;
                        for (int l = threadIdx.x; l < lend; l += blockDim.x) {
                            buf[l * 4 + 0] = @xyz2(i, l0 + l, 0);
                            buf[l * 4 + 1] = @xyz2(i, l0 + l, 1);
                            buf[l * 4 + 2] = @xyz2(i, l0 + l, 2);
                            buf[l * 4 + 3] = remainR[l0 + l];
                        }
                        __syncthreads();

                        for (int l = 0; l < lend; l++){
                            out_type x2 = buf[l * 4 + 0];
                            out_type y2 = buf[l * 4 + 1];
                            out_type z2 = buf[l * 4 + 2];
                            out_type d = level * dist2(x1, y1, z1, x2, y2, z2);
                            out_type w = __expf(d) * buf[l * 4 + 3];
                            suml += w;
                        }
                        __syncthreads();
                    }
                    if (k < n)
                        ratioL[k] = remainL[k] / suml;
                }
                __syncthreads();

                for (int l0 = 0; l0 < m; l0 += blockDim.x){
                    int l = l0 + threadIdx.x;
                    out_type x2 = 0, y2 = 0, z2 = 0;
                    if (l < m){
                        x2 = @xyz2(i, l, 0);
                        y2 = @xyz2(i, l, 1);
                        z2 = @xyz2(i, l, 2);
                    }
                    out_type sumr = 0;
                    for (int k0 = 0; k0 < n; k0 += Block){
                        int kend = min(n, k0 + Block) - k0;
                        for (int k = threadIdx.x; k < kend; k += blockDim.x){
                            buf[k * 4 + 0] = @xyz1(i, k0 + k, 0);
                            buf[k * 4 + 1] = @xyz1(i, k0 + k, 1);
                            buf[k * 4 + 2] = @xyz1(i, k0 + k, 2);
                            buf[k * 4 + 3] = ratioL[k0 + k];
                        }
                        __syncthreads();

                        for (int k = 0; k < kend; k++){
                            out_type x1 = buf[k * 4 + 0];
                            out_type y1 = buf[k * 4 + 1];
                            out_type z1 = buf[k * 4 + 2];
                            out_type d = level * dist2(x1, y1, z1, x2, y2, z2);
                            out_type w = __expf(d) * buf[k * 4 + 3];
                            sumr += w;
                        }
                        __syncthreads();
                    }

                    if (l < m){
                        sumr *= remainR[l];
                        out_type consumption = fminf(remainR[l] / (sumr + 1e-9f), 1.0f);
                        ratioR[l] = consumption * remainR[l];
                        remainR[l] = fmaxf(0.0f, remainR[l] - sumr);
                    }
                }
                __syncthreads();

                for (int k0 = 0; k0 < n; k0 += blockDim.x){
                    int k = k0 + threadIdx.x;
                    out_type x1 = 0, y1 = 0, z1 = 0;
                    if (k < n){
                        x1 = @xyz1(i, k, 0);
                        y1 = @xyz1(i, k, 1);
                        z1 = @xyz1(i, k, 2);
                    }
                    out_type suml = 0;
                    for (int l0 = 0; l0 < m; l0 += Block){
                        int lend = min(m, l0 + Block)-l0;
                        for (int l = threadIdx.x; l < lend; l += blockDim.x){
                            buf[l * 4 + 0] = @xyz2(i, l0 + l, 0);
                            buf[l * 4 + 1] = @xyz2(i, l0 + l, 1);
                            buf[l * 4 + 2] = @xyz2(i, l0 + l, 2);
                            buf[l * 4 + 3] = ratioR[l0 + l];
                        }
                        __syncthreads();

                        out_type rl = ratioL[k];
                        if (k < n){
                            for (int l = 0; l < lend; l++){
                                out_type x2 = buf[l * 4 + 0];
                                out_type y2 = buf[l * 4 + 1];
                                out_type z2 = buf[l * 4 + 2];
                                out_type d = level * dist2(x1, y1, z1, x2, y2, z2);
                                out_type w = __expf(d) * rl * buf[l*4+3];
                                @match(i, l0 + l, k) += w;
                                suml += w;
                            }
                        }
                        __syncthreads();
                    }
                    if (k < n)
                        remainL[k] = fmaxf(0.0f, remainL[k] - suml);
                }
                __syncthreads();
            }
        }
    }

    approxmatch_gpu_kernel<<<32, 512>>>(@ARGS);
'''

matchcost_gpu_src = '''
    __global__ void matchcost_gpu_kernel(@ARGS_DEF) {
        @PRECALC
        @alias(xyz1, in0)
        @alias(xyz2, in1)
        @alias(match, in2)

        int b = in0_shape0;
        int n = in0_shape1;
        int m = in1_shape1;

        const int Block = 1024;
        __shared__ out_type allsum[512];
        __shared__ out_type buf[Block * 3];

        for (int i = blockIdx.x; i < b; i += gridDim.x) {
            out_type subsum = 0;
            for (int k0 = 0; k0 < n; k0 += blockDim.x) {
                int k = k0 + threadIdx.x;
                out_type x1 = 0, y1 = 0, z1 = 0;
                if (k < n) {
                    x1 = @xyz1(i, k, 0);
                    y1 = @xyz1(i, k, 1);
                    z1 = @xyz1(i, k, 2);
                }

                for (int l0 = 0; l0 < m; l0 += Block) {
                    int lend = min(m, l0 + Block) - l0;
                    for (int l = threadIdx.x; l < lend * 3; l += blockDim.x)
                        buf[l] = xyz2_p[i * m * 3 + l0 * 3 + l];
                    __syncthreads();

                    if (k < n) {
                        for (int l = 0; l < lend; l++) {
                            out_type x2 = buf[l * 3 + 0];
                            out_type y2 = buf[l * 3 + 1];
                            out_type z2 = buf[l * 3 + 2];
                            out_type d = dist2(x1, y1, z1, x2, y2, z2);
                            subsum += d * @match(i, l0 + l, k);
                        }
                    }
                    __syncthreads();
                }
            }

            allsum[threadIdx.x] = subsum;
            for (int j = 1; j < blockDim.x; j <<= 1) {
                __syncthreads();
                if ((threadIdx.x & j) == 0 && threadIdx.x + j < blockDim.x) {
                    allsum[threadIdx.x] += allsum[threadIdx.x + j];
                }
            }

            if (threadIdx.x == 0)
                @out(i) = allsum[0];
            __syncthreads();
        }
    }

    matchcost_gpu_kernel<<<32, 512>>>(@ARGS);
'''

matchcost_grad1_gpu_src = '''
    __global__ void matchcost_grad1_gpu_kernel(@ARGS_DEF) {
        @PRECALC
        @alias(grad, in0)
        @alias(xyz1, in1)
        @alias(xyz2, in2)
        @alias(match, in3)

        int b = grad_shape0;
        int n = xyz1_shape1;
        int m = xyz2_shape1;

        for (int i = blockIdx.x; i < b ; i += gridDim.x){
            for (int l = threadIdx.x; l < n; l += blockDim.x){
                out_type x1 = @xyz1(i, l, 0);
                out_type y1 = @xyz1(i, l, 1);
                out_type z1 = @xyz1(i, l, 2);
                out_type dx = 0, dy = 0, dz = 0;
                for (int k = 0; k < m; k++){
                    out_type x2 = @xyz2(i, k, 0);
                    out_type y2 = @xyz2(i, k, 1);
                    out_type z2 = @xyz2(i, k, 2);
                    out_type d = @match(i, k, l) * 2;
                    dx += (x1 - x2) * d;
                    dy += (y1 - y2) * d;
                    dz += (z1 - z2) * d;
                }
                @out(i, l, 0) = dx * @grad(i);
                @out(i, l, 1) = dy * @grad(i);
                @out(i, l, 2) = dz * @grad(i);
            }
        }
    }

    matchcost_grad1_gpu_kernel<<<32, 512>>>(@ARGS);
'''

matchcost_grad2_gpu_src = '''
    __global__ void matchcost_grad2_gpu_kernel(@ARGS_DEF) {
        @PRECALC
        @alias(grad, in0)
        @alias(xyz1, in1)
        @alias(xyz2, in2)
        @alias(match, in3)

        int b = grad_shape0;
        int n = xyz1_shape1;
        int m = xyz2_shape1;

        __shared__ out_type sum_grad[256 * 3];
        for (int i = blockIdx.x; i < b; i += gridDim.x) {
            int kbeg = m * blockIdx.y / gridDim.y;
            int kend = m * (blockIdx.y + 1) / gridDim.y;
            for (int k = kbeg; k < kend; k++) {
                out_type x2 = @xyz2(i, k, 0);
                out_type y2 = @xyz2(i, k, 1);
                out_type z2 = @xyz2(i, k, 2);
                out_type subsumx = 0, subsumy = 0, subsumz = 0;
                for (int j = threadIdx.x; j < n; j += blockDim.x) {
                    out_type x1 = x2 - @xyz1(i, j, 0);
                    out_type y1 = y2 - @xyz1(i, j, 1);
                    out_type z1 = z2 - @xyz1(i, j, 2);
                    out_type d = @match(i, k, j) * 2;
                    subsumx += x1 * d;
                    subsumy += y1 * d;
                    subsumz += z1 * d;
                }
                sum_grad[threadIdx.x * 3 + 0] = subsumx;
                sum_grad[threadIdx.x * 3 + 1] = subsumy;
                sum_grad[threadIdx.x * 3 + 2] = subsumz;

                for (int j = 1; j < blockDim.x; j <<= 1) {
                    __syncthreads();
                    int j1 = threadIdx.x;
                    int j2 = threadIdx.x + j;
                    if ((j1 & j) == 0 && j2 < blockDim.x){
                        sum_grad[j1 * 3 + 0] += sum_grad[j2 * 3 + 0];
                        sum_grad[j1 * 3 + 1] += sum_grad[j2 * 3 + 1];
                        sum_grad[j1 * 3 + 2] += sum_grad[j2 * 3 + 2];
                    }
                }
                if (threadIdx.x == 0){
                    @out(i, k, 0) = sum_grad[0] * @grad(i);
                    @out(i, k, 1) = sum_grad[1] * @grad(i);
                    @out(i, k, 2) = sum_grad[2] * @grad(i);
                }
                __syncthreads();
            }
        }
    }

    matchcost_grad2_gpu_kernel<<<dim3(32, 32), 256>>>(@ARGS);
'''

class EarthMoverDistance(Function):
    ''' A loss layer that computes Earth Mover's distance from pc1 to pc2. Only supports GPU.

    :param pc1:  input point cloud
    :type pc1: jittor array

    :param pc2:  input point cloud
    :type pc2: jittor array

    :param reduction: reduction method in batches, can be 'mean', 'sum', or None. Default: 'mean'.
    :type reduction: str, optional
            
    :param dims: a string that represents each dimension, can be
            '[BNC]' ([batch, number of points, xyz]), or
            '[BCN]' ([batch, xyz, number of points]). Default: 'BNC'.
    :type dims: str, optional

    Example:

    >>> import jittor as jt
    >>> from jittor.loss3d import EarthMoverDistance
    >>> jt.flags.use_cuda = True
    >>> pc1 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> pc2 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> EMD = EarthMoverDistance(dims='BNC')
    >>> emd = EMD(pc1, pc2)
    >>> print('EMD =', emd.item())
    '''
    def execute(self, pc1, pc2, reduction='mean', dims='BNC'):
        assert dims in ['BNC', 'BCN']
        if dims == 'BCN':
            pc1, pc2 = pc1.permute(0, 2, 1), pc2.permute(0, 2, 1)

        batch_size_1, N, _ = pc1.shape
        batch_size_2, M, _ = pc2.shape
        assert batch_size_1 == batch_size_2
        batch_size = batch_size_1

        temp = jt.zeros([batch_size, (N + M) * 2], pc1.dtype)
        match = jt.code(
            shape=[batch_size, M, N],
            dtype=pc1.dtype,
            inputs=[pc1, pc2, temp],
            cuda_header=EMD_gpu_header,
            cuda_src=approxmatch_gpu_src,
        )

        emd = jt.code(
            shape=[batch_size],
            dtype=pc1.dtype,
            inputs=[pc1, pc2, match],
            cuda_header=EMD_gpu_header,
            cuda_src=matchcost_gpu_src,
        )

        self.saved_vars = (pc1, pc2, match, reduction)

        if reduction is None:
            return emd
        elif reduction == 'sum':
            return emd.sum()
        elif reduction == 'mean':
            return emd.mean()

    def grad(self, grad):
        pc1, pc2, match, reduction = self.saved_vars

        if reduction == 'sum':
            grad = jt.ones([pc1.shape[0]]) * grad
        elif reduction == 'mean':
            grad = jt.ones([pc1.shape[0]]) * grad / pc1.shape[0]

        grad_pc1 = jt.code(
            shape=pc1.shape,
            dtype=pc1.dtype,
            inputs=[grad, pc1, pc2, match],
            cuda_src=matchcost_grad1_gpu_src,
        )

        grad_pc2 = jt.code(
            shape=pc2.shape,
            dtype=pc2.dtype,
            inputs=[grad, pc1, pc2, match],
            cuda_src=matchcost_grad2_gpu_src,
        )

        return grad_pc1, grad_pc2


def earth_mover_distance(pc1, pc2, reduction='mean', dims='BNC'):
    ''' Earth Mover's distance from pc1 to pc2. Only supports GPU.

    :param pc1:  input point cloud
    :type pc1: jittor array

    :param pc2:  input point cloud
    :type pc2: jittor array

    :param reduction: reduction method in batches, can be 'mean', 'sum', or None. Default: 'mean'.
    :type reduction: str, optional
            
    :param dims: a string that represents each dimension, can be
            '[BNC]' ([batch, number of points, xyz]), or
            '[BCN]' ([batch, xyz, number of points]). Default: 'BNC'.
    :type dims: str, optional


    Example:

    >>> import jittor as jt
    >>> from jittor.loss3d import earth_mover_distance
    >>> jt.flags.use_cuda = True
    >>> pc1 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> pc2 = jt.rand([10, 100, 3], dtype=jt.float32)
    >>> emd = earth_mover_distance(pc1, pc2, dims='BNC')
    >>> print('EMD =', emd.item())
    '''
    return EarthMoverDistance.apply(pc1, pc2, reduction, dims)
