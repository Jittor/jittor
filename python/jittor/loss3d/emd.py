# Author: Zheng-Ning Liu 
# 
# The gpu implementation is original provided by Haoqiang Fan and Kaichun Mo,
# <https://github.com/daerduoCarey/PyTorchEMD>.

import jittor as jt
from jittor import Function
from jittor.backends.cuda.kernels.loss3d import emd as _cuda_emd


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
        match = _cuda_emd.approximate_match(pc1, pc2, temp)

        emd = _cuda_emd.match_cost(pc1, pc2, match)

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

        grad_pc1 = _cuda_emd.match_cost_grad1(grad, pc1, pc2, match)

        grad_pc2 = _cuda_emd.match_cost_grad2(grad, pc1, pc2, match)

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
