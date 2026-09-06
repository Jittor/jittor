import jittor as jt
from jittor.backends.cuda.kernels.ccl.ccl_3d import label_image as _label_image


def ccl_3d(data_3d):
    ''' 
    3D connected component labelling, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm
    Args:
        [in]param data_3d: binary three-dimensional vector
            type data_3d: jittor array

    Returns:
        [out]result : labeled three-dimensional vector

    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> data_3d = jt.zeros((10, 11, 12), dtype=jt.uint32)
    >>> data_3d[2:4, :, :] = 1
    >>> data_3d[5:7, :, :] = 1
    >>> result = ccl_3d(data_3d)
    >>> print(result[:, 0, 0])
    >>> print(
        jt.unique(result, return_counts=True, return_inverse=True)[0],
        jt.unique(result, return_counts=True, return_inverse=True)[2])
    '''

    data_3d = data_3d.astype(jt.uint32)
    cX = data_3d.shape[0]
    cY = data_3d.shape[1]
    cZ = data_3d.shape[2]
    changed = jt.ones([1], dtype=jt.uint32)
    data_3d_copy = data_3d.copy()
    data_3d = data_3d.reshape(cX * cY * cZ)
    result = _label_image(data_3d, changed, cX, cY, cZ)
    result = result.reshape((cX, cY, cZ)) * data_3d_copy
    value = jt.unique(result)
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
