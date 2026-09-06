import jittor as jt
from jittor.backends.cuda.kernels.ccl.ccl_2d import label_image as _label_image


def ccl_2d(data_2d):
    ''' 
    2D connected component labelling, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm
    Args:
        [in]param data_2d: binary two-dimensional vector
            type data_2d: jittor array

    Returns:
        [out]result: labeled two-dimensional vector

    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> import cv2
    >>> import numpy as np
    >>> img = cv2.imread('testImg.png', 0)
    >>> a = img.mean()
    >>> img[img <= a] = 0
    >>> img[img > a] = 1
    >>> img = jt.Var(img)

    >>> result = ccl_2d(img)
    >>> print(jt.unique(result, return_counts=True, return_inverse=True)[0], jt.unique(result, return_counts=True, return_inverse=True)[2])
    >>> cv2.imwrite('testImg_result.png', result.numpy().astype(np.uint8) * 50)
    '''

    data_2d = data_2d.astype(jt.uint32)
    cY = data_2d.shape[0]
    cX = data_2d.shape[1]
    data_2d_copy = data_2d.clone()
    changed = jt.ones([1], dtype=jt.uint32)
    data_2d = data_2d.reshape(cX * cY)
    result = _label_image(data_2d, changed, cX, cY)
    result = result.reshape((cY, cX)) * data_2d_copy
    value = jt.unique(result)
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
