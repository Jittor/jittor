from jittor.backends.cuda.kernels.ccl.ccl_link import label_image as _label_image


def ccl_link(score_map, link_map, result_comp_area_thresh=6):
    """
    Find components in score map and link them with link map, original code from https://github.com/DanielPlayne/playne-equivalence-algorithm.
    Args:
        [in]param score_map: binary two-dimensional vector
            type score_map: jittor array
        [in]param link_map: two-dimensional vector with 8 channels
            type link_map: jittor array
        [in]param result_comp_area_thresh: threshold of component area
            type result_comp_area_thresh: int
    Returns:
        [out]result: labeled two-dimensional vector
    Example:
    >>> import jittor as jt
    >>> jt.flags.use_cuda = 1
    >>> import cv2
    >>> import numpy as np
    >>> score_map = jt.Var(np.load("score_map.npy"))
    >>> link_map = jt.Var(np.load("link_map.npy"))
    >>> score_map = score_map >= 0.5
    >>> link_map = link_map >= 0.8
    >>> for i in range(8):
    >>>     link_map[:, :, i] = link_map[:, :, i] & score_map

    >>> result = ccl_link(score_map, link_map)
    >>> cv2.imwrite('pixellink.png', result.numpy().astype(np.uint8) * 50)
    """
    import jittor as jt
    score_map = score_map.astype(jt.uint32)
    link_map = link_map.astype(jt.uint32)
    cY = score_map.shape[0]
    cX = score_map.shape[1]
    changed = jt.ones([1], dtype=jt.uint32)
    score_map = score_map.reshape(cX * cY)
    result = _label_image(score_map, link_map, changed, cX, cY)

    result = result.reshape((cY, cX))

    value, _, cnt = jt.unique(result, return_inverse=True, return_counts=True)
    value = (cnt > result_comp_area_thresh) * value
    value = value[value != 0]

    map_result = jt.zeros((int(value.max().numpy()[0]) + 1), dtype=jt.uint32)
    map_result[value] = jt.index(value.shape)[0] + 1
    result = map_result[result]

    return result
