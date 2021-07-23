# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import numpy as np
from collections.abc import Sequence, Mapping
from PIL import Image
import time

def get_random_list(n):
    return list(np.random.permutation(range(n)))

def get_order_list(n):
    return [i for i in range(n)]


def collate_batch(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    real_size = len(batch)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, jt.Var):
        temp_data = jt.stack([data for data in batch], 0)
        return temp_data
    if elem_type is np.ndarray:
        temp_data = np.stack([data for data in batch], 0)
        return temp_data
    elif np.issubdtype(elem_type, np.integer):
        return np.int32(batch)
    elif isinstance(elem, int):
        return np.int32(batch)
    elif isinstance(elem, float):
        return np.float32(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Mapping):
        return {key: collate_batch([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return tuple(collate_batch(samples) for samples in transposed)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [collate_batch(samples) for samples in transposed]
    elif isinstance(elem, Image.Image):
        temp_data = np.stack([np.array(data) for data in batch], 0)
        return temp_data
    else:
        raise TypeError(f"Not support type <{elem_type.__name__}>")

class HookTimer:
    def __init__(self, obj, attr):
        self.origin = getattr(obj, attr)
        self.duration = 0.0
        setattr(obj, attr, self)

    def __call__(self, *args, **kw):
        start = time.time()
        rt = self.origin(*args, **kw)
        self.duration += time.time() - start
        return rt

