# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import random
import os

import numpy as np
import jittor.nn as jnn
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None
    skip_this_test = True

mid = 0
if "jittor" in os.uname()[1]:
    mid = 1

def resize_and_crop(x, bbox, interpolation="nearest", out_size=[224,224]):
    N, k = bbox.shape
    H, W, C = x.shape
    assert k==4
    shape = [N, out_size[0], out_size[1], C]
    # shape = [N,H,W]
    #      fx   x  cx
    #    +------------>
    # fy | a dx |  b
    #    | dy    
    #  y | -    o  -
    #    |
    # cy | c    |  d
    #    v
    img = x
    bb = [ bbox.reindex(shape, ["i0", str(i)]) for i in range(4) ]
    hid = jt.index(shape, dim=1)
    wid = jt.index(shape, dim=2)
    cid = jt.index(shape, dim=3)
    one = jt.array(1.0).broadcast(shape)
    x = bb[0]*(H-1.0)+hid*((H-1)*1.0/(shape[1]-1))*(bb[2]-bb[0])
    y = bb[1]*(W-1.0)+wid*((W-1)*1.0/(shape[2]-1))*(bb[3]-bb[1])
    if interpolation=="nearest":
        return img.reindex([x.round(), y.round(), cid])
    if interpolation=="bilinear":
        fx, fy = x.floor(), y.floor()
        cx, cy = fx+one, fy+one
        dx, dy = x-fx, y-fy
        a = img.reindex_var([fx, fy, cid])
        b = img.reindex_var([cx, fy, cid])
        c = img.reindex_var([fx, cy, cid])
        d = img.reindex_var([cx, cy, cid])
        dnx, dny = one-dx, one-dy
        ab = dx*b + dnx*a
        cd = dx*d + dnx*c
        o = ab*dny + cd*dy
        return o
    raise(f"Not support {interpolation}")

def test_case(box_num, out_size, time_limit):
    boxes = []
    for i in range(box_num):
        t = [random.random() * 0.9, random.random() * 0.9, random.random() * 0.9, random.random() * 0.9]
        t2 = [min(t[0], t[2]), min(t[1], t[3]), max(t[0], t[2]) + 0.1, max(t[1], t[3]) + 0.1]
        boxes.append(t2)
    img = jt.random([121, 121, 3])
    out = resize_and_crop(img, jt.array(boxes), interpolation='bilinear', out_size=out_size)
    with jt.profile_scope() as rep:
        our_out = out.data
    t = 0
    fused_op_num = 0
    for i in range(1, len(rep)):
        t += float(rep[i][3]) / 1e9
        name = rep[i][0]
        if name.startswith('[') and (not '[graph:]' in name):
            fused_op_num += 1
    assert fused_op_num == 1, fused_op_num
    assert t <= time_limit, t

def check_equal(arr, j_layer, p_layer):
    jittor_arr = jt.array(arr)
    pytorch_arr = torch.Tensor(arr)
    jittor_result = j_layer(jittor_arr)
    pytorch_result = p_layer(pytorch_arr)
    np.testing.assert_allclose(pytorch_result.detach().numpy(), jittor_result.numpy(), rtol=1e-6)

class TestResizeAndCrop(unittest.TestCase):
    def test(self):
        test_case(100, [224, 224], 0.45)
        test_case(100, [180, 224], 0.3)
        test_case(20, [1024, 1024], [1.2, 1.8][mid])
        test_case(20, [1024, 666], [0.8,1.0][mid])

    @unittest.skipIf(torch is None, "no torch found")
    def test_resize(self):
        import torch.nn.functional as F
        x = np.array(range(2*3*25)).reshape(2,3,5,5).astype("float32")
        for r_size in [3,4,5,6]:
            for align_corners in [True,False]:
                check_equal(x,
                    jnn.Resize((r_size, r_size), 'bilinear', align_corners),
                    lambda x: F.interpolate(x, size=(r_size, r_size), mode='bilinear',align_corners=align_corners))

    @unittest.skipIf(torch is None, "no torch found")
    def test_upsample(self):
        arr = np.random.randn(2,3,224,224)
        check_equal(arr, jnn.Upsample(scale_factor=2), tnn.Upsample(scale_factor=2))
        check_equal(arr, jnn.Upsample(scale_factor=0.5), tnn.Upsample(scale_factor=0.5))
        # pytorch change behav when scale_factor changed
        # this test cannot pass
        # check_equal(arr, jnn.Upsample(scale_factor=0.2), tnn.Upsample(scale_factor=0.2))

    @unittest.skipIf(torch is None, "no torch found")
    def test_pixelshuffle(self):
        arr = np.random.randn(2,4,224,224)
        check_equal(arr, jnn.PixelShuffle(upscale_factor=2), tnn.PixelShuffle(upscale_factor=2))
        arr = np.random.randn(1,3*3,224,224)
        check_equal(arr, jnn.PixelShuffle(upscale_factor=3), tnn.PixelShuffle(upscale_factor=3))

if __name__ == "__main__":
    unittest.main()
