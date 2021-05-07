# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt

def simple_presum(x):
    src = '''
__inline_static__
@python.jittor.auto_parallel(1)
void kernel(int n0, int i0, in0_type* x, in0_type* out, int nl) {
    out[i0*(nl+1)] = 0;
    for (int i=0; i<nl; i++)
        out[i0*(nl+1)+i+1] = out[i0*(nl+1)+i] + x[i0*(nl+1)+i];
}

kernel(in0->num/in0->shape[in0->shape.size()-1], 0, in0_p, out0_p, in0->num);
    '''
    return jt.code(x.shape[:-1]+(x.shape[-1]+1,), x.dtype, [x],
        cpu_src=src, cuda_src=src)


class OneHotCategorical:
    def __init__(self, probs=None, logits=None):
        assert not (probs is None and logits is None)
        if probs is None:
            # cannot align to pytorch
            probs = jt.sigmoid(logits)
        with jt.no_grad():
            self.probs = probs / probs.sum(-1, True)
            self.cum_probs = simple_presum(probs)
            self.cum_probs_l = self.cum_probs[..., :-1]
            self.cum_probs_r = self.cum_probs[..., 1:]

    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r).float()
        return one_hot
        


class Categorical:
    def __init__(self, probs=None, logits=None):
        OneHotCategorical.__init__(self, probs, logits)

    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r)
        index = one_hot.index(one_hot.ndim-1)
        return (one_hot * index).sum(-1)
