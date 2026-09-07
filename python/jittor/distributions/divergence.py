# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import math
from .continuous import Normal, Uniform
from .discrete import Categorical, OneHotCategorical, Geometric, Bernoulli


def kl_divergence(cur_dist, old_dist):
    import jittor as jt
    assert isinstance(cur_dist, type(old_dist))
    if isinstance(cur_dist, Normal):
        vr = (cur_dist.sigma / old_dist.sigma)**2
        t1 = ((cur_dist.mu - old_dist.mu) / old_dist.sigma)**2
        return 0.5*(vr+t1-1-jt.safe_log(vr))
    if isinstance(cur_dist, Categorical) or isinstance(cur_dist,OneHotCategorical):
        t = cur_dist.probs * (cur_dist.logits-old_dist.logits)
        return t.sum(-1)
    if isinstance(cur_dist, Uniform):
        res = jt.safe_log((old_dist.high - old_dist.low) / (cur_dist.high - cur_dist.low))
        if old_dist.low > cur_dist.low or old_dist.high < cur_dist.high:
            res = math.inf
        return res
    if isinstance(cur_dist, Geometric):
        return -cur_dist.entropy() - jt.safe_log(-old_dist.prob+1) / cur_dist.prob - old_dist.logits
    if isinstance(cur_dist, Bernoulli):
        # KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        p, q = cur_dist.probs, old_dist.probs
        return p * (jt.safe_log(p) - jt.safe_log(q)) + (1 - p) * (jt.safe_log(1 - p) - jt.safe_log(1 - q))
    # No branch matched. Falling off the end returned None, which every caller
    # then fed into arithmetic; torch raises NotImplementedError for a pair it
    # has no registered formula for, and so do we.
    raise NotImplementedError(
        "kl_divergence is not implemented for %s; the supported distributions "
        "are Normal, Categorical, OneHotCategorical, Uniform, Geometric and "
        "Bernoulli" % type(cur_dist).__name__)
