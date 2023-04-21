import os

import numpy as np
import jittor as jt
from jittor import nn

def igamma(alpha, x):
    cuda_header = open(os.path.join(os.path.realpath(os.path.dirname(__file__)), "src", "igamma.h")).read()
    cuda_src = '''
        @alias(x, in0)
        @alias(px ,out0)
        int batch_size = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_shape0 * x_stride0 / batch_size;
        float alpha = data["alpha"];
        igamma_kernel<<<batch_size, 16>>>(x_p, px_p, alpha, batch_shape);   
    '''
    out = jt.code(x.shape, x.dtype, [x], cuda_header=cuda_header, cuda_src=cuda_src, data={"alpha": alpha})
    return out

if __name__ == "__main__":
    jt.flags.use_cuda = 1
    x = jt.array([[3.0010145,1.2304333,0.6540321,1.2690034,2.2610369,1.0665643],
        [2.895258, 1.3860005,1.7031978,1.1610458,1.1038274,1.3469076],
        [1.8637942,4.900345,1.8010278,3.946886, 4.9244375,3.9527063],
        [3.2821345,7.232482,3.6884353,3.9013095,1.8305631,1.5873598],
        [5.496692,0.8749316,3.0507984,0.48301435,3.5199432,8.469268],
        [1.1756407,2.9947124,4.2062683,4.733881,6.1016,6.1085234]])
    alpha = 3
    print(igamma(alpha, x))
