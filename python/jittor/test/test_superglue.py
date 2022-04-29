# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import os

from jittor.test.misc import superglue
from jittor.test.misc.superglue import SuperGlue
import time

@jt.flag_scope(use_cuda=1)
def main():
    global superglue
    superglue.split_size = int(os.environ.get("split_size", "12"))
    # superglue.split_size = 1000000

    batch = 30
    num = 2000
    dim = 128

    # jt.display_memory_info()
    # os.system("nvidia-smi")
    # breakpoint()

    with jt.no_grad():

        config = {
            'superglue': {
                'sinkhorn_iterations': 25,
                'match_threshold': 0.01,
                'keypoint_position_dim': 2,
                'descriptor_dim': dim,
                'use_dual_softmax': True,
                'GNN_layers': ['self', 'cross'] * 9,
            }
        }

        superglue = SuperGlue(config.get('superglue', {}))

        superglue.eval()

        data = {
            'keypoints0': jt.rand((batch, num, 2), dtype=jt.float),
            'keypoints1': jt.rand((batch, num, 2), dtype=jt.float),
            'shape0': jt.rand((batch, 2), dtype=jt.float),
            'shape1': jt.rand((batch, 2), dtype=jt.float),
            'descriptors0': jt.rand((batch, dim, num), dtype=jt.float),
            'descriptors1': jt.rand((batch, dim, num), dtype=jt.float),
            'scores0': jt.rand((batch, num), dtype=jt.float),
            'scores1': jt.rand((batch, num), dtype=jt.float),
            'all_matches': jt.randint(0, num, (batch, num, 2), dtype=jt.int),
            'return_match': False,
            # 'match_num': match_num
        }

        use_fp16 = int(os.environ.get("use_fp16", "0"))
        if use_fp16:
            jt.flags.amp_reg = 2
            for k,v in data.items():
                if isinstance(v, jt.Var) and v.dtype == "float32":
                    v.assign(v.float16())
            for v in superglue.parameters():
                if v.dtype == "float32":
                    v.assign(v.float16())
            jt.sync_all(True)

        import pickle
        jt.sync_all(True)
        for x in range(5):
            print(x)
            jt.gc()
            x = superglue(data)['loss']
            x.sync()
            jt.display_memory_info()
            # os.system("nvidia-smi")
            # breakpoint()
            # print(data)
            # print(x)
        
        # with open("/tmp/record.pkl", "wb") as f:
        #     pickle.dump([data, x], f, pickle.HIGHEST_PROTOCOL)

        # with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
        #     x = superglue(data)['loss']
        #     x.sync()
        #     jt.get_max_memory_treemap()
        # exit(0)

        jt.sync_all(True)
        time0 = time.time()
        jt.flags.profiler_enable = int(os.environ.get("profiler", "0"))

        for x in range(20):
            print(x)
            # jt.display_memory_info()
            x = superglue(data)['loss']
            x.sync()
            # print(x)

        jt.sync_all(True)
        time1 = time.time()
        print("avg time:", (time1 - time0) / 20)
        return (time1 - time0) / 20


class TestSuperglue(unittest.TestCase):
    def test(self):
        if not jt.has_cuda: return
        t1 = main()
        os.environ["use_fp16"] = "1"
        t2 = main()
        os.environ["use_fp16"] = "0"
        assert t1*0.55 > t2

if __name__ == "__main__":
    unittest.main()