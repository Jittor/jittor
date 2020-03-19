# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
from .test_log import find_log_with_re
from .test_fused_op import retry

# simple parser for parsing {a:1,b:2}
def simple_parser(s):
    ss = s.split(":")
    res = []
    for x in ss[:-1]:
        j = len(x)-1
        if j<0:
            res.append("")
            continue
        while j>=0 and x[j] in " \n":
            j -= 1
        k = j
        while k>=0 and x[k] not in " \n{},":
            k -= 1
        res.append(f'{x[:k+1]}"{x[k+1:j+1]}"{x[j+1:]}')
    res.append(ss[-1])
    res = ":".join(res)
    return eval(res)

gid = 0

class TestReorderTuner(unittest.TestCase):
    def test(self):
        a = jt.ones((8,8,8))
        a.data
        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100"
        ) as logs:
            b = a + a
            b.data
        
        logs = find_log_with_re(logs, 
            "Run tuner reorder: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1
        assert logs[0][0] == "1", "confidence of reorder should be 1"
        candidates = simple_parser(logs[0][1])
        assert candidates == {
            "order0":[0,], "order1":[0,1,], "order2":[0,1,2,]
        }

    def test_with_split(self):
        a = jt.ones((8,8,8))
        a.data
        global gid
        gid+=1
        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            compile_options={
                "split0": 4, "split1": 4, "split2": 4,
                "test_reorder_tuner":gid
            }
        ) as logs:
            b = a + a
            b.data
        
        logs = find_log_with_re(logs, 
            "Run tuner reorder: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1
        assert logs[0][0] == "1", "confidence of reorder should be 1"
        candidates = simple_parser(logs[0][1])
        assert candidates == {
            "order0":[0,], "order1":[0,1,], "order2":[0,1,2,], 
            "order3":[0,1,2,], "order4":[0,1,2,], "order5":[0,1,2,], 
        }, candidates

    @retry(10)
    def test_searcher(self):
        a = jt.ones((80,80,80))
        a.data
        global gid
        gid+=1
        with jt.log_capture_scope(
            log_v=0, log_vprefix="jit_searcher=1000",
            jit_search_kernel=1, 
            compile_options={
                "compile_shape":1,
                "test_reorder_tuner":gid
            }
        ) as logs:
            b = a + a
            b.data
        ls = find_log_with_re(logs, "Choices")
        assert len(ls) == 6, (ls, logs)
        ls = find_log_with_re(logs, "Best choices\\(.*\\): (.*)$")
        assert len(ls) == 1
        best = simple_parser(ls[0])
        assert best == {
            "compile_shape": 1, "order0": 0, "order1": 0, "order2": 0,
            "test_reorder_tuner":gid
        }



if __name__ == "__main__":
    unittest.main()