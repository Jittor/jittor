# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import unittest
import time
import jittor as jt
from jittor import LOG
import math
import numpy as np
from .test_core import expect_error
from .test_fused_op import retry

def check_cache_code(fname):
    check_code = True
    error_line_num = -1
    with open(fname) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if ('memory_checker->check_hit(' in lines[i]):
                continue
            code = lines[i]
            address_pos = []
            for j in range(len(code)):
                if code[j] == '[':
                    address_pos.append(j)
                if code[j] == ']':
                    sp = address_pos[-1] - 1
                    address_pos = address_pos[:-1]
                    if sp>=4 and code[sp-4:sp+1]=="shape":
                        continue
                    while ((sp >= 0) and ((code[sp] >= 'A' and code[sp] <= 'Z') or (code[sp] >= 'a' and code[sp] <= 'z') or 
                    (code[sp] >= '0' and code[sp] <= '9') or code[sp] == '_' or code[sp] == '.' or (sp > 0 and code[sp] == '>' and code[sp - 1] == '-'))):
                        if (sp > 0 and code[sp] == '>' and code[sp - 1] == '-'):
                            sp -= 2
                        else:
                            sp -= 1
                    sp += 1
                    check_var = code[sp:j + 1]
                    temp_i = i - 1
                    have_check = False
                    while (temp_i >= 0 and 'memory_checker->check_hit(' in lines[temp_i]):
                        if check_var in lines[temp_i]:
                            have_check = True
                            break
                        temp_i -= 1
                    if (not have_check):
                        check_code = False
                        error_line_num = i
                        break
            if (not check_code):
                break
    assert check_code, "check cache not found in line " + str(error_line_num) + " of file " + fname

class TestCache(unittest.TestCase):   
    def test_reduce(self):
        @retry(10)
        def check(n, m, reduce_dim, cache_report_, error_rate_threshold):
            a = jt.random([n,m])
            a.sync()
            with jt.profile_scope(compile_options = {
                "check_cache": 1, "replace_strategy": 1, "page_size": 4 << 10, #2 << 20
                "vtop": 0,
                "tlb_size": 64, "tlb_ways": 4, "tlb_line_size": 1,
                "L1_size": 32 << 10, "L1_ways": 8, "L1_line_size": 64,
                "L2_size": 256 << 10, "L2_ways": 8, "L2_line_size": 64,
                "L3_size": 15 << 20, "L3_ways": 20, "L3_line_size": 64
            }, enable_tuner=0) as report:
                c = a.sum(reduce_dim)
                c.sync()
            
            check_cache_code(report[1][1])
            cache_report = report[-1][-5:]
            for i in range(len(cache_report)):
                cache_report[i] = int(cache_report[i])
            for i in range(len(cache_report)):
                assert abs(cache_report[i] - cache_report_[i]) <= int(cache_report_[i] * error_rate_threshold), "cache report error: " + report[-2][-(len(cache_report) - i)] + " error, " + str(cache_report[i]) + "!=" + str(cache_report_[i])
        error_threshold = 0.02
        check(100, 10000, 0, [3010004, 989, 125729, 63129, 63129], error_threshold)
        check(100, 10000, 1, [3000104, 981, 62510, 62510, 62510], error_threshold)
        check(10, 98765, 0, [3061719, 2034, 129645, 129645, 67905], error_threshold)
        check(10, 98765, 1, [2962964, 969, 61733, 61733, 61733], error_threshold)
        check(7779, 97, 0, [2263790, 740, 47170, 47170, 47170], error_threshold)
        check(7779, 97, 1, [2271472, 748, 47650, 47650, 47650], error_threshold)
        check(1024, 1024, 0, [3146756, 1029, 65603, 65603, 65603], error_threshold)
        check(1024, 1024, 1, [3146756, 1028, 65603, 65603, 65603], error_threshold)
    
if __name__ == "__main__":
    unittest.main()
