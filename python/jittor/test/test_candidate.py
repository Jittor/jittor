# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

def check_candidate(x, fail_func):
    ans = []
    n = x.shape[0]
    for i in range(n):
        ok = True
        for j in range(len(ans)):
            if (fail_func(x[ans[j], :], x[i, :])):
                ok = False
                break
        if (ok):
            ans.append(i)
    return np.array(ans, dtype=int)

def check(shape, fail_cond, fail_func):
    a = jt.random(shape)
    selected = jt.candidate(a, fail_cond)
    a_ = a.data
    selected_out = selected.data
    selected_ans = check_candidate(a_, fail_func)
    assert selected_out.tolist() == selected_ans.tolist(), (selected_out, selected_ans)
        
def check1(selected, comer):
    return selected[0]>comer[0] or selected[1]>comer[1] or selected[2]>comer[2]
        
def check2(selected, comer):
    return selected[0]>comer[0] and selected[1]>comer[1] and selected[2]>comer[2]
        
def check3(selected, comer):
    threshold = 0.01
    s_1 = selected[2]*selected[3]
    s_2 = comer[2]*comer[3]
    s_inter_h = max(0,min(selected[2]+selected[0],comer[2]+comer[0])-max(selected[0],comer[0]))
    s_inter_w = max(0,min(selected[3]+selected[1],comer[3]+comer[1])-max(selected[1],comer[1]))
    s_inter = s_inter_h*s_inter_w
    iou = s_inter / (s_1 + s_2 - s_inter)
    return iou < threshold

class TestCandidateOp(unittest.TestCase):
    def test(self):
        # increse sequence
        check([100000,3], '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))or(@x(j,2)>@x(i,2))', check1)
        # no all increse sequence
        check([100000,3], '(@x(j,0)>@x(i,0))and(@x(j,1)>@x(i,1))and(@x(j,2)>@x(i,2))', check2)
        # nms
        # [x0, y0, h, w]
        threshold = '0.01'
        s_1 = '@x(j,2)*@x(j,3)'
        s_2 = '@x(i,2)*@x(i,3)'
        s_inter_h = 'std::max((Tx)0,std::min(@x(j,2)+@x(j,0),@x(i,2)+@x(i,0))-std::max(@x(j,0),@x(i,0)))'
        s_inter_w = 'std::max((Tx)0,std::min(@x(j,3)+@x(j,1),@x(i,3)+@x(i,1))-std::max(@x(j,1),@x(i,1)))'
        s_inter = s_inter_h+'*'+s_inter_w
        iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
        check([3000,4], iou+'<'+threshold, check3)

if __name__ == "__main__":
    unittest.main()