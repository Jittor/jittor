// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CandidateOp : Op {
    Var* x;
    Var* y;
    string fail_cond;
    /**
    Candidate Operator Perform an indirect candidate filter by given a fail condition.
    
    x is input, y is output index, satisfy::

        not fail_cond(y[0], y[1]) and
        not fail_cond(y[0], y[2]) and not fail_cond(y[1], y[2]) and
        ...
        ... and not fail_cond(y[m-2], y[m-1])

    Where m is number of selected candidates.

    Pseudo code::
    
        y = []
        for i in range(n):
            pass = True
            for j in y:
                if (@fail_cond):
                    pass = false
                    break
            if (pass):
                y.append(i)
        return y

    * [in] x:   input var for filter

    * [in] fail_cond:   code for fail condition

    * [in] dtype:   type of return indexes

    * [out] index: .

    Example::

        jt.candidate(jt.random(100,2), '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))')
        # return y satisfy:
        #    x[y[0], 0] <= x[y[1], 0] and x[y[1], 0] <= x[y[2], 0] and ... and x[y[m-2], 0] <= x[y[m-1], 0] and
        #    x[y[0], 1] <= x[y[1], 1] and x[y[1], 1] <= x[y[2], 1] and ... and x[y[m-2], 1] <= x[y[m-1], 1]
     */
    CandidateOp(Var* x, string&& fail_cond, NanoString dtype=ns_int32);
    void infer_shape() override;
    
    const char* name() const override { return "candidate"; }
    DECLARE_jit_run;
};

} // jittor