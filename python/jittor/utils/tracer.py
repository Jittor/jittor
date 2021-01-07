# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt

def fill_module_name(m, name):
    ps = []
    stack = []
    def callback(parents, k, v, n):
        stack.append(str(k))
        for k2, p in v.__dict__.items():
            if isinstance(p, jt.Var):
                ps.append(p)
                p.name(".".join(stack[1:]+[str(k2)]))
        v._trace_name = str(k)
    def callback_leave(parents, k, v, n):
        stack.pop()
    m.dfs([], name, callback, callback_leave)
