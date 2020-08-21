# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jiyan.ast as ja


def split_tuple_assign(self):
    '''This pass split tuple assign statement into multiple assign statements.

Example::

    import jiyan.ast as ja
    from jiyan.passes.split_tuple_assign import split_tuple_assign

    def test(shape):
        n,c,h,w = shape
        a,b = 1,2
        b,a = a,b

    jir = ja.AST.from_func(test)
    split_tuple_assign(jir)
    print(jir)

    # print following func
    def test(shape):
        n = shape[0]
        c = shape[1]
        h = shape[2]
        w = shape[3]
        a = 1
        b = 2
        _b = b
        b = a
        a = _b
    '''
    self.dfs(check)

def check(self):
    if not isinstance(self, ja.Assign):
        return
    t = self.targets[0]
    v = self.value
    if isinstance(v, (ja.Tuple, ja.List)):
        assert len(v.elts) == len(t.elts), (len(t), len(v.elts), t)
        code = []
        n = len(v.elts)
        # find used and replace with tmp backup
        # a,b = b,a ---> tmp = a; a,b = b,tmp
        for i in range(n):
            need_bk = False
            for j in range(i+1, n):
                s = t.elts[i].id
                e = v.elts[j]
                if e.check_name_used(s):
                    if not need_bk:
                        need_bk = True
                        bk_name = self.get_tmp_name('_'+s)
                        code.append(ja.Assign(bk_name, t.elts[i]))
                    e.replace_name({s:bk_name})
        # a,b = 1,2 ---> a=1; b=2
        for x, y in zip(t.elts, v.elts):
            code.append(ja.Assign(x, y))
        self.replace(code)
    elif isinstance(t, ja.Tuple):
        code = []
        if not isinstance(v, ja.Name):
            bk_name = self.get_tmp_name('__')
            code.append(ja.Assign(bk_name, v))
            v = ja.Name(bk_name)
        for i, x in enumerate(t.elts):
            code.append(ja.Assign(x, ja.Subscript(v, i)))
        self.replace(code)
