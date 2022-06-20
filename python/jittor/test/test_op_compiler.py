# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import LOG
import numpy as np
from .test_core import expect_error

jit_eval = jt.core.op_compiler.eval
jit_precompile = jt.core.op_compiler.precompile

class TestOpCompiler(unittest.TestCase):
    def test_eval(self):
        def check(expr, vars={}):
            for k,v in vars.items():
                locals()[k] = int(v)
            _v1 = None
            _v2 = None
            try:
                _v1 = jit_eval(expr, vars)
            except:
                pass
            try:
                _v2 = eval(expr)
            except:
                pass
            LOG.vv(f"check {expr} = {_v1}, {_v2}, {_v1 == _v2}")
            assert _v1 == _v2
        check("10+2*6")
        check("100 * 2 + 12")
        check("100*2+12")
        check("100 * ( 2 + 12 )")
        check("100*(2+12)")
        check("100 * ( 2 + 12 ) / 14")
        check("100*(2+12)/14")
        check("-1")
        check("- 1")
        vars = {"a":"123", "b":"2"}
        check("a", vars)
        check("a+b", vars)
        # python divide is different with c++
        # check("a/b", vars)
        check("-1 +a *b", vars)
        check("*****", vars)
        
    def test_precompile_ifdef(self):
        vars = {"JIT_a":"1"}
        check = lambda expr, result: \
            self.assertEqual(jit_precompile(vars, expr), result)
        check("#ifdef JIT_a\nxxx\n#endif", "xxx\n")
        check("#ifdef JIT_a\nxxx\n#else\nyyy\n #endif", "xxx\n")
        check("#ifndef JIT_a\nxxx\n#else\nyyy\n #endif", "yyy\n ")
        check("#ifdef JIT_b\nxxx\n#else\nyyy\n #endif", "yyy\n ")
        check("#ifdef b\nxxx\n#else\nyyy\n #endif",
              "#ifdef b\nxxx\n#else\nyyy\n #endif")
        for va in [0,1]:
            for vb in [0,1]:
                vars["JIT_a"] = "1"
                vars["JIT_b"] = "1"
                if not va:  del vars["JIT_a"]
                if not vb:  del vars["JIT_b"]
                check((
                    "#ifdef JIT_a\n"
                    "#ifdef JIT_b\n"
                    "0\n"
                    "#else\n"
                    "1\n"
                    "#endif\n"
                    "#else\n"
                    "#ifdef JIT_b\n"
                    "2\n"
                    "#else\n"
                    "3\n"
                    "#endif\n"
                    "#endif\n"
                ), f"{3 - (va*2+vb)}\n")
        
    def test_precompile(self):
        vars = {"a":"2", "b":"5", "a1":"1", "a2":"2", "OP":"mean"}
        check = lambda expr, result: \
            self.assertEqual(jit_precompile(vars, expr), result) 
        check("@", "@")
        check("@a", "2")
        # check("//@a\n@a", "//@a\n2")
        check("//@a\n@a", "\n2")
        # check("@a//@a", "2//@a")
        check("@a//@a", "2")
        check("@{-a +b* 2}", "8")
        # check("@{-a +b* 2}/*@{-a +b* 2}*/", "8/*@{-a +b* 2}*/")
        check("@{-a +b* 2}/*@{-a +b* 2}*/", "8")
        check("@for(i,a,b,+@i)", "+2+3+4")
        check("@for(i, a+1, b*2-3, -@{i*2})", " -6 -8 -10 -12")
        check("@for(i, b, a,-1,@i)", "543")
        check("@for(i, b, a,-1,@for(j,0,i,@i@j))", "505152535440414243303132")
        check("@{a@{a-1}+10}", "11")
        check("@{a@a}", "2")
        check("@if(0,1,0)", "0")
        check("@if(1,1,0)", "1")
        check("@if(0,1)", "")
        check("@if(1,1)", "1")
        check("@for(i,0,8,@if(i%2,+@i))", "+1+3+5+7")
        check("@{1<1}", "0")
        check("@{!1}", "0")
        check("@{!!1}", "1")
        check("@{!!1<<2}", "4")
        check("@{a<b*a1}", "1")
        check("@{a^b == 7}", "2")
        check("@{(a^b) == 7}", "1")
        check("@{b<<a == 5*4}", "1")
        expect_error(lambda: jit_precompile(vars, "@{a"))
        expect_error(lambda: jit_precompile(vars, "@for(a"))
        expect_error(lambda: jit_precompile(vars, "@for(i,l,r)"))
        expect_error(lambda: jit_precompile(vars, "@for(i,l,(@i,,,,))"))
        expect_error(lambda: jit_precompile(vars, "@for(i,0,10000,@i)"))
        expect_error(lambda: jit_precompile(vars, "@for(i,0,-1,@i)"))
        expect_error(lambda: jit_precompile(vars, "@asd"))
        expect_error(lambda: jit_precompile(vars, "@if"))
        expect_error(lambda: jit_precompile(vars, "@if(1,1,1,1)"))
        expect_error(lambda: jit_precompile(vars, "@if(1)"))
        expect_error(lambda: jit_precompile(vars, "#define OP1(a,b) a+b\n@expand_macro(OP1,1)"))

    def test_strcmp(self):
        vars = {"Tx":"float"}
        check = lambda expr, result: \
            self.assertEqual(jit_precompile(vars, expr), result)
        check("@strcmp(aaa,aaa)", "0")
        check("@strcmp(aaa,bbb)", "-1")
        check("@strcmp(ccc,bbb)", "1")
        check("@{@strcmp(aaa,aaa)}", "0")
        check("@{@strcmp(aaa,bbb)}", "-1")
        check("@{@strcmp(ccc,bbb)}", "1")

        code = \
"""@define(T_NCCL,
    @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, ncclFloat)
    @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, ncclInt)
    @if(@strcmp(@Tx,float64)==0, ncclFloat64)
    @if(@strcmp(@Tx,int64)==0, ncclInt64)
)
ncclBcast(..., @T_NCCL, ...)
"""
        assert "ncclFloat" in jit_precompile({"Tx":"float"}, code)
        assert "ncclFloat" in jit_precompile({"Tx":"float32"}, code)
        assert "ncclFloat64" in jit_precompile({"Tx":"float64"}, code)
        assert "ncclInt" in jit_precompile({"Tx":"int"}, code)
        assert "ncclInt" in jit_precompile({"Tx":"int32"}, code)
        assert "ncclInt64" in jit_precompile({"Tx":"int64"}, code)
        
    def test_mif(self):
        vars = {"Tx":"float"}
        check = lambda expr, result: \
            self.assertEqual(jit_precompile(vars, expr), result)
        check("#if aa>1\n@Tx\n#else\n@Tx@@1\n#endif", "#if aa>1\nfloat\n#else\nfloat1\n#endif")


if __name__ == "__main__":
    unittest.main()