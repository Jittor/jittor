# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import LOG
import os
import re
import platform

class TestAsmTuner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        inline = "inline"
        if jt.flags.cc_type == "clang":
            inline = "__attribute__((always_inline))"
        self.cc_content='''
#include <cmath>
#include <algorithm>
#include "var.h"
#include "ops/broadcast_to_op.h"
#include "ops/binary_op.h"
#include "fused_op.h"
#define op0_Tx float32
#define op0_DIM 2
#define op0_BCAST 1
#define op0_index_t int32_t
#define op1_Tx float
#define op1_DIM 2
#define op1_BCAST 0
#define op1_index_t int32_t
#define op2_Tx float
#define op2_Ty float32
#define op2_Tz float32
#define op2_OP subtract
#define op2_index_t int32_t
using namespace jittor;
#define INLINE_FUNC '''+inline+''' void 
INLINE_FUNC func0(op0_Tx* __restrict__ op0_xp, op1_Tx* __restrict__ op1_xp, op2_Tz* __restrict__ op2_zp) {
    //@begin  replace "vmova(.*,.*\(.*\))" "vmovnt\g<1>"
    (void)(__builtin_assume_aligned(op0_xp, alignment));
    (void)(__builtin_assume_aligned(op1_xp, alignment));
    (void)(__builtin_assume_aligned(op2_zp, alignment));
    op2_index_t range0 = 1048576;
    op2_index_t range1 = 32;
    op0_index_t op0_xstride1 = 1;
    auto op0_xstride0 = op0_xstride1 * range1;
    op1_index_t op1_xstride1 = 1;
    auto op1_xstride0 = op1_xstride1 * range1;
    op2_index_t op2_zstride1 = 1;
    auto op2_zstride0 = op2_zstride1 * range1;
    for (op2_index_t id0 = 0; id0<range0; id0++) {
        for (op2_index_t id1 = 0; id1<range1; id1++) {
            auto op0_xid = + 0 * op0_xstride0 + id1 * op0_xstride1;
            auto op0_zd          = op0_xp[op0_xid];
            auto op1_xid = + id0 * op1_xstride0 + id1 * op1_xstride1;
            auto op1_zd          = op1_xp[op1_xid];
            op2_index_t op2_i = + id0 * op2_zstride0 + id1 * op2_zstride1;
            op2_zp[op2_i] = ((op1_zd       )-(op0_zd       ));
        }
    }
    //@end
}
void jittor::FusedOp::jit_run() {
    auto op0_x = ((BroadcastToOp*)(ops[0]))->x;
    auto op1_x = ((BroadcastToOp*)(ops[1]))->x;
    auto op2_z = ((BinaryOp*)(ops[2]))->z;
    auto* __restrict__ op0_xp = op0_x->ptr<op0_Tx>();
    auto* __restrict__ op1_xp = op1_x->ptr<op1_Tx>();
    auto* __restrict__ op2_zp = op2_z->ptr<op2_Tz>();
    func0(op0_xp,op1_xp,op2_zp);
}
        '''

        self.src_path=os.path.join(jt.flags.cache_path, 'jit', 'asm_test_op.cc')
        self.asm_path = os.path.join(jt.flags.jittor_path, "utils/asm_tuner.py")
        self.so_path=self.src_path.replace(".cc",".so")

    def run_cmd(self, cmd):
        return jt.compiler.run_cmd(cmd)

    def check_cc(self, content, check_movnt):
        LOG.vv("check_cc")
        with open(self.src_path,"w") as f:
            f.write(content)

        cmd = jt.flags.python_path + " " + \
            jt.flags.jittor_path+"/utils/asm_tuner.py --cc_path=" + jt.flags.cc_path + " '" + self.src_path + "'" + " -DJIT -DJIT_cpu " + jt.compiler.fix_cl_flags(jt.flags.cc_flags) + " -o '" + self.so_path + "'";
        self.run_cmd(cmd)

        s_path=self.so_path.replace(".so",".s")
        bo=False
        with open(s_path) as f:
            for line in f:
                if line.find("vmovnt")!=-1:
                    bo=True
                    break
        if check_movnt and jt.flags.cc_type == "clang":
            assert bo

    @unittest.skipIf(platform.system() == 'Darwin', 'will crash on macOS')
    def test_asm_tuner(self):
        self.check_cc(self.cc_content,True)
        self.check_cc(self.cc_content.replace("@begin","233").replace("@end","666"), False)

if __name__ == "__main__":
    unittest.main()
