# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor_utils import run_cmd
import sys

class TestConsole(unittest.TestCase):
    def test_console(self):
        run_cmd(f"{sys.executable} -m jittor_utils.config --cxx-example > tmp.cc", jt.flags.cache_path)
        s = run_cmd(f"{jt.flags.cc_path} tmp.cc $({sys.executable} -m jittor_utils.config --include-flags --libs-flags --cxx-flags) -o tmp.out && ./tmp.out", jt.flags.cache_path)
        print(s)
        assert "jt.Var" in s
        assert "pred.shape 2 1000" in s

if __name__ == "__main__":
    unittest.main()