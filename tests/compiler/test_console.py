# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

from _helpers.child_process import PYTHON, shell

class TestConsole(unittest.TestCase):
    def test_console(self):
        # Both children import jittor_utils, so they have to reach this checkout.
        shell(f"{PYTHON} -m jittor_utils.config --cxx-example > tmp.cc",
              cwd=jt.flags.cache_path, merge_stderr=True, check=True)
        s = shell(
            f"{jt.flags.cc_path} tmp.cc "
            f"$({PYTHON} -m jittor_utils.config --include-flags --libs-flags --cxx-flags) "
            f"-o tmp.out && ./tmp.out",
            cwd=jt.flags.cache_path, merge_stderr=True, check=True).stdout
        print(s)
        assert "jt.Var" in s
        assert "pred.shape 2 1000" in s

if __name__ == "__main__":
    unittest.main()