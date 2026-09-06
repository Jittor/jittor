# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os

from _helpers.onednn import requires_onednn


class TestMklTestOp(unittest.TestCase):
    def test(self):
        # `jt.mkl_ops` is a query, not an accessor: it is None until the lazy
        # loader has fired, so this read used to raise AttributeError rather
        # than run. See tests/_helpers/onednn.py.
        mkl_ops = requires_onednn()
        assert mkl_ops.mkl_test().data==123

if __name__ == "__main__":
    unittest.main()
