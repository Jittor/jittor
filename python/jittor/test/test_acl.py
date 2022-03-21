# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from .test_core import expect_error
import numpy as np

@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):

    @jt.flag_scope(use_acl=1)
    def test_array(self):
        print("use_acl", jt.flags.use_acl)
        a = jt.array([1,2,3])
        np.testing.assert_allclose(a.numpy(), [1,2,3])

    @jt.flag_scope(use_acl=1)
    def test_add(self):
        a = jt.array([1,2,3])
        b = a+a
        np.testing.assert_allclose(b.numpy(), [2,4,6])

    def test_meminfo(self):
        jt.display_memory_info()

if __name__ == "__main__":
    unittest.main()
