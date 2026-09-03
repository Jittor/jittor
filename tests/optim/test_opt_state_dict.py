# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest


class TestOptStateDict(unittest.TestCase):
    @unittest.expectedFailure
    def test_opt_state_dict(self):
        self.fail("KI-TEST-001: optimizer state-dict coverage is not implemented")

if __name__ == "__main__":
    unittest.main()
