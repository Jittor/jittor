# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

if __name__ == "__main__":
    import unittest

    suffix = "__main__.py"
    assert __file__.endswith(suffix)
    test_dir = __file__[:-len(suffix)]
    import os
    test_files = os.listdir(test_dir)
    for test_file in test_files:
        if not test_file.startswith("test_"):
            continue
        test_name = test_file.split(".")[0]
        exec(f"from . import {test_name}")
        test_mod = globals()[test_name]
        print(test_name)
        for i in dir(test_mod):
            obj = getattr(test_mod, i)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                globals()[test_name+"_"+i] = obj

    unittest.main()
