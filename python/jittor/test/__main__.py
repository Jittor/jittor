# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

if __name__ == "__main__":
    import unittest, os

    suffix = "__main__.py"
    assert __file__.endswith(suffix)
    test_dir = __file__[:-len(suffix)]

    skip_l = int(os.environ.get("test_skip_l", "0"))
    skip_r = int(os.environ.get("test_skip_r", "1000000"))
    test_only = None
    if "test_only" in os.environ:
        test_only = set(os.environ.get("test_only").split(","))

    test_files = os.listdir(test_dir)
    test_files = sorted(test_files)
    suite = unittest.TestSuite()
    
    for _, test_file in enumerate(test_files):
        if not test_file.startswith("test_"):
            continue
        if _ < skip_l or _ > skip_r:
            continue
        test_name = test_file.split(".")[0]
        if test_only and test_name not in test_only:
            continue

        print("Add Test", _, test_name)
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(
            "jittor.test."+test_name))

    result = unittest.TextTestRunner(verbosity=3).run(suite)
    if len(result.errors) or len(result.failures):
        exit(1)
