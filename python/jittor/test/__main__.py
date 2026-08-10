# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

def main():
    import os
    import unittest

    if __package__:
        from ._runner import (
            load_suite,
            result_exit_code,
            run_separate_tests,
            select_tests,
            test_config_from_env,
        )
    else:
        from _runner import (
            load_suite,
            result_exit_code,
            run_separate_tests,
            select_tests,
            test_config_from_env,
        )

    unittest.TestLoader.sortTestMethodsUsing = None
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config = test_config_from_env(os.environ)
    selected_tests = select_tests(os.listdir(test_dir), **{
        "skip_l": config["skip_l"],
        "skip_r": config["skip_r"],
        "skip_markers": config["skip_markers"],
        "test_only": config["test_only"],
    })

    for selected in selected_tests:
        print("Add Test", selected.index, selected.name)

    suite = unittest.TestSuite()
    separate_status = 0
    if config["separate"]:
        import jittor_utils

        log_path = os.path.join(
            jittor_utils.home(), ".cache", "jittor", "test.log"
        )
        separate_status = run_separate_tests(
            [selected.module for selected in selected_tests],
            log_path,
        )
    else:
        load_suite(selected_tests, unittest.defaultTestLoader, suite)

    result = unittest.TextTestRunner(verbosity=3).run(suite)
    return result_exit_code(separate_status, result)


if __name__ == "__main__":
    import sys

    sys.exit(main())
