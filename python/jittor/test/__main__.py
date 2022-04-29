# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

if __name__ == "__main__":
    import unittest, os
    unittest.TestLoader.sortTestMethodsUsing = None

    suffix = "__main__.py"
    assert __file__.endswith(suffix)
    test_dir = __file__[:-len(suffix)]

    skip_l = int(os.environ.get("test_skip_l", "0"))
    skip_r = int(os.environ.get("test_skip_r", "1000000"))
    skip = os.environ.get("test_skip", "").split(",")
    test_only = None
    if "test_only" in os.environ:
        test_only = set(os.environ.get("test_only").split(","))

    test_files = os.listdir(test_dir)
    test_files = sorted(test_files)
    suite = unittest.TestSuite()
    test_names = []
    seperate_test = os.environ.get("seperate_test", "1") == "1"
    
    for _, test_file in enumerate(test_files):
        test_name = test_file.split(".")[0]
        tests = unittest.defaultTestLoader.loadTestsFromName(
            "jittor.test."+test_name)
            
        if not test_file.startswith("test_"):
            continue
        if _ < skip_l or _ > skip_r:
            continue
        if test_only and test_name not in test_only:
            continue
        for s in skip:
            if s in test_name:
                continue

        print("Add Test", _, test_name)
        if seperate_test:
            test_names.append("jittor.test."+test_name)
        else:
            suite.addTest(tests)

    if seperate_test:
        import subprocess as sp
        import sys
        import time
        import jittor_utils
        start = time.time()
        errors = ""
        f = open(jittor_utils.home()+"/.cache/jittor/test.log", "w")
        for i,test_name in enumerate(test_names):
            progress = f"{i}/{len(test_names)}"
            print(f"[RUN TEST {progress}]", test_name)
            r = sp.run(" ".join([sys.executable, '-m', test_name, '-v']), stdout=sp.PIPE, stderr=sp.STDOUT, timeout=60*10, shell=True)
            out = r.stdout.decode('utf8')
            sys.stdout.write(out)
            f.write(out)
            msg = f"[RUN TEST {progress} OK]"
            if r.returncode:
                msg = f"[RUN TEST {progress} FAILED]"
            msg = msg + f" {test_name} {time.time()-start:.1f}\n"
            if r.returncode:
                errors += msg
            sys.stdout.write(msg)
            f.write(msg)
        sys.stdout.write(errors)
        f.write(errors)
        f.close()

    result = unittest.TextTestRunner(verbosity=3).run(suite)
    if len(result.errors) or len(result.failures):
        exit(1)
