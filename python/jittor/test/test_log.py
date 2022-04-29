# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os
import re
import jittor as jt
from jittor import LOG

def find_log_with_re(logs, pattern=None, **args):
    if pattern:
        pattern = re.compile(pattern)
    flogs = []
    for log in logs:
        for arg in args:
            if log[arg] != args[arg]:
                break
        else:
            if pattern:
                res = re.findall(pattern, log["msg"])
                if len(res):
                    flogs.append(res[0])
            else:
                flogs.append(log["msg"])
    return flogs

class TestLog(unittest.TestCase):
    def test_log_capture(self):
        with jt.log_capture_scope(log_v=1000, log_vprefix="") as logs:
            LOG.v("1")
            LOG.vv("2")
            LOG.i("3")
            LOG.w("4")
            LOG.e("5")
            a = jt.zeros([10])
            a.sync()
        # TODO: why need manually delete this variable?
        del a
        logs2 = LOG.log_capture_read()
        assert len(logs2)==0

        for i in range(5):
            assert logs[i]['msg'] == str(i+1)
            assert logs[i]['level'] == 'iiiwe'[i]
            assert logs[i]['name'] == 'test_log.py'
        finished_log = [ l["msg"] for l in logs 
            if l["name"]=="executor.cc" and "return vars:" in l["msg"]]
        assert len(finished_log)==1 and "[10,]" in finished_log[0]


if __name__ == "__main__":
    unittest.main()