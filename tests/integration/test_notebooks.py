# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest, os
import jittor as jt
from jittor import LOG
import sys
import jittor_utils as jit_utils
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.manual]

dirname = os.path.join(jt.flags.jittor_path, "notebook")
notebook_dir = os.path.join(jit_utils.home(), ".cache","jittor","notebook")
tests = []
for mdname in os.listdir(dirname):
    if not mdname.endswith(".src.md"): continue
    # temporary disable model_test
    if "GAN" in mdname: continue
    tests.append(mdname[:-3])

def _run_notebook(name):
    LOG.i(f"Run test {name} from {dirname}")
    ipynb_name = os.path.join(notebook_dir, name+".ipynb")
    jt.compiler.run_cmd("ipython "+ipynb_name)

def _init_notebooks():
    cmd = sys.executable+" "+os.path.join(dirname, "md_to_ipynb.py")
    LOG.i("init notebooks:", cmd)
    jt.compiler.run_cmd(cmd)

class TestNodebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            jt.compiler.run_cmd("ipython --help")
        except Exception as error:
            raise unittest.SkipTest("IPython is unavailable: {}".format(error))
        _init_notebooks()


def _make_notebook_test(name):
    def test(self):
        _run_notebook(name)

    return test


for name in tests:
    setattr(TestNodebooks, "test_" + name.replace(".src", ""), _make_notebook_test(name))

if __name__ == "__main__":
    unittest.main()
