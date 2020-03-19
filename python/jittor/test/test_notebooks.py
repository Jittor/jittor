# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest, os
import jittor as jt
from jittor import LOG
import json

dirname = os.path.join(jt.flags.jittor_path, "notebook")
notebook_dir = os.path.join(jt.flags.cache_path, "notebook")
tests = []
for mdname in os.listdir(dirname):
    if not mdname.endswith(".src.md"): continue
    tests.append(mdname[:-3])

try:
    jt.compiler.run_cmd("ipython --help")
    has_ipython = True
except:
    has_ipython = False

def test(name):
    LOG.i(f"Run test {name} from {dirname}")
    ipynb_name = os.path.join(notebook_dir, name+".ipynb")
    jt.compiler.run_cmd("ipython "+ipynb_name)

def init():
    jt.compiler.run_cmd("python3 "+os.path.join(dirname, "md_to_ipynb.py"))

src = """class TestNodebooks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        init()
"""
for name in tests:
    src += f"""
    @unittest.skipIf(not has_ipython, "No IPython found")
    def test_{name.replace(".src","")}(self):
        test("{name}")
    """

LOG.vvv("eval src\n"+src)
exec(src)

if __name__ == "__main__":
    unittest.main()
