from .md_to_ipynb import dirname, notebook_dir
import os
import sys
import shutil
from distutils.dir_util import copy_tree

copy_tree(dirname, notebook_dir)
os.chdir(notebook_dir)
os.system(f"{sys.executable} -m jupyter notebook")