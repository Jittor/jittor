from .md_to_ipynb import dirname, notebook_dir
import os
import sys

cmd = f"cp -r {dirname}/* {notebook_dir}/ && cd {notebook_dir} && jupyter notebook {' '.join(sys.argv[1:])}"
print("run cmd:", cmd)
os.system(cmd)