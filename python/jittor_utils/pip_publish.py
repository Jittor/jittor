import os
import glob
import shutil
import sys

home_path = os.path.join(os.path.dirname(__file__), "..", "..")
home_path = os.path.abspath(home_path)

def callback(func, path, exc_info):
    print(f"remove \"{path}\" failed.")

def rmtree(path):
    if os.path.isdir(path):
        print(f"remove \"{path}\" recursive.")
        shutil.rmtree(path, onerror=callback)

def remove_tmpfile():
    dist_file = home_path+"/dist"
    egg_file = glob.glob(home_path+"/**/*egg-info")
    rmtree(dist_file)
    for e in egg_file:
        rmtree(e)

def run_cmd(cmd):
    print("[CMD]", cmd)
    assert os.system(cmd)==0

os.chdir(home_path)
remove_tmpfile()

run_cmd(f"{sys.executable} ./setup.py sdist")
run_cmd(f"{sys.executable} -m twine upload dist/*")

remove_tmpfile()