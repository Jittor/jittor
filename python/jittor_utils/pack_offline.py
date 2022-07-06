urls = [
    ("https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp.tgz", "dnnl_lnx_2.2.0_cpu_gomp.tgz"),
    ("https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz", "dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz"),
    ("https://codeload.github.com/NVIDIA/cub/tar.gz/1.11.0", "cub-1.11.0.tgz"),
    ("https://codeload.github.com/Jittor/cutt/zip/v1.2", "cutt-1.2.zip"),
    ("https://codeload.github.com/NVIDIA/nccl/tar.gz/v2.8.4-1", "nccl.tgz"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
]

import urllib
from pathlib import Path
import os
import glob
import shutil
import sys

cpath = os.path.join(str(Path.home()), ".cache", "jittor", "offpack")
os.makedirs(cpath+"/python/jittor_offline", exist_ok=True)


for url, file_path in urls:
    file_path = os.path.join(cpath, "python/jittor_offline", file_path)
    print("download", url, file_path)
    urllib.request.urlretrieve(
        url, file_path
    )

with open(os.path.join(cpath, "MANIFEST.in"), "w") as f:
    f.write("include python/jittor_offline/*")
with open(os.path.join(cpath, "__init__.py"), "w") as f:
    f.write("")
with open(os.path.join(cpath, "setup.py"), "w") as f:
    f.write("""
import setuptools


setuptools.setup(
    name="jittor_offline",
    version="0.0.7",
    author="jittor",
    author_email="jittor@qq.com",
    description="jittor project",
    long_description="jittor_offline",
    long_description_content_type="text/markdown",
    url="https://github.com/jittor/jittor",
    project_urls={
        "Bug Tracker": "https://github.com/jittor/jittor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["jittor_offline"],
    package_dir={"": "python"},
    package_data={'': ['*', '*/*', '*/*/*','*/*/*/*','*/*/*/*/*','*/*/*/*/*/*']},
    python_requires=">=3.7",
    install_requires=[
        "jittor>=1.3.4.16",
    ],
)
""")


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

home_path = cpath
os.chdir(cpath)
remove_tmpfile()

run_cmd(f"{sys.executable} ./setup.py sdist")
run_cmd(f"{sys.executable} -m twine upload dist/*")

remove_tmpfile()