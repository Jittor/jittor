error_msg = """Jittor only supports Linux and macOS currently.
For other OS, use Jittor may be risky.
If you insist on installing, please set the environment variable : export FORCE_INSTALL=1
We strongly recommend docker installation:

# CPU only (Linux)
>>> docker run -it --network host jittor/jittor
# CPU and CUDA (Linux)
>>> docker run -it --network host jittor/jittor-cuda
# CPU only (Mac and Windows)
>>> docker run -it -p 8888:8888 jittor/jittor

Reference:
1. Windows/Mac/Linux install Jittor in Docker: https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-15-00-00-docker/
"""
from warnings import warn
import os
import platform

if not platform.system() in ['Linux', 'Darwin']:
    assert os.environ.get("FORCE_INSTALL", '0') != '1', error_msg

import setuptools
from setuptools import setup, find_packages
import os
import sys

path = os.path.dirname(__file__)
with open(os.path.join(path, "README.md"), "r", encoding='utf8') as fh:
    long_description = fh.read()

with open(os.path.join(path, "python/jittor/__init__.py"), "r", encoding='utf8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = line.split("'")[1]
            break
    else:
        raise RuntimeError("Unable to find version string.")

version_require = (3,7)
if os.name == 'nt':
    version_require = (3,8)
if sys.version_info < version_require:
    raise RuntimeError("Python version not match, require %s, current %s"
        %(version_require, sys.version_info))

setuptools.setup(
    name='jittor',  
    version=version,
    # scripts=[],
    author="Jittor Group",
    author_email="ran.donglang@gmail.com",
    description="a Just-in-time(JIT) deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://jittor.org",
    # packages=setuptools.find_packages(),
    python_requires='>=3.7',

    packages=["jittor", "jittor.test", "jittor.models", "jittor.utils", "jittor_utils"],
    package_dir={'': 'python'},
    package_data={'': ['*', '*/*', '*/*/*','*/*/*/*','*/*/*/*/*','*/*/*/*/*/*']},
    # include_package_data=True,
    install_requires=[
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
        'pywin32 >= 1.0 ; platform_system=="Windows"'
    ],
 )

# upload to pip:
# rm -rf dist && python3.7 ./setup.py sdist && python3.7 -m twine upload dist/*
