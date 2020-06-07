error_msg = "Jittor only supports Ubuntu>=16.04 currently."

try:
    with open("/etc/os-release", "r", encoding='utf8') as f:
        s = f.read().splitlines()
        m = {}
        for line in s:
            a = line.split('=')
            m[a[0]] = a[1].replace("\"", "")
except:
    raise RuntimeError(error_msg)
assert m["NAME"] == "Ubuntu" and float(m["VERSION_ID"])>16, error_msg

import setuptools
from setuptools import setup, find_packages
import os

path = os.path.dirname(__file__)
with open(os.path.join(path, "README.md"), "r", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='jittor',  
    version='1.1.4.4',
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
    package_dir={'': os.path.join(path, 'python')},
    package_data={'': ['*', '*/*', '*/*/*','*/*/*/*','*/*/*/*/*','*/*/*/*/*/*']},
    # include_package_data=True,
    install_requires=[
        "pybind11",
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
    ],
 )

# python3.7 setup.py sdist
# python3.7 -m twine upload dist/*
