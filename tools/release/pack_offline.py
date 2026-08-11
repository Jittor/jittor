"""Build the optional Jittor offline-data source distribution.

The command is intentionally inert when imported. Downloads and build output
are written only below ``--output-dir`` (or JITTOR_LAB_ROOT when omitted).
"""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


URLS = (
    (
        "https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp.tgz",
        "dnnl_lnx_2.2.0_cpu_gomp.tgz",
    ),
    (
        "https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz",
        "dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz",
    ),
    ("https://codeload.github.com/NVIDIA/cub/tar.gz/1.11.0", "cub-1.11.0.tgz"),
    ("https://codeload.github.com/Jittor/cutt/zip/v1.2", "cutt-1.2.zip"),
    ("https://codeload.github.com/NVIDIA/nccl/tar.gz/v2.8.4-1", "nccl.tgz"),
    (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
)

SETUP_SOURCE = """\
from setuptools import setup


setup(
    name="jittor_offline",
    version="0.0.7",
    author="jittor",
    author_email="jittor@qq.com",
    description="Offline runtime assets for Jittor",
    long_description="Optional offline runtime assets for Jittor.",
    long_description_content_type="text/plain",
    url="https://github.com/Jittor/jittor",
    project_urls={"Bug Tracker": "https://github.com/Jittor/jittor/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["jittor_offline"],
    package_dir={"": "python"},
    package_data={"": ["*", "*/*", "*/*/*", "*/*/*/*"]},
    python_requires=">=3.7",
    install_requires=["jittor>=1.3.4.16"],
)
"""


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _default_output_dir():
    repo_root = _repo_root()
    lab_root = Path(os.environ.get("JITTOR_LAB_ROOT", str(repo_root.parent / "jittor-lab")))
    return lab_root.expanduser().resolve() / "_state" / "release" / "offline-package"


def _write_package_files(work_root):
    package_root = work_root / "python" / "jittor_offline"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (work_root / "MANIFEST.in").write_text(
        "recursive-include python/jittor_offline *\n", encoding="utf-8"
    )
    (work_root / "setup.py").write_text(SETUP_SOURCE, encoding="utf-8")
    return package_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    print("offline package output:", output_dir)
    for url, filename in URLS:
        print("download {} -> {}".format(url, filename))
    if args.dry_run:
        print("dry-run: no directories, downloads, or build artifacts created")
        return 0

    work_root = output_dir / "work"
    dist_root = output_dir / "dist"
    if work_root.exists():
        shutil.rmtree(str(work_root))
    if dist_root.exists():
        shutil.rmtree(str(dist_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    package_root = _write_package_files(work_root)

    for url, filename in URLS:
        destination = package_root / filename
        urllib.request.urlretrieve(url, str(destination))

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist_root)],
        cwd=str(work_root),
        check=True,
    )
    archives = sorted(dist_root.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError("expected one offline sdist, found {}".format(archives))
    print("offline package built at", archives[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
