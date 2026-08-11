"""Legacy CentOS 7 release-build helper with repository-external state."""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import shutil
import subprocess


DOCKERFILE = r"""
FROM centos:7

WORKDIR /root
RUN yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make -y
RUN wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz
RUN tar xzf Python-3.8.3.tgz
RUN cd Python-3.8.3 && ./configure --enable-optimizations && make altinstall -j8
RUN yum install centos-release-scl -y
RUN yum install devtoolset-7-gcc-c++ which -y
RUN scl enable devtoolset-7 'g++ --version'
RUN python3.8 -m pip install numpy tqdm pillow astunparse
""".lstrip()


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _default_state_root(repo_root):
    lab_root = Path(os.environ.get("JITTOR_LAB_ROOT", str(repo_root.parent / "jittor-lab")))
    return lab_root.expanduser().resolve() / "_state" / "release" / "polish-centos"


def _run(command, cwd=None, dry_run=False):
    print("RUN:", " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, cwd=None if cwd is None else str(cwd), check=True)


def run_in_centos(env, source_root=None, state_root=None, dry_run=False):
    repo_root = Path(source_root or _repo_root()).expanduser().resolve()
    state = Path(state_root or _default_state_root(repo_root)).expanduser().resolve()
    dockerfile = state / "CentOS7.Dockerfile"
    source = state / "src"

    if dry_run:
        print("CentOS source:", repo_root)
        print("CentOS state:", state)
    else:
        state.mkdir(parents=True, exist_ok=True)
        dockerfile.write_text(DOCKERFILE, encoding="utf-8")
        if source.exists():
            shutil.rmtree(str(source))
        source.mkdir(parents=True)
        shutil.copytree(str(repo_root / "python" / "jittor"), str(source / "jittor"))
        shutil.copytree(str(repo_root / "python" / "jittor_utils"), str(source / "jittor_utils"))

    _run(
        ["docker", "build", "--tag", "jittor-centos7-build:legacy", "-f", str(dockerfile), "."],
        cwd=repo_root,
        dry_run=dry_run,
    )
    smoke = (
        "scl enable devtoolset-7 "
        "'PYTHONPATH=/root/.cache/jittor/src {} python3.8 -m jittor.selftest'"
    ).format(env)
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        "{}:/root/.cache/jittor".format(state),
        "jittor-centos7-build:legacy",
        "bash",
        "-lc",
        smoke,
    ]
    _run(command, dry_run=dry_run)
    # Preserve the historical two-run stability check.
    _run(command, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", nargs="?", default="cc_path=g++ nvcc_path=''")
    parser.add_argument("--source-root", type=Path, default=_repo_root())
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_in_centos(
        args.env,
        source_root=args.source_root,
        state_root=args.state_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
