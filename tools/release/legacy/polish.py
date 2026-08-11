"""Legacy source-polish utility retained outside the runtime package."""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _default_output_root(repo_root):
    lab_root = Path(os.environ.get("JITTOR_LAB_ROOT", str(repo_root.parent / "jittor-lab")))
    return lab_root.expanduser().resolve() / "_state" / "release" / "polish"


def _run_python_import(environment):
    probe = "import jittor as jt; print('JITTOR_POLISH_CACHE=' + jt.flags.cache_path)"
    print("RUN:", sys.executable, "-c", probe)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    print(result.stdout, end="")
    marker = "JITTOR_POLISH_CACHE="
    matches = [
        line[len(marker) :] for line in result.stdout.splitlines() if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError("could not determine Jittor cache path")
    return Path(matches[0])


def _find_object_root(state_root, data_files):
    expected = {"{}.o".format(Path(name).name) for name in data_files}
    candidates = []
    for path in state_root.rglob("obj_files"):
        if path.is_dir() and expected.issubset({item.name for item in path.iterdir()}):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError("expected one CentOS object directory, found {}".format(candidates))
    return candidates[0]


def _archive_source(repo_root, output_path):
    command = [
        "tar",
        "--exclude=build",
        "--exclude=.git",
        "--exclude=.ipynb_checkpoints",
        "--exclude=__pycache__",
        "--exclude=__data__",
        "--exclude=my",
        "--exclude=dist",
        "--exclude=.vscode",
        "--exclude=.github",
        "-czf",
        str(output_path),
        "-C",
        str(repo_root),
        ".",
    ]
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("all", "native"), default="all")
    parser.add_argument("--source-root", type=Path, default=_repo_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.source_root.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_root(repo_root)
    )
    print("source root:", repo_root)
    print("output root:", output_root)
    if args.dry_run:
        print("dry-run: no compilation or archive creation performed")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JITTOR_HOME", str(output_root / "jittor-home"))
    os.environ.setdefault("cache_name", "release-polish-bootstrap")

    import jittor as jt

    data_path = Path(jt.flags.jittor_path) / "src" / "__data__"
    if not data_path.is_dir():
        raise RuntimeError("Jittor embedded source data is unavailable: {}".format(data_path))

    git_version = jt.compiler.run_cmd("git rev-parse HEAD", str(data_path))
    jt.LOG.i("git_version", git_version)
    data_files = [name for name in jt.compiler.files if "__data__" in name]
    jt.LOG.i("data_files", data_files)

    os_system_names = {"ubuntu": "Linux", "centos": "Linux", "macos": "Darwin"}
    if args.mode == "native":
        os_system_names = {"ubuntu": "Linux"}

    for os_name, os_type in os_system_names.items():
        if platform.system() != os_type:
            continue
        os_arch = platform.machine() if os_type == "Darwin" else ""
        compiler_name = "g++"
        device = "cpu"
        key = "{}-{}-{}".format(git_version, compiler_name, device)
        environment = os.environ.copy()
        environment.update(
            {
                "cache_name": "build/{}/{}".format(compiler_name, device),
                "cc_path": compiler_name,
                "nvcc_path": "",
            }
        )
        if platform.machine() in ("x86_64", "AMD64"):
            environment["cc_flags"] = "-march=core2"
        if key != "ubuntu":
            key += "-" + os_name
        if os_arch:
            key += "-" + os_arch
        if platform.machine() == "sw_64":
            key += "-sw_64"

        if os_name == "centos":
            from polish_centos import run_in_centos

            shell_env = "cache_name=build/g++/cpu cc_path=g++ nvcc_path=''"
            run_in_centos(shell_env, source_root=repo_root, state_root=output_root / "centos")
            object_root = _find_object_root(output_root / "centos", data_files)
        else:
            _run_python_import(environment)
            object_root = _run_python_import(environment) / "obj_files"

        object_files = []
        for name in data_files:
            object_path = object_root / "{}.o".format(Path(name).name)
            if not object_path.is_file():
                raise RuntimeError("compiled object is missing: {}".format(object_path))
            object_files.append(str(object_path))
        subprocess.run(
            ["ld", "-r", *object_files, "-o", str(output_root / "{}.o".format(key))],
            check=True,
        )

    if args.mode == "native":
        return 0

    archive_path = output_root / "jittor.tgz"
    _archive_source(repo_root, archive_path)
    print("local polish artifacts are ready at", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
