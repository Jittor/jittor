"""Legacy Doxygen builder retained until the Stage 8 documentation migration."""

from __future__ import print_function

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


DOXYGEN_URL = "https://cloud.tsinghua.edu.cn/f/dfa8f16ab00c4fa6b158/?dl=1"
CONFIG_URL = "https://cloud.tsinghua.edu.cn/f/caf3c3aa518248d5ad73/?dl=1"


def _fix_config(input_path, output_path, source_path, result_path):
    lines = input_path.read_text(encoding="utf-8").splitlines(True)
    updated = []
    for line in lines:
        if line.startswith("INPUT                  ="):
            line = "INPUT                  ={}\n".format(source_path)
        elif line.startswith("OUTPUT_DIRECTORY       ="):
            line = "OUTPUT_DIRECTORY       ={}\n".format(result_path)
        updated.append(line)
    output_path.write_text("".join(updated), encoding="utf-8")


def _extract_archive(archive_path, destination):
    destination = destination.resolve()
    with tarfile.open(str(archive_path), "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError("unsafe Doxygen archive member: {}".format(member.name))
        archive.extractall(str(destination))


def _default_state_root(repo_root):
    lab_root = Path(os.environ.get("JITTOR_LAB_ROOT", str(repo_root.parent / "jittor-lab")))
    return lab_root.expanduser().resolve() / "_state" / "tools" / "doxygen"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    parser.add_argument("--source-root", type=Path, default=repo_root)
    parser.add_argument("--state-root", type=Path, default=_default_state_root(repo_root))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source_root = args.source_root.expanduser().resolve()
    state_root = args.state_root.expanduser().resolve()
    required = (
        source_root / "python" / "jittor" / "src",
        source_root / "README.src.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("source tree is incomplete: {}".format(", ".join(missing)))

    print("source root: {}".format(source_root))
    print("state root: {}".format(state_root))
    if args.dry_run:
        print("dry-run: no files downloaded, copied, or generated")
        return 0

    downloads = state_root / "downloads"
    workspace = state_root / "workspace"
    source_copy = workspace / "source"
    output = state_root / "output"
    downloads.mkdir(parents=True, exist_ok=True)
    if workspace.exists():
        shutil.rmtree(str(workspace))
    source_copy.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)

    shutil.copytree(str(source_root / "python"), str(source_copy / "python"))
    notebooks = source_root / "examples" / "notebooks"
    if notebooks.is_dir():
        shutil.copytree(str(notebooks), str(source_copy / "examples" / "notebooks"))
    shutil.copy2(str(source_root / "README.src.md"), str(source_copy / "README.src.md"))

    archive_path = downloads / "doxygen-1.8.17.tar.gz"
    config_path = downloads / "Doxyfile"
    if not archive_path.is_file():
        urllib.request.urlretrieve(DOXYGEN_URL, str(archive_path))
    if not config_path.is_file():
        urllib.request.urlretrieve(CONFIG_URL, str(config_path))

    doxygen_root = workspace / "doxygen-1.8.17"
    if not doxygen_root.is_dir():
        _extract_archive(archive_path, workspace)
    executable = doxygen_root / "bin" / "doxygen"
    if not executable.is_file():
        raise RuntimeError("Doxygen executable is missing: {}".format(executable))

    generated_config = doxygen_root / "bin" / "Doxyfile"
    _fix_config(config_path, generated_config, source_copy, output)
    subprocess.run(
        [str(executable), str(generated_config)],
        cwd=str(executable.parent),
        check=True,
    )
    print("legacy Doxygen output: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
