#!/usr/bin/env python3
"""Reject incomplete or polluted Jittor source distributions."""

from __future__ import print_function

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


REQUIRED_SOURCE_PATHS = (
    "examples/README.md",
    "examples/gan/simple_cgan.py",
    "examples/notebooks/basics.md",
    "examples/notebooks/basics.ipynb",
    "requirements/examples.txt",
    "tools/README.md",
    "tools/build/build_aarch64_mkl.sh",
    "tools/release/pack_offline.py",
)

FORBIDDEN_DIRECTORY_NAMES = frozenset(
    (
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".git",
        ".nox",
        "dist",
    )
)


class SourceDistributionError(Exception):
    """Raised when a source archive cannot be audited."""


def _expected_source_paths(repo_root):
    command = (
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "examples",
        "tools",
        "requirements/examples.txt",
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise SourceDistributionError("cannot inventory source checkout: {}".format(error))
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise SourceDistributionError("cannot inventory source checkout: {}".format(detail))
    paths = frozenset(item for item in result.stdout.decode("utf-8").split("\0") if item)
    missing_sentinels = sorted(set(REQUIRED_SOURCE_PATHS) - paths)
    if missing_sentinels:
        raise SourceDistributionError(
            "source checkout is missing required paths: {}".format(", ".join(missing_sentinels))
        )
    return paths


def _member_issues(name, source):
    issues = []
    path = PurePosixPath(name)
    if not name:
        return ["{} contains an empty member name".format(source)]
    if name.startswith("/"):
        issues.append("{} contains an absolute member: {!r}".format(source, name))
    if "\\" in name:
        issues.append("{} contains a non-POSIX member: {!r}".format(source, name))
    if any(ord(character) < 32 for character in name):
        issues.append("{} contains a control character in member: {!r}".format(source, name))
    if ".." in path.parts:
        issues.append("{} contains a parent traversal member: {!r}".format(source, name))
    return issues


def _pollution_reason(relative):
    parts = PurePosixPath(relative).parts
    directories = parts if relative.endswith("/") else parts[:-1]
    for part in directories:
        if part in FORBIDDEN_DIRECTORY_NAMES:
            return "generated directory {!r}".format(part)
    if parts and parts[0] == "build":
        return "generated top-level build directory"
    if relative.endswith((".pyc", ".pyo")):
        return "generated Python bytecode"
    return None


def audit_sdist(path, expected_paths):
    path = Path(path)
    if not path.is_file():
        raise SourceDistributionError("source distribution does not exist: {}".format(path))
    try:
        with tarfile.open(str(path), "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise SourceDistributionError("cannot read source distribution {}: {}".format(path, error))

    issues = []
    names = [member.name for member in members]
    for name in names:
        issues.extend(_member_issues(name, path))
    for name, count in sorted(Counter(names).items()):
        if count > 1:
            issues.append("{} contains duplicate member: {}".format(path, name))

    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        issues.append(
            "{} must contain exactly one archive root, found {}".format(path, sorted(roots))
        )
        return issues, frozenset()

    root = next(iter(roots))
    relative_names = set()
    member_by_relative = {}
    for member in members:
        parts = PurePosixPath(member.name).parts
        if not parts or parts[0] != root:
            continue
        relative = PurePosixPath(*parts[1:]).as_posix()
        if not relative or relative == ".":
            continue
        relative_names.add(relative)
        member_by_relative[relative] = member
        reason = _pollution_reason(relative)
        if reason:
            issues.append("polluting sdist member ({}): {}".format(reason, relative))

    missing = sorted(set(expected_paths) - relative_names)
    for relative in missing:
        issues.append("missing required source-distribution member: {}".format(relative))

    governed_members = {
        relative
        for relative, member in member_by_relative.items()
        if not member.isdir()
        and (
            relative.startswith("examples/")
            or relative.startswith("tools/")
            or relative == "requirements/examples.txt"
        )
    }
    unexpected = sorted(governed_members - set(expected_paths))
    for relative in unexpected:
        issues.append("unexpected source-distribution member: {}".format(relative))

    for relative in expected_paths:
        member = member_by_relative.get(relative)
        if member is not None and not member.isfile():
            issues.append("required source-distribution member is not a file: {}".format(relative))

    return issues, frozenset(relative_names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args(argv)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        expected_paths = _expected_source_paths(repo_root)
        issues, members = audit_sdist(args.sdist, expected_paths)
    except SourceDistributionError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    if issues:
        for issue in issues:
            print("ERROR: {}".format(issue), file=sys.stderr)
        return 1
    print("source distribution OK: {} ({} members)".format(args.sdist, len(members)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
