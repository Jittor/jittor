# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms in LICENSE.txt.
"""Linux cgroup memory limits used by the compilation scheduler."""

import os
import re


def _read(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except (OSError, UnicodeError):
        return ""


def _unescape_mount(path):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), path)


def limit_memory(memory, cgroup_file="/proc/self/cgroup", mountinfo_file="/proc/self/mountinfo"):
    """Clamp host memory to readable cgroup hard limits; leave it unchanged otherwise.

    Resolve both v1 and v2 memberships against their mounts, including mounts
    rooted at the container's own cgroup. Only visible ancestors can be read.
    Keep this algorithm in sync with utils/cgroup_memory.h.
    """
    groups = {}
    for line in _read(cgroup_file).splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[2].startswith("/"):
            continue
        if "memory" in fields[1].split(","):
            groups["cgroup"] = fields[2]
        elif fields[:2] == ["0", ""]:
            groups["cgroup2"] = fields[2]

    for line in _read(mountinfo_file).splitlines():
        before, separator, after = line.partition(" - ")
        fields, fs = before.split(), after.split()
        if not separator or len(fields) < 6 or len(fs) < 3:
            continue
        kind = fs[0]
        if kind not in groups:
            continue
        if kind == "cgroup" and "memory" not in fs[2].split(","):
            continue
        root, mount = map(_unescape_mount, fields[3:5])
        root, mount = root.rstrip("/") or "/", mount.rstrip("/") or "/"
        group = groups[kind].rstrip("/") or "/"
        # A namespace-root membership does not match unrelated subtree mounts.
        if group == root:
            relative = ""
        elif root == "/":
            relative = group[1:]
        elif group.startswith(root + "/"):
            relative = group[len(root) + 1:]
        else:
            continue
        # Do not escape the visible mount when /proc reports a path outside
        # the cgroup namespace (for example, /../some-other-group).
        if not mount.startswith("/") or any(p in (".", "..") for p in relative.split("/")):
            continue
        directory = os.path.join(mount, relative).rstrip("/") or "/"
        leaf = True
        while True:
            if kind == "cgroup2" or leaf or _read(os.path.join(directory, "memory.use_hierarchy")) == "1":
                name = "memory.max" if kind == "cgroup2" else "memory.limit_in_bytes"
                value = _read(os.path.join(directory, name))
                if value and all("0" <= c <= "9" for c in value):
                    try:
                        memory = min(memory, int(value))
                    except ValueError:
                        pass
            if directory == mount:
                break
            directory = os.path.dirname(directory)
            leaf = False
    return memory
