# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""No tracked file names a personal machine.

Manuals, skills, scripts and comments are written on someone's machine, and the
easy thing to type is that machine's own home directory, interpreter path, lab
mount, ssh line or host name. Each one is wrong for every other reader: the
command does not run, the default points at a directory that does not exist, and
a test gated on a host name silently behaves differently everywhere else.

So every tracked text file is scanned for the shapes those details take, and a
hit fails with ``file:line``. The portable spellings are
``$JITTOR_LAB_ROOT/<topic>/...`` and ``$JITTOR_LAB_ROOT/_state/<topic>/<run>/``
for checkouts and state, ``$JITTOR_HOME`` for caches, ``<jittor-python>`` and
``<real-torch-python>`` for interpreters, ``<user>@<host>`` for remote shells,
``CUDA_VISIBLE_DEVICES=<gpu>`` for devices, and an environment variable for
anything a test must locate.

What is deliberately not flagged: placeholder homes (``/home/x/``,
``/home/user/``, ``/home/<...>/``), loopback and documentation addresses
(``127.0.0.1``, ``0.0.0.0``, RFC 5737 ranges), and dotted version strings such
as ``10.3.3.141``, which only count as addresses next to a port, an ``@`` or a
URL scheme. JSON baselines and ``uv.lock`` are generated data and are skipped.
"""

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Generated data that records other people's environments verbatim.
SKIPPED_SUFFIXES = (".json",)
SKIPPED_NAMES = frozenset(["uv.lock"])

#: Home-directory names that are obviously placeholders, not people.
PLACEHOLDER_HOMES = frozenset(["x", "user", "username", "you", "name", "runner"])

#: Remote hosts that are public services, not machines.
PUBLIC_HOSTS = frozenset(["github.com", "gitlab.com", "gitee.com"])

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = r"(?:%s\.){3}%s" % (_OCTET, _OCTET)

#: Character classes are used where a literal would make this file match itself.
_RULES = (
    ("personal home directory",
     re.compile(r"(?<![\w.$-])/home/([A-Za-z0-9_][A-Za-z0-9_.-]*)/")),
    ("lab root under /root", re.compile(r"/root/jittor[-]lab")),
    ("private cluster mount", re.compile(r"/apdceph[f]s|/jizhic[f]s|(?<![\w.$-])/data[1]/")),
    ("ssh login to a named host",
     re.compile(r"\bssh\b[^\n`'\"]*?\b[A-Za-z0-9_.-]+@([A-Za-z0-9_.-]+)")),
    ("user@address", re.compile(r"\b[A-Za-z0-9_.-]+@(%s)\b" % _IPV4)),
    ("address with a port", re.compile(r"(?<![\w.])(%s):\d{2,5}\b" % _IPV4)),
    ("address in a URL", re.compile(r"\b(?:https?|ftp|tcp)://(%s)\b" % _IPV4)),
    ("host-name gating",
     re.compile(r"(?:os\.uname\(\)(?:\[1\]|\.nodename)|platform\.node\(\)|socket\.gethostname\(\))"
                r"\s*(?:==|!=|\bin\b)"
                r"|\bin\s+(?:os\.uname\(\)(?:\[1\]|\.nodename)|platform\.node\(\)|socket\.gethostname\(\))")),
)

#: Cheap substrings that must appear on a line before any rule can match it.
_TRIGGERS = ("/home/", "/root/", "/apdceph", "/jizhic", "/data1", "@", "://",
             "uname", "platform.node", "gethostname", ":")


def _address_is_public_placeholder(address):
    first = address.split(".")
    return (address.startswith("127.") or address == "0.0.0.0"
            or address.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
            or first[0] == "0")


def _allowed(rule, match):
    if rule == "personal home directory":
        return match.group(1) in PLACEHOLDER_HOMES
    if rule == "ssh login to a named host":
        return match.group(1) in PUBLIC_HOSTS
    if rule in ("user@address", "address with a port", "address in a URL"):
        return _address_is_public_placeholder(match.group(1))
    return False


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(REPO_ROOT),
                         stdout=subprocess.PIPE, check=True)
    for name in out.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if not name or name.endswith(SKIPPED_SUFFIXES):
            continue
        if name.rsplit("/", 1)[-1] in SKIPPED_NAMES:
            continue
        path = REPO_ROOT / name
        if not path.is_file():  # deleted in the working tree, not yet staged
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        yield name, data.decode("utf-8", "replace")


def find_private_details():
    """``[(path, line, rule, text)]`` for every hit in the tracked tree."""
    hits = []
    for name, text in _tracked_text_files():
        if not any(trigger in text for trigger in _TRIGGERS):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not any(trigger in line for trigger in _TRIGGERS):
                continue
            for rule, pattern in _RULES:
                for match in pattern.finditer(line):
                    if not _allowed(rule, match):
                        hits.append((name, number, rule, line.strip()[:160]))
    return hits


class TestNoPrivatePaths(unittest.TestCase):
    def test_the_scan_reads_the_tree(self):
        # An empty file list would make the real assertion pass while proving
        # nothing; this is the failure mode a scanner has to rule out first.
        names = [name for name, _ in _tracked_text_files()]
        self.assertGreater(len(names), 500, "git ls-files returned too little")
        self.assertIn("AGENTS.md", names)

    def test_the_rules_catch_what_they_are_for(self):
        samples = {
            "personal home directory": "/home/" + "alice/projects/jittor",
            "lab root under /root": "/root/" + "jittor-lab/_state/x",
            "private cluster mount": "/apdceph" + "fs_private/qy/jittor",
            "ssh login to a named host": "ssh -p 22 " + "bob@lab-box.internal",
            "user@address": "bob@" + "10.1.2.3",
            "address with a port": "connect 192.168.1.20" + ":29500",
            "address in a URL": "http://" + "172.16.0.5/v1",
            "host-name gating": "if 'gpu' in " + "os.uname()[1]:",
        }
        for rule, pattern in _RULES:
            match = pattern.search(samples[rule])
            self.assertTrue(match and not _allowed(rule, match),
                            "rule %r no longer catches %r" % (rule, samples[rule]))

    def test_the_rules_leave_placeholders_and_versions_alone(self):
        clean = (
            "/home/x/.cache/jittor", "/home/<user>/jittor", "$JITTOR_LAB_ROOT/_state/a",
            "ssh <user>@<host>", "git clone git@github.com:Jittor/jittor.git",
            "torch 10.3.3.141 and jittor 1.3.11.0", "http://127.0.0.1:18091/v1",
            "CUDA_VISIBLE_DEVICES=<gpu>", "socket.gethostname()",
        )
        for line in clean:
            for rule, pattern in _RULES:
                for match in pattern.finditer(line):
                    self.assertTrue(_allowed(rule, match),
                                    "rule %r flags the placeholder %r" % (rule, line))

    def test_no_tracked_file_names_a_personal_machine(self):
        hits = find_private_details()
        self.assertEqual(hits, [], "private machine details in tracked files "
                         "(use $JITTOR_LAB_ROOT/..., $JITTOR_HOME, <jittor-python>, "
                         "<user>@<host>, CUDA_VISIBLE_DEVICES=<gpu>, or an environment "
                         "variable):\n" + "\n".join(
                             "%s:%d: [%s] %s" % hit for hit in hits))


if __name__ == "__main__":
    unittest.main()
