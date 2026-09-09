"""The deferred-hardware manifest stays in sync with the board and the tree.

Tasks whose acceptance waits for hardware are folded into two board buckets.
The buckets record *that* they wait; ``agent/manuals/deferred-hardware.md``
records what to run when the hardware arrives, and which board rows that run
discharges. A manifest nobody checks goes stale exactly when it is needed --
on hardware day, by someone who was not here when it was written.
"""

from __future__ import print_function

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD = REPO_ROOT / "refactor-wip" / "architecture" / "refactor-board.md"
MANIFEST = REPO_ROOT / "agent" / "manuals" / "deferred-hardware.md"
NOXFILE = REPO_ROOT / "noxfile.py"

#: Board statuses that mean "the code may be written now, hardware decides".
_HARDWARE_BUCKETS = ("并入 硬件验收", "并入 多机硬件验收")

_TASK_ROW = re.compile(r"^\| ([0-9]+\.[0-9A-Z]+) \| [^|]*\| ([^|]*)\|", re.M)


def _board_rows():
    text = BOARD.read_text(encoding="utf-8")
    body = text[text.index("\n## 任务"):]
    return [(task, status.strip()) for task, status in _TASK_ROW.findall(body)]


def test_every_hardware_bucket_task_is_in_the_manifest():
    manifest = MANIFEST.read_text(encoding="utf-8")
    missing = [task for task, status in _board_rows()
               if status in _HARDWARE_BUCKETS and "`%s`" % task not in manifest]
    assert missing == [], (
        "these tasks wait for hardware but the manifest does not say what to "
        "run for them: %s" % missing)


def test_the_manifest_names_test_paths_that_exist():
    manifest = MANIFEST.read_text(encoding="utf-8")
    referenced = set(re.findall(r"`(tests/[\w/]+\.py)(?:::[\w:]+)?`", manifest))
    missing = sorted(path for path in referenced
                     if not (REPO_ROOT / path).is_file())
    assert missing == [], "manifest names paths that do not exist: %s" % missing


def test_the_manifest_names_nox_sessions_that_exist():
    """A promised command must be runnable; an absent one must stay absent.

    Both directions matter. The first keeps the manifest honest about what
    exists; the second is why the three "no session yet" entries cannot be
    quietly left behind -- when someone adds the session, this fails until the
    manifest stops saying it is missing.
    """
    manifest = MANIFEST.read_text(encoding="utf-8")
    noxfile = NOXFILE.read_text(encoding="utf-8")

    promised = set(re.findall(r"`nox -s (\w+)`", manifest))
    absent = sorted(name for name in promised
                    if "def %s(session)" % name not in noxfile)
    assert absent == [], "manifest promises sessions noxfile lacks: %s" % absent

    for name in ("hccl", "corex", "multinode"):
        declared_missing = "尚无 nox session" in manifest
        exists = "def %s(session)" % name in noxfile
        assert not (exists and declared_missing and "`nox -s %s`" % name
                    not in manifest), (
            "noxfile now has a %s session; drop the \"no session yet\" note "
            "and list the command" % name)
