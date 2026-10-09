"""The deferred-hardware manifest stays consistent with itself and the tree.

Some work can be written now and only accepted on hardware this project does
not always have. ``agent/manuals/deferred-hardware.md`` is the one list of that
work: a table near the top names each item, and the sections below say what to
run when the hardware arrives and which items that run discharges. A manifest
nobody checks goes stale exactly when it is needed -- on hardware day, by
someone who was not here when it was written.
"""

from __future__ import print_function

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "agent" / "manuals" / "deferred-hardware.md"
NOXFILE = REPO_ROOT / "noxfile.py"

#: The item table: the section that starts with this heading, up to the next one.
_TABLE_HEADING = "## 待硬件验收的项目"

#: An item id as the manifest writes it: a backticked ``<major>.<minor>``
#: label such as ``8.06`` or ``6.B02``.
_ITEM_ID = re.compile(r"`([0-9]{1,2}\.[0-9A-Z][0-9A-Z]{0,2})`")
_TABLE_ROW = re.compile(r"^\| `([0-9]{1,2}\.[0-9A-Z][0-9A-Z]{0,2})` \|", re.M)


def _split_manifest():
    text = MANIFEST.read_text(encoding="utf-8")
    start = text.index(_TABLE_HEADING)
    end = text.index("\n## ", start + len(_TABLE_HEADING))
    return text[start:end], text[:start] + text[end:]


def test_the_item_table_and_the_sections_agree():
    """Every listed item is explained somewhere, and nothing is explained unlisted.

    Both directions: an id cited by a section but missing from the table is
    work nobody can find from the top of the page; a table row no section
    cites is a promise with no command behind it.
    """
    table, body = _split_manifest()
    listed = _TABLE_ROW.findall(table)
    assert listed, "the deferred-hardware item table is empty or unparsable"
    duplicates = sorted({item for item in listed if listed.count(item) > 1})
    assert duplicates == [], "item ids listed twice: %s" % duplicates

    cited = set(_ITEM_ID.findall(body))
    unlisted = sorted(cited - set(listed))
    assert unlisted == [], (
        "these items are cited in the manifest but missing from its table: %s"
        % unlisted)
    uncited = sorted(set(listed) - cited)
    assert uncited == [], (
        "these table items are not explained by any section: %s" % uncited)


#: Every test root, not just ``tests/``. The pattern used to read ``tests/``
#: alone, so a promise that moved to ``compat/tests`` left the check behind
#: with it -- the manifest kept naming the old path and nothing said so.
_TEST_PATH = re.compile(r"`((?:tests|compat/tests|adapters/tests)/[\w/]+\.py)(?:::[\w:]+)?`")


def test_the_manifest_names_test_paths_that_exist():
    manifest = MANIFEST.read_text(encoding="utf-8")
    referenced = set(_TEST_PATH.findall(manifest))
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
