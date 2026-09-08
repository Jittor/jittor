"""Two test files with the same basename abort the whole collection.

``pyproject.toml`` sets ``testpaths = ["tests"]``, so a bare ``pytest`` pulls
the entire tree into one session. None of the test directories is a package,
so pytest derives a module name from the basename alone -- and a second file
with that basename raises "import file mismatch", which is a *collection*
error. Collection errors interrupt the run: pytest reports the error instead
of a summary and executes nothing at all.

That is how it stayed invisible. The gate output has no failure count to
notice, and a partition running one directory per pytest invocation never
sees the collision. It was found by reading a native-gate log that had no
summary line, three duplicated basenames deep.
"""

import collections
import pathlib

TESTS = pathlib.Path(__file__).resolve().parents[1]


def _test_files():
    from _helpers.paths import iter_test_files
    return iter_test_files()


def test_no_two_test_files_share_a_basename():
    by_name = collections.defaultdict(list)
    for path in _test_files():
        by_name[path.name].append(path.relative_to(TESTS.parent))
    clashes = {name: sorted(str(p) for p in paths)
               for name, paths in by_name.items() if len(paths) > 1}
    assert not clashes, (
        "these basenames appear more than once, which aborts collection for "
        "the whole tree; give the narrower-scoped file a qualified name: "
        "{}".format(clashes))


def test_the_tree_has_enough_test_files_to_make_the_check_meaningful():
    # Guards the check above against a glob that silently stops matching:
    # an empty file list would make it pass while proving nothing.
    assert len(_test_files()) > 200, len(_test_files())
