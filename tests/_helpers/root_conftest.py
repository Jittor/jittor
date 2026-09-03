"""Read ``tests/conftest.py`` as a file, because its module name is ambiguous.

Three structure tests used to say ``import conftest`` and assert on the module
object. That works when the selection starts at ``tests/structure`` and fails
with ``AttributeError: module 'conftest' has no attribute 'source_python_dir'``
in the whole Torch-mode session -- the same test, a different answer, decided by
which other directory was named alongside it. Measured both ways: 275 passed in
``tests/structure`` alone, three failures in the full session.

The cause is not a rename. pytest imports conftest modules under their bare
basename, and this tree has two of them -- ``tests/conftest.py`` and
``tests/compat/torch/conftest.py``. When the run reaches the second, it takes
over the name, and (measured) the first is then **not in ``sys.modules`` at
all**: pytest keeps its own reference as a plugin, so the module still works, it
is simply unreachable by name. A lookup by ``__file__`` does not save it either.

So the tests stop asking for the object. What they actually mean to assert is a
property of the *file*: that the root conftest takes ``source_python_dir`` from
``_helpers.child_process`` rather than re-implementing it, so the parent's
``sys.path`` and the child's ``PYTHONPATH`` cannot drift apart. That is a
statement about the source text, it is true in every selection, and it is what
this module provides.

The behaviour itself is tested against ``_helpers.child_process``, which is
where the implementation lives.
"""

from pathlib import Path


ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def root_conftest_source():
    return ROOT_CONFTEST.read_text(encoding="utf-8")


def root_conftest_imports_from_the_helper(name):
    """Whether ``tests/conftest.py`` takes ``name`` from ``_helpers.child_process``."""
    return ("from _helpers.child_process import %s" % name) in root_conftest_source()
