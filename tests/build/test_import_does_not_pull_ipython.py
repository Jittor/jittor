"""``import jittor`` must not import IPython to find out whether it is in one.

`jittor_utils.in_ipynb` used to answer that question with ``from IPython import
get_ipython``. On a machine where IPython is installed and unused that pulled
the whole package into every ``import jittor``: measured 0.267 s of a 1.02 s
hot-cache CUDA import, 18% of it, for a value that is False.

A running IPython shell injects ``get_ipython`` into builtins, so the presence
of that name is the whole test -- and unlike the import, it also works when
IPython is not installed at all.

The assertion is on ``sys.modules`` rather than on a timing, because a timing
threshold on this machine is not trustworthy: with several partitions
compiling, the same measurement swings by two orders of magnitude (see the
handoff, "并发编译时量不出性能数字"). "IPython was never imported" is exact.

A meta-path trap is installed ahead of the import to keep that assertion from
being vacuous. The earlier control -- "IPython is importable on this machine,
so a stray import would be visible" -- asserted something about the *gate
environment* rather than about jittor, and failed it: IPython is declared in
``requirements/docs.txt`` only, so the test venv has none and the control was
red everywhere except a docs build. A trap is not vacuous by construction, and
``test_the_trap_can_fire`` is the control for the control.
"""

import sys

from _helpers.child_process import run_python_child


#: Run with the module to import and the name to blame as the two arguments.
#: Anything that reaches for IPython raises, naming the caller, so the failure
#: says what asked rather than only that an import failed.
_IPYTHON_TRAP = r"""
import importlib.abc
import sys


class _NoIPython(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "IPython" or name.startswith("IPython."):
            raise AssertionError(
                "%s asked for IPython; jittor_utils.in_ipynb reads builtins "
                "instead and must not import it" % sys.argv[2])
        return None


sys.meta_path.insert(0, _NoIPython())
__import__(sys.argv[1])
print("IMPORTED", sys.argv[1], "IPython" in sys.modules)
"""


def _trap_run(module, blames):
    return run_python_child(["-c", _IPYTHON_TRAP, module, blames],
                            merge_stderr=True)


def test_importing_jittor_does_not_import_ipython():
    result = _trap_run("jittor", "import jittor")
    assert result.returncode == 0, result.stdout
    assert "IMPORTED jittor False" in result.stdout, result.stdout


def test_the_trap_can_fire():
    """A trap that cannot fire would make the test above pass for free."""
    result = _trap_run("IPython", "this test")
    assert result.returncode != 0, result.stdout
    assert "this test asked for IPython" in result.stdout, result.stdout


def test_in_ipynb_is_false_outside_ipython_and_needs_no_import():
    import jittor_utils

    assert jittor_utils.in_ipynb() is False
    assert "IPython" not in sys.modules, \
        "in_ipynb() imported IPython on the negative path"
