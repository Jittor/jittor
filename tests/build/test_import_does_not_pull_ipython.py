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
"""

import sys

from _helpers.child_process import run_python_child


def test_importing_jittor_does_not_import_ipython():
    result = run_python_child(
        ["-c", "import sys, jittor;"
               "print('IPYTHON_LOADED', 'IPython' in sys.modules)"],
        merge_stderr=True)
    assert "IPYTHON_LOADED False" in result.stdout, result.stdout


def test_ipython_is_installed_here_so_the_check_is_not_vacuous():
    """Without this the test above passes on a machine that has no IPython.

    That is the shape that keeps producing false confidence in this tree: a
    gate that holds because the thing it guards against cannot happen in the
    current environment, not because the code stopped doing it.
    """
    result = run_python_child(
        ["-c", "import IPython; print('IPYTHON_IMPORTABLE')"],
        merge_stderr=True)
    assert "IPYTHON_IMPORTABLE" in result.stdout, result.stdout


def test_in_ipynb_is_false_outside_ipython_and_needs_no_import():
    import jittor_utils

    assert jittor_utils.in_ipynb() is False
    assert "IPython" not in sys.modules, \
        "in_ipynb() imported IPython on the negative path"
