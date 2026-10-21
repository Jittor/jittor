"""Each operator keeps the source line that built it.

The launch origin is captured once per native entry from Python and reused for
every operator that entry builds, and a capture over frames already seen skips
the full walk. Both are only right if the answer never changes: an operator
built on one line must not take the line of another, and one built inside a
helper is attributed to the helper's line, not to its caller's.
"""
import textwrap

from _helpers.child_process import run_child_script
from _helpers.capability import require_accelerator


def test_operators_keep_the_line_that_built_them():
    require_accelerator("cuda")
    result = run_child_script(textwrap.dedent('''
        import inspect
        import jittor as jt

        def helper(value):
            return value * 7.0              # HELPER_LINE

        def line_of(marker):
            source = open(__file__).read().splitlines()
            return next(i + 1 for i, text in enumerate(source) if marker in text)

        with jt.runtime.scope(use_cuda=1, trace_py_var=0, auto_flush_ops=0):
            first = inspect.currentframe().f_lineno + 1
            a = jt.array([2.0, 1.0]) * 3.0      # three operators, one entry
            a.sync()
            caller = inspect.currentframe().f_lineno + 1
            b = helper(a)
            b.sync()
            for _ in range(3):                  # the same lines again, now cached
                last = inspect.currentframe().f_lineno + 1
                c = jt.array([4.0]) - 1.0
                c.sync()
            history = jt.core.async_launch_history("cuda", 0)
            assert f"{__file__}:{first}" in history, history
            assert f"{__file__}:{line_of('HELPER_LINE')}" in history, history
            assert f"{__file__}:{caller}" not in history, history
            assert f"{__file__}:{last}" in history, history
            print("ORIGIN-LINES-PASS", flush=True)
    '''), env={"JT_SYNC": "0", "trace_py_var": "0"}, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORIGIN-LINES-PASS" in result.stdout
