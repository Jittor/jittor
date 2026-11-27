"""Allocator facade dispatch contracts, executable without importing Jittor."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


_SOURCE = Path(__file__).resolve().parents[2] / 'compat/torch/installers/cuda/api.py'


class TestEmptyCacheContract(unittest.TestCase):
    def _function(self, mode, fail=None):
        tree = ast.parse(_SOURCE.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_empty_cache')
        calls = []
        def gc():
            calls.append('gc')
            if fail == 'gc':
                raise RuntimeError('collection failed')
        def sync(value):
            self.assertIs(value, True)
            calls.append('sync')
            if fail == 'sync':
                raise RuntimeError('synchronization failed')
        scope = {'jt': SimpleNamespace(gc=gc, sync_all=sync),
                 '_cuda_runtime': lambda: SimpleNamespace(empty_cache_mode=mode)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SOURCE), 'exec'), scope)
        return scope['_empty_cache'], calls

    def test_modes_and_unknown_mode(self):
        for mode in ('0', 'false', 'no', 'off', 'none', 'noop'):
            function, calls = self._function(mode)
            function()
            self.assertEqual(calls, [])
        for mode in ('', '1', 'true', 'yes', 'on', 'gc'):
            function, calls = self._function(mode)
            function()
            self.assertEqual(calls, ['gc'])
        for mode in ('sync', 'full'):
            function, calls = self._function(mode)
            function()
            self.assertEqual(calls, ['sync', 'gc'])
        function, calls = self._function('typo')
        with self.assertRaises(ValueError):
            function()
        self.assertEqual(calls, [])

    def test_failures_propagate_and_failed_sync_stops_collection(self):
        for mode, fail, expected in [('gc', 'gc', ['gc']),
                                     ('sync', 'sync', ['sync']),
                                     ('sync', 'gc', ['sync', 'gc'])]:
            function, calls = self._function(mode, fail)
            with self.assertRaises(RuntimeError):
                function()
            self.assertEqual(calls, expected)

    def test_default_selects_collection(self):
        tree = ast.parse(_SOURCE.read_text())
        defaults = [n.args[1].value for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and len(n.args) == 2
                    and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == 'JITTOR_TORCH_CUDA_EMPTY_CACHE'
                    and isinstance(n.args[1], ast.Constant)]
        self.assertEqual(defaults, ['gc'])


if __name__ == '__main__':
    unittest.main()
