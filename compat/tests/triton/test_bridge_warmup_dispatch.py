"""CPU control-flow contracts; compiler and tensor leaves are explicit fixtures.

These tests execute the source-owned bridge closure without importing Jittor or
initializing a CUDA driver. They verify dispatch, not compilation or GPU output.
"""
import ast
from pathlib import Path
import types
import unittest


def source_bridge(upstream_run, backend_run):
    source = Path(__file__).resolve().parents[2] / 'triton' / '__init__.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == '_bridge_run']
    namespace = dict(orig_run=upstream_run,
                     backend=types.SimpleNamespace(run=backend_run),
                     _args_have_jittor_var=lambda args, kwargs:
                     any(isinstance(value, TensorLeaf)
                         for value in list(args) + list(kwargs.values())))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_bridge_run']


class TensorLeaf:
    """Classification fixture; not a Jittor Var or a numerical tensor."""


class TestBridgeWarmupDispatch(unittest.TestCase):
    def test_warmup_preserves_original_result_arguments_and_no_launch(self):
        calls, artifact, owner, tensor = [], object(), object(), TensorLeaf()

        def upstream(self, *args, grid, warmup, **kwargs):
            calls.append((self, args, grid, warmup, kwargs))
            return artifact

        def no_launch(*args, **kwargs):
            raise AssertionError('compile-only warmup entered backend launcher')

        def grid(_):
            raise AssertionError('warmup evaluated launch grid')

        bridge = source_bridge(upstream, no_launch)
        result = bridge(owner, tensor, 17, grid=grid, warmup=True,
                        BLOCK=32, num_warps=4)
        self.assertIs(result, artifact)
        self.assertEqual(calls, [(owner, (tensor, 17), grid, True,
                                  dict(BLOCK=32, num_warps=4))])

    def test_compile_error_is_not_swallowed(self):
        error = RuntimeError('compiler rejected signature')

        def upstream(*args, **kwargs):
            raise error

        bridge = source_bridge(upstream, lambda *args: self.fail('launch'))
        with self.assertRaises(RuntimeError) as caught:
            bridge(object(), TensorLeaf(), grid=(1,), warmup=True)
        self.assertIs(caught.exception, error)

    def test_actual_launch_keeps_backend_cpu_rejection(self):
        error = ValueError('CPU tensor argument query_start_loc_ptr')

        def backend(*args):
            raise error

        bridge = source_bridge(lambda *a, **k: self.fail('upstream launch'), backend)
        with self.assertRaises(ValueError) as caught:
            bridge(object(), TensorLeaf(), grid=(1,), warmup=False)
        self.assertIs(caught.exception, error)

    def test_actual_jittor_launch_preserves_backend_arguments(self):
        calls, result, owner, tensor = [], object(), object(), TensorLeaf()

        def backend(*args):
            calls.append(args)
            return result

        bridge = source_bridge(lambda *a, **k: self.fail('upstream launch'), backend)
        self.assertIs(bridge(owner, tensor, grid=(2,), warmup=False, N=32), result)
        self.assertEqual(calls, [(owner, (tensor,), dict(N=32), (2,))])

    def test_non_jittor_arguments_keep_upstream_launch_and_warmup(self):
        for warmup in (False, True):
            calls = []

            def upstream(*args, **kwargs):
                calls.append((args, kwargs))
                return 'upstream-artifact'

            bridge = source_bridge(upstream, lambda *a: self.fail('backend'))
            self.assertEqual(bridge('owner', 1, grid=(3,), warmup=warmup),
                             'upstream-artifact')
            self.assertEqual(calls, [(('owner', 1), dict(grid=(3,), warmup=warmup))])


if __name__ == '__main__':
    unittest.main()
