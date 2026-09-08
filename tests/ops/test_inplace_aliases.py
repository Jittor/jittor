"""Only explicitly declared, genuinely mutating ``Var`` methods end in ``_``."""

import ast
from pathlib import Path
import unittest

import numpy as np

import jittor as jt


_FORBIDDEN = {
    "all_", "any_", "argmax_", "argmin_", "chunk_", "cpu_", "cuda_",
    "gather_", "isfinite_", "isinf_", "isnan_", "mean_", "nonzero_",
    "norm_", "sort_", "sum_", "tolist_", "topk_", "unbind_", "var_",
}

_EXPECTED = {
    "abs_", "add_", "clamp_", "constant_", "deg2rad_", "erf_", "erfinv_",
    "eye_", "fill_", "gauss_", "hardsigmoid_", "hardswish_", "index_add_",
    "index_fill_", "invariant_uniform_", "kaiming_normal_", "kaiming_uniform_",
    "log2_", "masked_fill_", "mul_", "multiply_", "normal_", "one_", "pow_",
    "rad2deg_", "random_", "relu_invariant_gauss_", "requires_grad_",
    "rrelu_", "rsqrt_", "scatter_", "scatter_add_", "scatter_reduce_",
    "sigmoid_", "sqr_", "sqrt_", "squeeze_", "sub_", "t_", "transpose_",
    "tril_", "triu_", "trunc_normal_", "uniform_", "unsqueeze_",
    "xavier_gauss_", "xavier_uniform_", "zero_",
}


class TestExplicitInplaceAliases(unittest.TestCase):
    def test_non_mutating_and_non_var_results_have_no_inplace_alias(self):
        self.assertEqual(_FORBIDDEN & set(dir(jt.Var)), set())

    def test_suffix_surface_is_an_explicit_allowlist(self):
        actual = {
            name for name in dir(jt.Var)
            if name.endswith("_") and not name.startswith("__")
        }
        self.assertEqual(actual, _EXPECTED)
        self.assertEqual(set(jt._INPLACE_ALIASES), {
            "deg2rad_", "hardsigmoid_", "hardswish_", "log2_", "masked_fill_",
            "mul_", "pow_", "rad2deg_", "rrelu_", "rsqrt_",
            "scatter_reduce_", "sqr_", "sub_", "squeeze_", "t_",
            "transpose_", "unsqueeze_",
        })

    def test_root_does_not_scan_signatures_to_invent_aliases(self):
        root = Path(jt.__file__).read_text(encoding="utf-8")
        tree = ast.parse(root)
        self.assertNotIn("co_varnames", root)
        scans = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("vars", "dir")
        ]
        self.assertEqual(scans, [])

    def test_declared_aliases_mutate_and_return_self(self):
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype="float32"))
        result = x.sub_(1)
        self.assertIs(result, x)
        np.testing.assert_array_equal(x.numpy(), [0.0, 1.0, 2.0])

        matrix = jt.array(np.arange(6, dtype="float32").reshape(2, 3))
        result = matrix.transpose_(0, 1)
        self.assertIs(result, matrix)
        np.testing.assert_array_equal(
            matrix.numpy(), np.arange(6, dtype="float32").reshape(2, 3).T)

        mask = jt.array(np.array([True, False, True]))
        result = x.masked_fill_(mask, 9)
        self.assertIs(result, x)
        np.testing.assert_array_equal(x.numpy(), [9.0, 1.0, 9.0])

    def test_truth_reductions_remain_available_without_fake_var_aliases(self):
        value = jt.array(np.array([[True, True], [True, False]]))
        np.testing.assert_array_equal(jt.all(value, dim=1).numpy(), [True, False])
        np.testing.assert_array_equal(jt.any(value, dim=1).numpy(), [True, True])


if __name__ == "__main__":
    unittest.main()
