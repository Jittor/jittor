"""Public identity and compact numerical coverage across linalg domain owners."""

from _helpers import capability as _test_capability
import importlib
import pickle

import jittor as jt
import numpy as np
import pytest


def test_public_functions_keep_owner_identity_and_legacy_pickle_globals():
    owners = {
        "decompositions": ("svd", "svdvals", "eig", "eigh", "eigvalsh", "cholesky", "qr"),
        "solving": ("inv", "inv_ex", "pinv", "matrix_power", "det", "slogdet", "solve"),
        "norms": ("matrix_rank", "matrix_norm", "vector_norm", "norm", "cond"),
        "contractions": ("einsum",),
        "complex": ("complex_inv", "complex_eig", "complex_eigh", "complex_qr", "complex_svd", "complex_pinv"),
    }
    for module_name, names in owners.items():
        module = importlib.import_module("jittor.linalg." + module_name)
        for name in names:
            function = getattr(jt.linalg, name)
            assert function is getattr(module, name)
            assert function.__module__ == module.__name__
            assert pickle.loads(pickle.dumps(function)) is function
            legacy = ("cjittor.linalg\n" + name + "\n.").encode("ascii")
            assert pickle.loads(legacy) is function
    assert set(jt.linalg.__all__) <= set(dir(jt.linalg))
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(jt.linalg, "missing_linalg_function")


def test_result_types_retain_names_fields_and_pickle_identity():
    for name, typename, fields in (
        ("SVD", "svd", ("U", "S", "Vh")),
        ("INVEX", "inv_ex", ("inverse", "info")),
    ):
        cls = getattr(jt.linalg, name)
        assert cls.__name__ == typename
        assert cls._fields == fields
        assert cls.__module__ == "jittor.linalg.results"
        value = cls(*range(len(fields)))
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is cls
        assert restored == value


@pytest.mark.parametrize("use_cuda", [0, 1], ids=["cpu", "cuda"])
def test_linalg_domain_values_and_solve_gradients(use_cuda):
    if use_cuda and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA unavailable")
    a_np = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    b_np = np.array([2., 5.], dtype=np.float32)
    with jt.flag_scope(use_cuda=use_cuda):
        a, b = jt.array(a_np), jt.array(b_np)
        solution = jt.linalg.solve(a, b)
        da, db = jt.grad(solution.sum(), [a, b])
        expected = np.linalg.solve(a_np, b_np)
        rhs_grad = np.linalg.solve(a_np.T, np.ones_like(b_np))
        np.testing.assert_allclose(solution.numpy(), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(db.numpy(), rhs_grad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(da.numpy(), -np.outer(rhs_grad, expected), rtol=1e-5, atol=1e-6)
        u, s, vh = jt.linalg.svd(a)
        np.testing.assert_allclose(u.numpy() @ np.diag(s.numpy()) @ vh.numpy(), a_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(jt.linalg.matrix_norm(a).numpy(), np.linalg.norm(a_np), rtol=1e-5)
        np.testing.assert_allclose(jt.linalg.einsum("ij,j->i", a, b).numpy(), a_np @ b_np, rtol=1e-5)
        jt.sync_all(True)
