"""Host-only contracts for optional ACL routing and native indexing ownership."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def definitions(relative, names=None, **namespace):
    path = ROOT / "python" / "jittor" / relative
    tree = ast.parse(path.read_text(encoding="utf8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (names is None or node.name in names)]
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class IndexingRouting(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.native_calls = []
        self.cascades = []
        self.provider_result = None
        native_calls, cascades = self.native_calls, self.cascades

        class Var:
            def __init__(self, source="input", parent=None, dtype="float32"):
                self.source, self.parent, self.dtype = source, parent, dtype
                self.ndim, self.shape = 2, [2, 2]
                self.assignments = []
                self.casts = []
                self.where_calls = 0
                self.view = None

            def cast(self, dtype):
                self.casts.append(dtype)
                return Var("cast", dtype=dtype)

            def stop_grad(self):
                self.stopped = True
                return self

            def getitem(self, slices, return_x=None):
                native_calls.append(("get", self, slices, return_x))
                result = Var("native_getitem", parent=self)
                return result if return_x is None else (result, self)

            def setitem(self, slices, value, reduce=None):
                native_calls.append(("set", self, slices, value, reduce))
                return Var("native_setitem")

            def _needs_cascade_setitem(self):
                return self.parent is not None

            # 5.02: the tensor-level index records the view it created, and a
            # recorded view is what makes the ancestry walk unnecessary.
            def _set_view_of(self, base, slices):
                self.view = (base, slices)
                return self

            def _is_view(self):
                return self.view is not None

            def check_cascade_setitem(self, result):
                if result.source != "native_setitem":
                    raise AssertionError("native cascade received a foreign producer")
                cascades.append((self, result))
                return self.assign(result)

            def assign(self, result):
                self.assignments.append(result)
                return self

            def where(self):
                self.where_calls += 1
                return (Var("row", dtype="int64"), Var("column", dtype="int64"))

        self.Var = Var
        self.jt = SimpleNamespace(Var=Var, array=lambda value, dtype: Var("scalar", dtype=dtype))

        def dispatch(operation, *args):
            self.calls.append((operation, args))
            return self.provider_result

        self.owner = definitions("misc/indexing.py", jt=self.jt, np=np,
                                 try_dispatch=dispatch,
                                 dispatch_context=lambda *args: SimpleNamespace(backend="cpu"),
                                 _native_var_getitem=Var.getitem,
                                 _native_var_setitem=Var.setitem)
        self.owner.install_var_indexing()
        self.jt.getitem = self.owner.var_getitem
        self.jt.setitem = self.owner.var_setitem

    def test_root_var_and_indexing_keep_explicit_function_identities(self):
        self.assertIs(self.jt.getitem, self.Var.getitem)
        self.assertIs(self.jt.setitem, self.Var.setitem)
        self.assertIs(self.Var.__getitem__, self.owner.getitem)
        self.assertIs(self.Var.__setitem__, self.owner.setitem)
        self.assertIs(self.Var.slice_var, self.owner.getitem)

    def test_setitem_returns_a_result_and_assignment_writes_back(self):
        x = self.Var()
        self.provider_result = self.Var("acl_setitem")
        result = self.jt.setitem(x, slice(None), 7)
        self.assertIs(result, self.provider_result)
        self.assertEqual(x.assignments, [])
        x[:, :] = 7
        self.assertEqual(x.assignments, [result])
        self.assertEqual(self.cascades, [])
        self.assertEqual(self.native_calls, [])

    def test_integer_getters_and_held_chain_assignment_keep_native_producers(self):
        self.provider_result = self.Var("acl_result")
        x = self.Var()
        for getter in (lambda: self.jt.getitem(x, 0), lambda: x.getitem((0,)),
                       lambda: x[0]):
            view = getter()
            self.assertEqual(view.source, "native_getitem")
            view[:, :] = 3
        # No getter may let the optional provider produce the result a writeback
        # consumes; that guard is still `_needs_cascade_setitem` alone, so which
        # results a backend is allowed to produce did not change with 5.02.
        self.assertEqual(self.calls, [])
        # The first two are the raw op, which deliberately does not create a
        # view -- a gather is a computation, not a claim about two names -- so
        # they still go through the ancestry walk. `x[0]` is the tensor-level
        # index, which records the view, and a recorded view writes back through
        # `assign` instead.
        self.assertEqual(len(self.cascades), 2)
        self.assertIsNone(self.jt.getitem(x, 0).view)
        self.assertEqual(x[0].view, (x, 0))

    def test_return_x_and_reduction_overloads_bypass_optional_provider(self):
        self.provider_result = self.Var("acl_result")
        x = self.Var()
        result, original = x.getitem(slice(None), 1)
        self.assertIs(original, x)
        self.assertEqual(result.source, "native_getitem")
        result = x.setitem(slice(None), 2, "add")
        self.assertEqual(result.source, "native_setitem")
        self.assertEqual(self.native_calls[-1][-1], "add")
        self.assertEqual(self.calls, [])
        self.assertEqual(x.assignments, [])

    def test_boolean_masks_reach_acl_without_host_coordinate_conversion(self):
        x, mask = self.Var(), self.Var(dtype="bool")
        self.provider_result = self.Var("acl_getitem")
        self.assertIs(x[mask], self.provider_result)
        self.assertIs(self.calls[-1][1][1], mask)
        self.assertEqual(mask.where_calls, 0)
        self.provider_result = None
        x.getitem(mask)
        self.assertEqual(mask.where_calls, 1)
        self.assertIsInstance(self.native_calls[-1][2], tuple)

    def test_provider_failure_does_not_try_native_fallback(self):
        error = RuntimeError("ACL execution failed")

        def failure(*args):
            raise error

        self.owner.var_getitem.__globals__["try_dispatch"] = failure
        with self.assertRaises(RuntimeError) as caught:
            self.Var().getitem(slice(None))
        self.assertIs(caught.exception, error)
        self.assertEqual(self.native_calls, [])

    def test_value_cast_is_limited_to_acl_plain_assignment(self):
        x, value = self.Var(dtype="float16"), self.Var(dtype="float32")
        namespace = self.owner.var_setitem.__globals__
        namespace["dispatch_context"] = lambda *args: SimpleNamespace(backend="acl")
        self.owner.var_setitem(x, slice(None), value)
        converted = self.native_calls[-1][3]
        self.assertEqual(converted.dtype, "float16")
        self.assertEqual(value.casts, ["float16"])
        self.owner.var_setitem(x, slice(None), value, "add")
        self.assertIs(self.native_calls[-1][3], value)
        self.assertEqual(value.casts, ["float16"])
        self.owner.var_setitem(x, slice(None), 0.0)
        scalar = self.native_calls[-1][3]
        self.assertEqual(scalar.dtype, "float16")
        self.assertTrue(scalar.stopped)
        for backend in ("cpu", "cuda"):
            namespace["dispatch_context"] = lambda *args: SimpleNamespace(backend=backend)
            self.owner.var_setitem(x, slice(None), value)
            self.assertIs(self.native_calls[-1][3], value)
        self.assertEqual(value.casts, ["float16"])


class DomainRouting(unittest.TestCase):
    def test_misc_owners_call_their_explicit_tensor_keys(self):
        calls = []
        result = object()

        def dispatch(key, *args):
            calls.append((key, args))
            return result

        class Var:
            ndim = 2

        jt = SimpleNamespace(Var=Var, misc=SimpleNamespace(_cumsum_dim=lambda dim, ndim: dim % ndim))
        names = {"all", "any", "flip", "nonzero", "split", "cumsum", "cub_cumsum",
                 "gather", "roll", "triu", "_scatter_into"}
        owner = definitions("misc/tensor_ops.py", names, jt=jt, try_dispatch=dispatch)
        x = Var()
        cases = [("all", (x,), "all"), ("any", (x,), "any"),
                 ("flip", (x, -1), "flip"), ("nonzero", (x,), "nonzero"),
                 ("split", (x, [1, 1]), "split"), ("cumsum", (x, -1), "cumsum"),
                 ("cub_cumsum", (x, -1), "cumsum"), ("gather", (x, 0, x), "gather"),
                 ("roll", (x, 1), "roll"), ("triu", (x, 0), "triu"),
                 ("_scatter_into", (x, 0, x, x, "add"), "scatter")]
        for name, args, key in cases:
            self.assertIs(getattr(owner, name)(*args), result)
            self.assertEqual(calls[-1][0], "tensor." + key)

    def test_builtin_adapters_keep_overloads_and_native_misses(self):
        calls, native = [], []
        result = object()

        def dispatch(key, *args):
            calls.append((key, args))
            return result

        def fallback(*args, **kwargs):
            native.append((args, kwargs))
            return "native"

        names = {"index", "arg_reduce", "where", "floor_int", "sigmoid"}
        owner = definitions("_runtime/core_api.py", names, _try_dispatch=dispatch,
                            NanoString=type("NanoString", (), {}), np=np,
                            **{"_native_" + name: fallback for name in names})
        self.assertIs(owner.index(shape=[2, 2], dtype="int64"), result)
        self.assertEqual(calls[-1], ("tensor.index", ([2, 2], None, "int64")))
        self.assertIs(owner.index([2, 2], "int64"), result)
        self.assertIs(owner.index([2, 2], np.int64), result)
        self.assertEqual(calls[-1][1][1:], (None, np.int64))
        self.assertIs(owner.where(1, 2, 3), result)
        self.assertIs(owner.where(cond=1, x=2, y=3), result)
        self.assertIs(owner.arg_reduce(1, "max", 0, True), result)
        self.assertIs(owner.floor_int(1), result)
        self.assertIs(owner.sigmoid(1), result)
        # Each function closes over the same execution namespace.
        owner.index.__globals__["_try_dispatch"] = lambda *args: None
        self.assertEqual(owner.index([2, 2], dtype="int64"), "native")
        self.assertEqual(native[-1], (([2, 2],), {"dtype": "int64"}))
        self.assertEqual(owner.where(1, "int32"), "native")
        self.assertEqual(native[-1], ((1, "int32"), {}))

    def test_concat_preserves_validation_and_dtype_promotion_before_provider(self):
        class Var:
            shape = [2, 2]

            def __init__(self, dtype):
                self.dtype = dtype

            def cast(self, dtype):
                return Var(dtype)

        calls = []
        result = object()

        def kernel(inputs, dim):
            calls.append((inputs, dim))
            return result

        def select(key, inputs, dim):
            self.assertEqual(key, "tensor.concat")
            return kernel

        jt = SimpleNamespace(flag_scope=lambda **kwargs: nullcontext(),
                             flags=SimpleNamespace(amp_reg=0),
                             amp_flags=SimpleNamespace(keep_reduce=4),
                             binary_dtype_infer=lambda *args: "float32")
        owner = definitions("misc/concatenation.py", {"concat", "_merge_dtypes"},
                            jt=jt, Sequence=(list, tuple), select_kernel=select)
        inputs = (Var("int32"), Var("float32"))
        self.assertIs(owner.concat(inputs, -1), result)
        self.assertEqual([value.dtype for value in calls[0][0]], ["float32", "float32"])
        self.assertEqual(calls[0][1], 1)
        self.assertEqual(inputs[0].dtype, "int32")
        with self.assertRaises(IndexError):
            owner.concat(inputs, 3)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
