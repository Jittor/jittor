"""Compile-cache ownership must last exactly as long as the cached entry."""
import ast
import gc
from pathlib import Path
import sys
import threading
import types
import weakref
from unittest import mock


def test_cache_retains_callable_identity_and_releases_it_when_cleared():
    path = Path(__file__).resolve().parents[2] / "triton/backend.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_compile")
    compiler = types.ModuleType("triton.compiler")
    compiler.ASTSource = object
    backends = types.ModuleType("triton.backends.compiler")
    backends.GPUTarget = lambda *args: args

    class Kernel:
        def __init__(self, name):
            self.__name__ = name

    def compile_kernel(kernel, **kwargs):
        return types.SimpleNamespace(
            metadata=types.SimpleNamespace(name=kernel.__name__, num_warps=4),
            asm={"cubin": kernel.__name__.encode(), "ptx": ""})

    namespace = {
        "real_triton": lambda: types.SimpleNamespace(compile=compile_kernel),
        "_Driver": types.SimpleNamespace(get=lambda device: types.SimpleNamespace(arch=89)),
        "_COMPILE_CACHE": {}, "_compile_lock": threading.Lock(),
        "_make_ast_source": lambda cls, kernel, signature, constants: kernel,
        "_ptx_param_count": lambda *args: 0, "JittorTritonError": RuntimeError,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with mock.patch.dict(sys.modules, {"triton.compiler": compiler,
                                       "triton.backends.compiler": backends}):
        kernel = Kernel("first")
        reference = weakref.ref(kernel)
        namespace["_compile"](kernel, {"x": "*fp32"}, {})
        del kernel
        gc.collect()
        assert reference() is not None, "the cache has lost the object behind its ID key"
        second = Kernel("second")
        assert namespace["_compile"](second, {"x": "*fp32"}, {})["name"] == "second"
        namespace["_COMPILE_CACHE"].clear()
        gc.collect()
        assert reference() is None, "clearing the cache must release its callable owners"
