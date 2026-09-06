"""Offline CUDA source-builder contracts; never import or compile Jittor."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "backends" / "cuda" / "kernels"


def load_definitions(path, names=None, **namespace):
    tree = ast.parse(path.read_text(encoding="utf8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and names is None:
            selected.append(node)
    tree.body = selected
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class NativeCudaSources(unittest.TestCase):
    def test_native_domains_do_not_store_cuda_algorithms(self):
        for filename in ("misc/tensor_ops.py", "math_util/gamma.py", "math_util/igamma.py",
                         "distributions.py", "math_util/src/igamma.h"):
            source = (ROOT / "python" / "jittor" / filename).read_text(encoding="utf8")
            for token in ("__global__", "__device__", "<<<"):
                self.assertNotIn(token, source, filename)

    def test_auto_parallel_keeps_cpu_body_outside_cuda_generator(self):
        codegen = load_definitions(KERNELS / "misc" / "codegen.py")
        native = load_definitions(ROOT / "python" / "jittor" / "misc" / "tensor_ops.py",
                                  {"auto_parallel"}, _cuda_codegen=codegen)
        generated = native.auto_parallel(
            2, "void sample(int n0, int i0, int n1, int i1, float* out) "
               "{ out[i0*n1+i1] = i1; }", block_num=256)
        gpu, cpu = generated.split("#else", 1)
        self.assertIn("sample_entry<<<p1,p2>>>", gpu)
        self.assertIn("tid = tid>>tn0", gpu)
        self.assertIn("for (int i0=0; i0<n0; i0++)", cpu)
        self.assertIn("for (int i1=0; i1<n1; i1++)", cpu)
        self.assertIn("sample_inner(n0,i0,n1,i1,out)", cpu)
        self.assertNotIn("__global__", cpu)

    def test_stack_cuda_accepts_a_caller_owned_cpu_source(self):
        class Var:
            def __init__(self, shape, dtype="float32"):
                self.shape, self.dtype = shape, dtype

            def reshape(self, shape):
                return Var(shape, self.dtype)

        calls = []

        def code(shape, dtype, inputs, **sources):
            calls.append(sources)
            return Var(shape, dtype)

        kernels = load_definitions(
            KERNELS / "misc" / "tensor_ops.py", jt=SimpleNamespace(Var=Var, code=code),
            _output_requires_grad=lambda value: False, _stop_grad_outputs=lambda value: value)
        kernels._stack_no_grad_cuda_fast(
            [Var((8, 2)), Var((8, 2))], 0,
            cpu_source=lambda suffix, n, writes: "caller_owned_cpu_source")
        self.assertEqual(calls[0]["cpu_src"], "caller_owned_cpu_source")
        self.assertIn("stack_kernel<<<grid, block>>>", calls[0]["cuda_src"])

    def test_polygamma_injects_shared_math_only_once(self):
        kernels = load_definitions(KERNELS / "math" / "gamma.py")
        header = kernels.polygamma_cuda_header("/* shared_cpu_math */\n")
        self.assertEqual(header.count("/* shared_cpu_math */"), 1)
        self.assertTrue(header.startswith("#define C10_HOST_DEVICE __host__ __device__\n"))
        self.assertIn("__global__ void polygamma_cuda", header)
        self.assertNotIn("static inline scalar_t zeta", header)

    def test_igamma_joins_shared_math_and_gpu_launch(self):
        calls = []

        def code(*args, **kwargs):
            calls.append(kwargs)
            return "result"

        kernels = load_definitions(KERNELS / "math" / "igamma.py",
                                   jt=SimpleNamespace(code=code))
        result = kernels.igamma(2.0, SimpleNamespace(shape=(8,), dtype="float32"),
                                "/* shared_cpu_math */\n")
        self.assertEqual(result, "result")
        self.assertEqual(calls[0]["cuda_header"].count("/* shared_cpu_math */"), 1)
        self.assertIn("igamma_kernel<<<batch_size, 16>>>", calls[0]["cuda_src"])
        self.assertEqual(calls[0]["data"], {"alpha": 2.0})


if __name__ == "__main__":
    unittest.main()
