"""Real CUDA constructor dtype regression for Conv1d/Conv3d/LayerNorm.

A skip is not backend acceptance. No constructor or execution is monkeypatched.
"""
import ctypes
import unittest

import pytest
import jittor as jt
import torch

pytestmark = pytest.mark.cuda


def run(torch, jt, pointer_probe):
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks
    before = counts()
    original_default = torch.get_default_dtype()
    result = dict(scope="actual_CUDA1_owned_constructor_dtype_NOT_full_VERL",
                  source_must_be_authenticated_by_caller=True, cases=[], checks={},
                  full_PPO_GRPO_or_L_claims=False)
    modules = []
    specs = (
        ("Conv1d", (2, 3, 2), {}, (1, 2, 4)),
        ("Conv3d", (2, 3, 2), {}, (1, 2, 3, 3, 3)),
        ("LayerNorm", ((2, 3),), {}, (2, 2, 3)),
    )

    def metadata(value):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
                    placement_backend=int(value.placement_backend),
                    device_id=int(value.device_id), public_device=str(value.device),
                    requires_grad=bool(value.requires_grad))

    def cuda(value):
        value.sync()
        pointer_probe(jt, value)
        return (int(value.placement_backend) == 1 and int(value.device_id) == 0
                and str(value.device) == "cuda:0")

    try:
        with forbid_backend_fallbacks(), jt.runtime.scope(
                use_cuda=1, device_id=0, backend_fallback="error"), torch.device("cuda:0"):
            for default, explicit, label in (
                    (torch.bfloat16, "absent", "default_bf16"),
                    (torch.float32, "absent", "default_fp32"),
                    (torch.bfloat16, torch.float32, "explicit_fp32"),
                    (torch.float32, torch.bfloat16, "explicit_bf16"),
                    (torch.bfloat16, None, "explicit_none_default_bf16")):
                torch.set_default_dtype(default)
                expected = default if explicit is None or explicit == "absent" else explicit
                for name, args, kwargs, shape in specs:
                    row = dict(name=name, case=label, expected_dtype=str(expected),
                               default_dtype=str(torch.get_default_dtype()),
                               native_amp_reg=int(jt.flags.amp_reg), phase="constructor")
                    result["cases"].append(row)
                    call_kwargs = dict(kwargs)
                    if explicit != "absent":
                        call_kwargs["dtype"] = explicit
                    module = getattr(torch.nn, name)(*args, **call_kwargs)
                    parameters = list(module.named_parameters())
                    row.update(constructor_returned=True,
                               parameters={key: metadata(value) for key, value in parameters})
                    row["parameters_real_cuda"] = bool(parameters) and all(cuda(p) for _, p in parameters)
                    row["dtype_exact"] = bool(parameters) and all(p.dtype == expected for _, p in parameters)
                    row["passed"] = row["parameters_real_cuda"] and row["dtype_exact"]
                    if name == "Conv1d":
                        row["immediate_hidden_parameters"] = {
                            key: metadata(getattr(module._conv[0], key)) for key in ("weight", "bias")}
                        row["immediate_hidden_alias_not_claimed"] = True
                    if label == "default_bf16" and row["passed"]:
                        modules.append((module, shape, row))
            for module, shape, row in modules:
                row["phase"] = "original_forward_backward"
                x = torch.ones(shape, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
                y = module(x)
                row["output"] = metadata(y)
                y.float().sum().backward()
                gradients = [x.grad] + [p.grad for p in module.parameters()]
                row["gradients"] = [metadata(g) if g is not None else None for g in gradients]
                row["actual_cuda_forward_backward_finite"] = (
                    cuda(x) and cuda(y) and bool(torch.isfinite(y).all().item())
                    and all(g is not None and g.dtype == torch.bfloat16 and cuda(g) and bool(torch.isfinite(g).all().item())
                            for g in gradients))
                if type(module).__name__ == "Conv1d":
                    inner = module._conv[0]
                    row["native_forward_refreshed_inner"] = (
                        inner.bias is module.bias and inner.weight.dtype == module.weight.dtype
                        and bool(torch.equal(inner.weight, module.weight.unsqueeze(-1))))
                    row["actual_cuda_forward_backward_finite"] &= row["native_forward_refreshed_inner"]
                row["passed"] &= row["actual_cuda_forward_backward_finite"]
    finally:
        torch.set_default_dtype(original_default)
    result["checks"] = dict(
        all_fifteen_constructor_cases_passed=len(result["cases"]) == 15
            and all(row["passed"] for row in result["cases"]),
        all_three_actual_cuda_forward_backward_returned=len(modules) == 3
            and all(row.get("actual_cuda_forward_backward_finite") is True for _, _, row in modules),
        default_dtype_restored=torch.get_default_dtype() == original_default,
        diagnostic_counts_unchanged=counts() == before,
        fallback_zero=int(jt.core.backend_fallback_count()) == 0)
    result["passed"] = all(result["checks"].values())
    return result



class TestOwnedConstructorDtypeCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('requires real CUDA; skip is not acceptance')
        torch.cuda.set_device(0)
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)

    def prove_cuda(self, backend, value):
        self.assertIs(backend, jt)
        self.assertEqual(tuple(jt.core.dispatch_context([value])), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        pointer = int(value.device_raw_ptr)
        self.assertGreater(pointer, 0)
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
        self.assertEqual(self.driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
        self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def test_owned_constructor_default_and_explicit_dtype(self):
        self.runtime_evidence = run(torch, jt, self.prove_cuda)
        self.assertTrue(self.runtime_evidence['passed'], self.runtime_evidence['checks'])
