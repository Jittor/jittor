"""Host contracts only: fake runtimes are not GPU or performance evidence."""

import ast
from contextlib import ExitStack, contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy  # Load extension once, outside the temporary sys.modules map.


class TestBenchmarkCuda(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.current = None
        self.precision = "high"
        self.torch = ModuleType("torch")
        self.torch._C = object()
        self.torch.__version__ = "stub-host-contract"
        self.torch.Tensor = type("Tensor", (), {})
        self.torch.device = lambda kind, index=None: SimpleNamespace(type=kind, index=index)
        self.torch.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        )
        self.torch.set_float32_matmul_precision = lambda value: setattr(self, "precision", value)
        self.torch.get_float32_matmul_precision = lambda: self.precision

        def set_device(index):
            self.current = index
            self.jt.flags.device_id = index
            self.events.append(("set_device", index))

        def accelerator(name):
            return SimpleNamespace(
                is_available=lambda: True, set_device=set_device,
                current_device=lambda: self.current,
                synchronize=lambda: self.events.append((name, "sync")),
                reset_peak_memory_stats=lambda *a: self.events.append((name, "reset", a)),
                max_memory_allocated=lambda *a: 1024,
            )

        self.torch.cuda = accelerator("cuda")
        self.torch.npu = accelerator("npu")
        self.jt = ModuleType("jittor")
        self.jt.compiler = SimpleNamespace(
            has_cuda=True, has_acl=False, has_rocm=False, has_corex=False,
        )
        self.jt.flags = SimpleNamespace(use_cuda=0, device_id=0)
        self.jt.rank, self.jt.world_size = 0, 1
        self.fallback_count = 0
        self.jt.core = SimpleNamespace(backend_fallback_count=lambda: self.fallback_count)
        self.jt.sync_all = lambda device: self.events.append(("jt_sync", device))

        @contextmanager
        def scope(**kwargs):
            previous = self.jt.flags.use_cuda
            self.events.append(("scope", kwargs))
            self.jt.flags.use_cuda = kwargs.get("use_cuda", previous)
            try:
                yield
            finally:
                self.jt.flags.use_cuda = previous

        @contextmanager
        def forbid():
            self.events.append("fallback_forbidden")
            yield

        self.jt.runtime = SimpleNamespace(scope=scope)
        activation = ModuleType("jittor.compat.shim.runtime")
        activation.activation_status = lambda jt: SimpleNamespace(active=True)
        publication = ModuleType("jittor.compat.torch.publication")
        publication.namespace_owner = lambda torch: self.jt
        fallback = ModuleType("jittor._runtime.fallback")
        fallback.forbid_backend_fallbacks = forbid
        self.modules = {
            "torch": self.torch, "torch_npu": ModuleType("torch_npu"), "jittor": self.jt,
            "jittor.compat.shim.runtime": activation,
            "jittor.compat.torch.publication": publication,
            "jittor._runtime.fallback": fallback,
        }
        spec = importlib.util.spec_from_file_location(
            "cuda_benchmark_host_contract", Path(__file__).with_name("benchmark_llm.py")
        )
        self.bench = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules):
            spec.loader.exec_module(self.bench)

    def initialize(self, framework, device, guard, local_rank=0):
        return self.bench.initialize_runtime(
            SimpleNamespace(framework=framework, device=device), guard,
            rank=0, local_rank=local_rank, world_size=1,
        )

    def tensor(self, device, index=0):
        value = SimpleNamespace(device=SimpleNamespace(type=device, index=index))
        value.device_id = index if device != "cpu" else -1
        value.location = lambda: "cpu" if device == "cpu" else "device"
        value.to = lambda destination: self.events.append(("to", destination)) or value
        value.cuda = lambda: self.events.append("tensor_cuda") or value
        return value

    def test_jittor_tensor_device_must_match_rank_not_only_current_device(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            runtime = self.initialize("jittor", "cuda", guard, local_rank=0)
            for wrapped in (False, True):
                with self.subTest(wrapped=wrapped):
                    native = self.tensor("cuda", 1)
                    value = SimpleNamespace(_var=native) if wrapped else native
                    self.assertEqual(self.torch.cuda.current_device(), 0)
                    with self.assertRaisesRegex(AssertionError, "wrong CUDA device"):
                        runtime.check_location(value)

    def test_cli_accepts_cuda(self):
        with patch.object(sys, "argv", ["bench", "--framework", "torch", "--device", "cuda",
                                       "--task", "train", "--model", "offline", "--output", "unused"]):
            self.assertEqual(self.bench.parse_args().device, "cuda")

    def test_invalid_cuda_authorization_fails_before_backend_initialization(self):
        with patch.dict(self.bench.os.environ, {"VERL_ALLOW_SHARED_CUDA": "true"}, clear=True), \
                patch.object(self.bench, "parse_args", return_value=SimpleNamespace(device="cuda")), \
                patch.object(self.bench, "initialize_runtime") as initialize:
            with self.assertRaisesRegex(ValueError, "VERL_ALLOW_SHARED_CUDA"):
                self.bench.main()
            initialize.assert_not_called()

    def test_distributed_backend_uses_nccl_not_hccl_or_gloo_on_cuda(self):
        for device, expected in (("cpu", "gloo"), ("cuda", "nccl"), ("npu", "hccl")):
            self.assertEqual(self.bench.distributed_backend(device), expected)

    def test_torch_cuda_device_sync_memory_and_precision(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            runtime = self.initialize("torch", "cuda", guard, local_rank=2)
            self.assertEqual(runtime.device.index, 2)
            self.assertIn(("set_device", 2), self.events)
            runtime.synchronize()
            self.assertIn(("cuda", "sync"), self.events)
            runtime.reset_peak_memory()
            self.assertIn(("cuda", "reset", (2,)), self.events)
            self.assertEqual(runtime.peak_memory(), 1024)
            runtime.check_location(self.tensor("cuda", 2))
            with self.assertRaises(AssertionError):
                runtime.check_location(self.tensor("cpu"))
            with self.assertRaises(AssertionError):
                runtime.check_location(self.tensor("cuda", 1))
            self.assertEqual(runtime.runtime_conditions()["float32_matmul_precision"], "highest")
            self.torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaises(AssertionError):
                runtime.runtime_conditions()

    def test_cuda_unavailable_or_shim_or_wrong_backend_fails(self):
        self.torch.cuda.is_available = lambda: False
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            with self.assertRaisesRegex(AssertionError, "CUDA.*unavailable"):
                self.initialize("torch", "cuda", guard)
        self.torch.cuda.is_available = lambda: True
        self.torch._torch_compat_install_context = object()
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            with self.assertRaisesRegex(AssertionError, "shim"):
                self.initialize("torch", "cuda", guard)
        for name in ("has_acl", "has_rocm", "has_corex"):
            with self.subTest(name=name):
                setattr(self.jt.compiler, name, True)
                with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                    with self.assertRaisesRegex(AssertionError, "not native CUDA"):
                        self.initialize("jittor", "cuda", guard)
                setattr(self.jt.compiler, name, False)

    def test_jittor_cuda_is_on_device_forbids_fallback_and_reports_memory(self):
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            runtime = self.initialize("jittor", "cuda", guard, local_rank=2)
            self.assertIn(("scope", {"use_cuda": 1}), self.events)
            self.assertIn(("scope", {"backend_fallback": "error"}), self.events)
            self.assertIn("fallback_forbidden", self.events)
            self.assertEqual(runtime.check_location(self.tensor("cuda", 2)), "device")
            runtime.synchronize()
            self.assertIn(("jt_sync", True), self.events)
            runtime.reset_peak_memory()
            self.assertEqual(runtime.peak_memory(), 1024)
            self.assertEqual(runtime.check_fallbacks(), 0)
            self.fallback_count = 1
            with self.assertRaises(AssertionError):
                runtime.check_fallbacks()
            self.jt.flags.use_cuda = 0
            with self.assertRaises(AssertionError):
                runtime.check_location(self.tensor("cuda", 2))

    def test_cpu_and_npu_existing_runtime_contracts(self):
        for framework in ("torch", "jittor"):
            for device in ("cpu", "npu"):
                with self.subTest(framework=framework, device=device):
                    self.jt.compiler.has_acl = device == "npu"
                    with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                        runtime = self.initialize(framework, device, guard)
                        runtime.synchronize()
                        runtime.check_location(self.tensor(device))
                        self.assertIsNone(runtime.runtime_conditions())

    def test_jittor_cuda_build_required_and_peak_must_be_measured(self):
        self.jt.compiler.has_cuda = False
        with patch.dict(sys.modules, self.modules), ExitStack() as guard:
            with self.assertRaisesRegex(AssertionError, "CUDA.*unavailable"):
                self.initialize("jittor", "cuda", guard)
        self.jt.compiler.has_cuda = True
        for framework in ("torch", "jittor"):
            for peak in (0, -1):
                with self.subTest(framework=framework, peak=peak):
                    self.torch.cuda.max_memory_allocated = lambda *a: peak
                    with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                        runtime = self.initialize(framework, "cuda", guard)
                        with self.assertRaisesRegex(AssertionError, "allocator peak"):
                            runtime.peak_memory()

    def test_cuda_selection_and_precision_no_ops_are_rejected(self):
        original = self.torch.cuda.set_device
        self.torch.cuda.set_device = lambda index: None
        for framework in ("torch", "jittor"):
            with self.subTest(framework=framework):
                with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                    with self.assertRaisesRegex(AssertionError, "device selection"):
                        self.initialize(framework, "cuda", guard, local_rank=2)
        self.torch.cuda.set_device = original
        self.torch.set_float32_matmul_precision = lambda value: None
        for framework in ("torch", "jittor"):
            with self.subTest(framework=framework):
                with patch.dict(sys.modules, self.modules), ExitStack() as guard:
                    with self.assertRaisesRegex(AssertionError, "precision policy"):
                        self.initialize(framework, "cuda", guard)

    def test_performance_context_cannot_promote_shared_cuda_to_l4(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.bench.performance_context("cuda"), {
                "shared_cuda": False, "performance_eligible": True,
                "runtime_authorization": {"VERL_ALLOW_SHARED_CUDA": "0",
                                          "VERL_L4_PERFORMANCE_ELIGIBLE": "1"},
            })
        with patch.dict("os.environ", {"VERL_ALLOW_SHARED_CUDA": "1",
                                       "VERL_L4_PERFORMANCE_ELIGIBLE": "1"}, clear=True):
            result = self.bench.performance_context("cuda")
            self.assertIs(result["shared_cuda"], True)
            self.assertIs(result["performance_eligible"], False)

    def test_performance_context_rejects_invalid_cuda_authorization(self):
        for name in ("VERL_ALLOW_SHARED_CUDA", "VERL_L4_PERFORMANCE_ELIGIBLE"):
            for value in ("", "true", "2"):
                with self.subTest(name=name, value=value):
                    with patch.dict("os.environ", {name: value}, clear=True):
                        with self.assertRaisesRegex(ValueError, name):
                            self.bench.performance_context("cuda")

    def run_inference_timing_contract(self, framework, warmup):
        """Execute the real infer timing AST with asynchronous fake work.

        This deliberately avoids loading a model or either runtime. The fake
        clock advances only at synchronize(), so missing sync boundaries or a
        first-use prefill inside the timed samples cannot accidentally pass.
        """
        source = Path(__file__).with_name("benchmark_llm.py")
        tree = ast.parse(source.read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        infer = next(node.orelse for node in ast.walk(main)
                     if isinstance(node, ast.If) and any(
                         isinstance(item, ast.FunctionDef) and item.name == "prefill_once"
                         for item in node.orelse))
        start = next(index for index, node in enumerate(infer)
                     if isinstance(node, ast.FunctionDef) and node.name == "prefill_once")
        events, calls = [], []
        clock = SimpleNamespace(now=0.0, pending=0.0, prefill=0, generate=0)

        class HostTensor:
            def __init__(self, value):
                self.value = numpy.array(value)
                self.shape = self.value.shape

            def __getitem__(self, key):
                return HostTensor(self.value[key])

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.value.tolist()

        class HostModel:
            def __call__(self, **kwargs):
                events.append("prefill")
                calls.append(("prefill", kwargs))
                clock.pending += 100.0 if clock.prefill == 0 else 5.0
                clock.prefill += 1
                return SimpleNamespace(logits=HostTensor(numpy.zeros((1, 4, 8))))

            def generate(self, **kwargs):
                events.append("generate")
                calls.append(("generate", kwargs))
                clock.pending += 50.0 if clock.generate == 0 else 2.0
                clock.generate += 1
                return HostTensor([[1, 2, 3, 4, 5, 6]])

        def synchronize():
            events.append("sync")
            clock.now += clock.pending
            clock.pending = 0.0

        def tick():
            events.append("clock")
            return clock.now

        input_ids = HostTensor([[1, 2, 3, 4]])
        attention_mask = HostTensor([[1, 1, 1, 1]])
        namespace = {
            "torch": SimpleNamespace(no_grad=nullcontext),
            "model": HostModel(), "input_ids": input_ids, "attention_mask": attention_mask,
            "synchronize": synchronize, "barrier": lambda: events.append("barrier"),
            "reset_peak_memory": lambda: events.append("reset_peak"), "peak_memory": lambda: 123,
            "time": SimpleNamespace(perf_counter=tick), "statistics": self.bench.statistics,
            "percentile": self.bench.percentile, "prompt": "host-only timing fixture",
            "tokenizer": SimpleNamespace(eos_token_id=0, decode=lambda tokens, **kwargs: "tokens-5-6"),
            "args": SimpleNamespace(framework=framework, warmup=warmup, steps=2, new_tokens=2),
            "result": {}, "value_location": lambda value: "host-contract-not-device-evidence",
        }
        exec(compile(ast.Module(body=infer[start:], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["result"], events, calls, input_ids, attention_mask

    def test_prefill_has_independent_cold_and_warmup_before_timed_samples(self):
        for framework in ("torch", "jittor"):
            for warmup in (0, 2):
                with self.subTest(framework=framework, warmup=warmup):
                    result, events, calls, ids, mask = self.run_inference_timing_contract(framework, warmup)
                    # A first-use duration of 100 must never appear in timed
                    # samples, even though generate has already warmed up.
                    self.assertEqual(result["prefill_times_s"], [5.0, 5.0])
                    self.assertEqual(result["cold_prefill_s"], 100.0)
                    self.assertEqual(result["prefill_cold_runs"], 1)
                    self.assertEqual(result["prefill_warmup_runs"], warmup)
                    reset = events.index("reset_peak")
                    self.assertEqual(events[:reset].count("prefill"), 1 + warmup)
                    self.assertEqual(events[reset:].count("prefill"), 2)
                    self.assertEqual(result["cold_generate_s"], 50.0)
                    self.assertEqual(result["generate_times_s"], [2.0, 2.0])
                    self.assertEqual(events[:reset].count("generate"), max(1, warmup))
                    self.assertEqual(result["generated_token_ids"], [5, 6])
                    for index, event in enumerate(events):
                        if event == "prefill":
                            self.assertEqual(events[index + 1], "sync")
                            if events[index - 1] == "clock":
                                self.assertEqual(events[index - 2], "sync")
                                self.assertEqual(events[index + 2], "clock")
                    for kind, kwargs in calls:
                        self.assertIs(kwargs["input_ids"], ids)
                        self.assertIs(kwargs["attention_mask"], mask)
                        self.assertIs(kwargs["use_cache"], True)
                        if kind == "prefill":
                            self.assertEqual(set(kwargs), {"input_ids", "attention_mask", "use_cache"})


if __name__ == "__main__":
    unittest.main()
