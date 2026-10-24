"""Host-only device contract tests; no framework import, JIT or device proof."""

from contextlib import ExitStack, contextmanager
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "workflow_probe", Path(__file__).with_name("probe_training_workflow.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "workflow_compare", Path(__file__).with_name("compare_training_workflow.py")
)
comparator = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(comparator)


class TestFullGradientContract(unittest.TestCase):
    def test_explicit_float_precedes_placement_and_uses_returned_module(self):
        events = []
        converted = object()
        placed = object()
        class Module:
            def float(self):
                events.append("float")
                return converted
        def move(module):
            self.assertIs(module, converted)
            events.append("move")
            return placed
        self.assertIs(probe.prepare_fp32_model(Module(), move), placed)
        self.assertEqual(events, ["float", "move"])

    def test_every_model_role_is_prepared_before_parameter_observation(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        prepared = {}
        for node in ast.walk(run):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "prepare_fp32_model":
                prepared.update({target.id: node.lineno for target in node.targets if isinstance(target, ast.Name)})
        self.assertEqual(set(prepared), {"model", "critic", "reference", "reward_model"})
        first_observation = min(node.lineno for node in ast.walk(run)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "observe_parameters")
        self.assertTrue(all(line < first_observation for line in prepared.values()))

    def config(self, hybrid=False):
        value = dict(model_type="qwen3", vocab_size=32, hidden_size=16,
                     intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=2, head_dim=8, tie_word_embeddings=False,
                     attention_bias=False)
        if hybrid:
            value.update(model_type="qwen3_5_text", vocab_size=248320, hidden_size=1024,
                         intermediate_size=3584, num_hidden_layers=24, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=256, tie_word_embeddings=True,
                         layer_types=["linear_attention"] * 3 + ["full_attention"])
            value["layer_types"] *= 6
            value.update(linear_num_key_heads=16, linear_num_value_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128,
                         linear_conv_kernel_dim=4)
        return value

    def test_fixed_checkpoint_shapes_and_tied_aliases(self):
        actor = probe.parameter_spec(self.config(True), "actor")
        critic = probe.parameter_spec(self.config(True), "critic")
        self.assertEqual(len(actor["shapes"]), 320)
        self.assertEqual(actor["numel"], 752393024)
        self.assertEqual(len(critic["shapes"]), 321)
        self.assertEqual(critic["numel"], 752394048)
        self.assertEqual(actor["aliases"], [["lm_head.weight", "model.embed_tokens.weight"]])
        self.assertEqual(critic["aliases"], [])

    def test_loading_rejects_missing_text_and_unexpected_text(self):
        config = self.config(True)
        empty = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        for info in ({"missing_keys": ["layers.0.mlp.up_proj.weight"]},
                     {"unexpected_keys": ["model.language_model.norm.weight"]},
                     {"mismatched_keys": ["norm.weight"]}, {"error_msgs": ["bad"]}):
            with self.subTest(info=info), self.assertRaises(AssertionError):
                probe.validate_loading_info(dict(empty, **info), config, "critic")
        result = probe.validate_loading_info(dict(empty, unexpected_keys=["model.visual.x", "mtp.fc.weight"]),
                                             config, "critic")
        self.assertEqual(len(result["unexpected_keys"]), 2)
        with self.assertRaisesRegex(AssertionError, "missing loading diagnostics"):
            probe.validate_loading_info({}, config, "critic")

    def test_all_parameter_dtypes_aliases_and_names_are_observed(self):
        config = self.config(True)
        spec = probe.parameter_spec(config, "actor")
        class Parameter:
            dtype = "float32"
            def __init__(self, shape):
                self.shape = shape
            def numel(self):
                return int(np.prod(self.shape))
        parameters = {name: Parameter(shape) for name, shape in spec["shapes"].items()}
        all_parameters = dict(parameters, **{"lm_head.weight": parameters["model.embed_tokens.weight"]})
        module = SimpleNamespace(named_parameters=lambda remove_duplicate=True:
                                 (parameters if remove_duplicate else all_parameters).items())
        checked = []
        torch = SimpleNamespace(float32="float32")
        self.assertEqual(probe.observe_parameters(module, config, "actor", torch, checked.append), spec)
        self.assertEqual(len(checked), 321)
        all_parameters["lm_head.weight"] = Parameter([config["vocab_size"], config["hidden_size"]])
        with self.assertRaisesRegex(AssertionError, "aliases"):
            probe.observe_parameters(module, config, "actor", torch, checked.append)
        all_parameters["lm_head.weight"] = parameters["model.embed_tokens.weight"]
        parameters["model.norm.weight"].dtype = "bfloat16"
        # A public cast that silently does nothing must still fail the actual
        # full-parameter readback; a successful method call is not evidence.
        module.float = lambda: module
        prepared = probe.prepare_fp32_model(module, lambda value: value)
        with self.assertRaisesRegex(AssertionError, "non-FP32"):
            probe.observe_parameters(prepared, config, "actor", torch, checked.append)
        parameters["model.norm.weight"].dtype = "float32"
        del all_parameters["lm_head.weight"]
        with self.assertRaisesRegex(AssertionError, "enumeration"):
            probe.observe_parameters(module, config, "actor", torch, checked.append)
        del parameters["model.norm.weight"]
        with self.assertRaisesRegex(AssertionError, "names/shapes"):
            probe.observe_parameters(module, config, "actor", torch, checked.append)

    def test_embedding_capture_uses_public_retain_without_detach(self):
        class Tensor:
            requires_grad = True
            def retain_grad(self):
                self.retained = True
            def detach(self):
                raise AssertionError("must preserve embedding parameter graph")
        tensor = Tensor()
        module = SimpleNamespace(get_input_embeddings=lambda: lambda ids: tensor)
        self.assertIs(probe.embedding_input(module, object()), tensor)
        self.assertTrue(tensor.retained)
        tensor.requires_grad = False
        with self.assertRaises(AssertionError):
            probe.embedding_input(module, object())

    def test_qwen35_adapter_requires_both_real_observed_bindings(self):
        def original(*args):
            return args
        def replacement(*args):
            return args
        replacement._jittor_transformers_qwen35_functional = True
        module = ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
        package = ModuleType("transformers.models.qwen3_5")
        package.modeling_qwen3_5 = module
        names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
        with patch.dict(sys.modules, {"transformers.models.qwen3_5": package}), \
                patch.object(probe.importlib.metadata, "version", return_value="5.9.0"):
            for name in names:
                setattr(module, name, original)
            with self.assertRaisesRegex(AssertionError, "activation"):
                probe.observe_qwen35_adapter(self.config(True), "jittor")
            self.assertFalse(probe.observe_qwen35_adapter(self.config(True), "torch")["active"])
            setattr(module, names[0], replacement)
            with self.assertRaisesRegex(AssertionError, "activation"):
                probe.observe_qwen35_adapter(self.config(True), "jittor")
            setattr(module, names[1], replacement)
            evidence = probe.observe_qwen35_adapter(self.config(True), "jittor")
            self.assertTrue(evidence["active"])
            self.assertEqual([entry["name"] for entry in evidence["bindings"]], list(names))
            self.assertTrue(all(len(entry["source_sha256"]) == 64 for entry in evidence["bindings"]))
            with self.assertRaisesRegex(AssertionError, "activation"):
                probe.observe_qwen35_adapter(self.config(True), "torch")

    def test_qwen35_actual_instance_fused_selection_is_rejected(self):
        def functional(*args):
            return args
        def fused(*args):
            return args
        functional._jittor_transformers_qwen35_functional = True
        fused._jittor_transformers_qwen35_functional = True  # marker alone must not pass
        hf = ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
        hf.Qwen3_5GatedDeltaNet = type("Qwen3_5GatedDeltaNet", (), {"__module__": hf.__name__})
        hf.torch_chunk_gated_delta_rule = hf.torch_recurrent_gated_delta_rule = functional
        package = ModuleType("transformers.models.qwen3_5")
        package.modeling_qwen3_5 = hf
        config = self.config(True)
        layers = {}
        for index, layer_type in enumerate(config["layer_types"]):
            if layer_type == "linear_attention":
                layer = hf.Qwen3_5GatedDeltaNet()
                layer.chunk_gated_delta_rule = layer.recurrent_gated_delta_rule = functional
                layers["model.layers.{}.linear_attn".format(index)] = layer
        model = SimpleNamespace(named_modules=lambda: layers.items())
        with patch.dict(sys.modules, {"transformers.models.qwen3_5": package}), \
                patch.object(probe.importlib.metadata, "version", return_value="5.9.0"):
            first = next(iter(layers.values()))
            first.chunk_gated_delta_rule = fused
            self.assertTrue(probe.observe_qwen35_adapter(config, "jittor")["active"])
            with self.assertRaisesRegex(AssertionError, "actual.*binding"):
                probe.observe_qwen35_instances(model, config, "actor", "jittor")
            self.assertIs(first.chunk_gated_delta_rule, fused)  # observer must never fix binding
            first.chunk_gated_delta_rule = functional
            evidence = probe.observe_qwen35_instances(model, config, "actor", "jittor")
            self.assertEqual(evidence["expected_layers"], 18)
            self.assertEqual(len(evidence["layers"]), 18)
            critic_layers = {name.replace("model.", "backbone.", 1): layer for name, layer in layers.items()}
            critic = SimpleNamespace(named_modules=lambda: critic_layers.items())
            self.assertEqual(probe.observe_qwen35_instances(critic, config, "critic", "jittor")["expected_layers"], 18)
            with self.assertRaisesRegex(AssertionError, "marker"):
                probe.observe_qwen35_instances(model, config, "actor", "torch")
            del functional._jittor_transformers_qwen35_functional
            official = probe.observe_qwen35_instances(model, config, "actor", "torch")
            self.assertTrue(all(not binding["functional_adapter"] for layer in official["layers"]
                                for binding in layer["bindings"]))
            with self.assertRaisesRegex(AssertionError, "marker"):
                probe.observe_qwen35_instances(model, config, "actor", "jittor")
            functional._jittor_transformers_qwen35_functional = True
            first.recurrent_gated_delta_rule = fused
            with self.assertRaisesRegex(AssertionError, "actual.*binding"):
                probe.observe_qwen35_instances(model, config, "actor", "jittor")
            first.recurrent_gated_delta_rule = functional
            removed = layers.pop(next(iter(layers)))
            with self.assertRaisesRegex(AssertionError, "layer coverage"):
                probe.observe_qwen35_instances(model, config, "actor", "jittor")
            layers["wrong.path"] = removed
            with self.assertRaisesRegex(AssertionError, "layer coverage"):
                probe.observe_qwen35_instances(model, config, "actor", "jittor")

    def test_schema4_rejects_scalar_only_input_evidence(self):
        manifest = {"schema_version": 4, "parameter_names": ["weight"],
                    "critic": "zero", "critic_parameter_names": [], "algorithm": "ppo",
                    "steps": 1, "executed_steps": [0]}
        # All old schema3 keys exist. Only the new real embedding evidence is absent.
        old_keys = ["input_ids", "response", "rewards", "response_mask", "values",
                    "advantages", "returns", "old_log_prob", "log_prob", "logits",
                    "loss", "clip_fraction", "ppo_kl", "lower_clip_fraction", "input_scale",
                    "input_grad/input_scale", "post_update_log_prob", "post_update_loss",
                    "token_level_scores", "param_grad/weight", "post_weight/weight"]
        with self.assertRaisesRegex(AssertionError, "embedding"):
            comparator.validate_coverage(manifest, {"step_000/" + k for k in old_keys})


class TestDeviceContract(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.torch = ModuleType("torch")
        self.torch._C = object()
        self.torch.Tensor = type("Tensor", (), {})
        self.torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: self.events.append("cuda_sync"),
        )
        self.torch.npu = SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: self.events.append("npu_sync"),
        )
        self.precision = "high"
        self.torch.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        )
        self.torch.set_float32_matmul_precision = lambda value: setattr(self, "precision", value)
        self.torch.get_float32_matmul_precision = lambda: self.precision
        self.jt = ModuleType("jittor")
        self.jt.Var = type("Var", (), {})
        self.jt.compiler = SimpleNamespace(
            has_cuda=True, has_acl=False, has_rocm=False, has_corex=False,
        )
        self.jt.flags = SimpleNamespace(use_cuda=0)

        @contextmanager
        def scope(**kwargs):
            self.events.append(("scope", kwargs))
            previous = self.jt.flags.use_cuda
            if "use_cuda" in kwargs:
                self.jt.flags.use_cuda = kwargs["use_cuda"]
            try:
                yield
            finally:
                self.jt.flags.use_cuda = previous

        @contextmanager
        def forbidden():
            self.events.append("fallback_forbidden")
            yield

        self.jt.runtime = SimpleNamespace(scope=scope)
        self.jt.sync_all = lambda device: self.events.append(("jt_sync", device))
        activation = ModuleType("jittor.compat.shim.runtime")
        activation.activation_status = lambda jt: SimpleNamespace(active=True)
        publication = ModuleType("jittor.compat.torch.publication")
        publication.namespace_owner = lambda torch: self.jt
        fallback = ModuleType("jittor._runtime.fallback")
        fallback.forbid_backend_fallbacks = forbidden
        self.modules = {
            "torch": self.torch, "torch_npu": ModuleType("torch_npu"),
            "jittor": self.jt, "jittor.compat.shim.runtime": activation,
            "jittor.compat.torch.publication": publication,
            "jittor._runtime.fallback": fallback,
        }

    def initialize(self, framework, device, stack):
        return probe.initialize(SimpleNamespace(framework=framework, device=device), stack)

    def tensor(self, device):
        events = self.events

        class Tensor:
            def __init__(self):
                self.device = SimpleNamespace(type=device)

            def to(self, destination):
                events.append(("to", destination))
                return self

            def cuda(self):
                events.append("tensor_cuda")
                return self

            def location(self):
                return "device" if device != "cpu" else "cpu"

        return Tensor()

    def test_cli_accepts_cuda(self):
        with patch.object(sys, "argv", ["probe", "--framework", "torch", "--device",
                                       "cuda", "--output", "unused.npz"]):
            self.assertEqual(probe.parse_args().device, "cuda")

    def test_official_cuda_moves_syncs_and_checks_actual_device(self):
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            torch, jt, move, sync, check = self.initialize("torch", "cuda", stack)
            self.assertIsNone(jt)
            value = self.tensor("cuda")
            self.assertIs(move(value), value)
            self.assertIn(("to", "cuda"), self.events)
            sync()
            self.assertIn("cuda_sync", self.events)
            check(value)
            with self.assertRaises(AssertionError):
                check(self.tensor("cpu"))

    def test_official_cuda_unavailable_is_not_cpu_fallback(self):
        self.torch.cuda.is_available = lambda: False
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            with self.assertRaisesRegex(AssertionError, "CUDA.*unavailable"):
                self.initialize("torch", "cuda", stack)

    def test_official_oracle_rejects_shim(self):
        self.torch._torch_compat_install_context = object()
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            with self.assertRaisesRegex(AssertionError, "oracle is shim"):
                self.initialize("torch", "cuda", stack)

    def test_jittor_cuda_moves_syncs_and_forbids_fallback(self):
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            torch, jt, move, sync, check = self.initialize("jittor", "cuda", stack)
            self.assertIs(jt, self.jt)
            self.assertIn(("scope", {"use_cuda": 1}), self.events)
            self.assertIn(("scope", {"backend_fallback": "error"}), self.events)
            self.assertIn("fallback_forbidden", self.events)
            move(self.tensor("cuda"))
            self.assertIn("tensor_cuda", self.events)
            sync()
            self.assertIn(("jt_sync", True), self.events)
            self.assertEqual(check(self.tensor("cuda")), "device")
            with self.assertRaises(AssertionError):
                check(self.tensor("cpu"))
            self.jt.flags.use_cuda = 0
            with self.assertRaises(AssertionError):
                check(self.tensor("cuda"))

    def test_jittor_cuda_rejects_other_accelerator_builds(self):
        for attribute in ("has_acl", "has_rocm", "has_corex"):
            with self.subTest(attribute=attribute):
                setattr(self.jt.compiler, attribute, True)
                with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
                    with self.assertRaisesRegex(AssertionError, "not.*CUDA"):
                        self.initialize("jittor", "cuda", stack)
                setattr(self.jt.compiler, attribute, False)

    def test_jittor_cuda_requires_cuda_build(self):
        self.jt.compiler.has_cuda = False
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            with self.assertRaisesRegex(AssertionError, "CUDA.*unavailable"):
                self.initialize("jittor", "cuda", stack)

    def test_existing_cpu_and_npu_contracts(self):
        for framework in ("torch", "jittor"):
            for device in ("cpu", "npu"):
                with self.subTest(framework=framework, device=device):
                    self.events.clear()
                    self.jt.compiler.has_acl = device == "npu"
                    with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
                        torch, jt, move, sync, check = self.initialize(framework, device, stack)
                        value = self.tensor(device)
                        self.assertIs(move(value), value)
                        sync()
                        check(value)
                        if framework == "torch" and device == "npu":
                            self.assertIn(("to", "npu"), self.events)
                            self.assertIn("npu_sync", self.events)
                        if framework == "jittor" and device == "cpu":
                            self.assertEqual(self.jt.flags.use_cuda, 0)
                            self.assertIn(("jt_sync", False), self.events)

    def test_cuda_precision_is_fixed_and_read_back_on_both_runtimes(self):
        expected = {
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest", "cudnn_benchmark": False,
        }
        for framework in ("torch", "jittor"):
            with self.subTest(framework=framework):
                with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
                    self.initialize(framework, "cuda", stack)
                    self.assertEqual(probe.cuda_runtime_conditions(self.torch), expected)

    def test_cuda_precision_no_op_and_late_change_are_rejected(self):
        self.torch.set_float32_matmul_precision = lambda value: None
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            with self.assertRaisesRegex(AssertionError, "CUDA precision"):
                self.initialize("torch", "cuda", stack)
        self.torch.set_float32_matmul_precision = lambda value: setattr(self, "precision", value)
        with patch.dict(sys.modules, self.modules), patch.dict("os.environ", clear=True), ExitStack() as stack:
            self.initialize("torch", "cuda", stack)
            self.torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaisesRegex(AssertionError, "CUDA precision"):
                probe.cuda_runtime_conditions(self.torch)


class TestPrecisionArtifacts(unittest.TestCase):
    def setUp(self):
        self.conditions = {
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest", "cudnn_benchmark": False,
        }

    def manifest(self, device="cuda", conditions=True):
        value = {"device": device}
        if conditions:
            value["runtime_conditions"] = dict(self.conditions)
        return value

    def test_matching_cuda_runtime_conditions_pass(self):
        self.assertEqual(comparator.validate_runtime_conditions(self.manifest(), self.manifest()),
                         self.conditions)

    def test_missing_cuda_conditions_on_either_or_both_sides_fail(self):
        for left, right in ((True, False), (False, True), (False, False)):
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(AssertionError, "CUDA runtime conditions"):
                    comparator.validate_runtime_conditions(self.manifest(conditions=left),
                                                           self.manifest(conditions=right))

    def test_matching_wrong_cuda_conditions_are_not_accepted(self):
        for key, value in (("cuda_matmul_allow_tf32", True), ("cudnn_allow_tf32", True),
                           ("float32_matmul_precision", "high"), ("cudnn_benchmark", True),
                           ("cuda_matmul_allow_tf32", 0), ("cudnn_allow_tf32", None)):
            with self.subTest(key=key, value=value):
                left, right = self.manifest(), self.manifest()
                left["runtime_conditions"][key] = right["runtime_conditions"][key] = value
                with self.assertRaisesRegex(AssertionError, "CUDA runtime conditions"):
                    comparator.validate_runtime_conditions(left, right)

    def test_missing_cuda_field_or_asymmetric_conditions_fail(self):
        for key in self.conditions:
            with self.subTest(key=key):
                left, right = self.manifest(), self.manifest()
                del right["runtime_conditions"][key]
                with self.assertRaises(AssertionError):
                    comparator.validate_runtime_conditions(left, right)

    def test_legacy_cpu_and_npu_artifacts_need_no_cuda_conditions(self):
        for device in ("cpu", "npu"):
            with self.subTest(device=device):
                self.assertIsNone(comparator.validate_runtime_conditions(
                    self.manifest(device, conditions=False), self.manifest(device, conditions=False)))


if __name__ == "__main__":
    unittest.main()
