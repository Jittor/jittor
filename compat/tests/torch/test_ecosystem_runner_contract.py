from _ecosystem_runner import (
    _evaluate_preserving_parameter_grads,
    _parameter_grad_state,
)
from _ecosystem_harness import (
    _assert_declared_parameter_grads,
    _assert_matching_parameter_grad_state,
)
import unittest


class _Parameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class _Model:
    def __init__(self):
        self.entries = [
            ("base.weight", _Parameter(False)),
            ("adapter.lora_A.weight", _Parameter(True)),
            ("adapter.lora_B.weight", _Parameter(True)),
        ]

    def named_parameters(self):
        return iter(self.entries)

    def parameters(self):
        return (parameter for _, parameter in self.entries)

    def eval(self):
        return self


def test_parameter_grad_state_preserves_frozen_parameters():
    model = _Model()

    state = _parameter_grad_state(model)

    assert state == {
        "trainable": ["adapter.lora_A.weight", "adapter.lora_B.weight"],
        "frozen": ["base.weight"],
    }
    assert [parameter.requires_grad for _, parameter in model.entries] == [False, True, True]


def test_eval_path_preserves_a_non_peft_model_freeze_contract():
    model = _Model()

    state = _evaluate_preserving_parameter_grads(model)

    assert state["frozen"] == ["base.weight"]
    assert model.entries[0][1].requires_grad is False


def test_harness_rejects_the_old_all_parameters_trainable_behavior():
    reference = {
        "parameters": {
            "trainable": ["adapter.lora_A.weight", "adapter.lora_B.weight"],
            "frozen": ["base.weight"],
        }
    }
    old_runner_result = {
        "parameters": {
            "trainable": [
                "base.weight", "adapter.lora_A.weight", "adapter.lora_B.weight"
            ],
            "frozen": [],
        }
    }

    with unittest.TestCase().assertRaisesRegex(
        AssertionError, "different trainable/frozen parameter set"
    ):
        _assert_matching_parameter_grad_state(
            unittest.TestCase(), "peft_lora", reference, old_runner_result
        )


def test_harness_rejects_when_both_runtimes_miss_a_declared_qwen_lora_grad():
    report = {
        "parameters": {
            "trainable": [
                "q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight",
                "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight",
            ],
            "frozen": ["q_proj.base_layer.weight"],
        },
        "parameter_grads": [
            "q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight",
            "v_proj.lora_B.default.weight",
        ],
    }

    with unittest.TestCase().assertRaisesRegex(
        AssertionError, "declared trainable-gradient contract"
    ):
        _assert_declared_parameter_grads(
            unittest.TestCase(), "peft_lora_qwen2", report
        )


class _PlacedTensor:
    def __init__(self):
        self.destination = None

    def to(self, device):
        self.destination = device
        return self


def test_npu_selection_explicitly_moves_host_values_and_scopes_policy():
    from contextlib import ExitStack, contextmanager
    import sys
    from types import SimpleNamespace
    from unittest import mock
    from _ecosystem_runner import _select_device

    changes = []

    @contextmanager
    def scope(**policy):
        changes.append(policy)
        yield

    backend = SimpleNamespace(
        runtime=SimpleNamespace(scope=scope),
        introspection=SimpleNamespace(capabilities=SimpleNamespace(
            backend=lambda _name: SimpleNamespace(enabled=True, failed=False))),
    )
    with mock.patch.dict(sys.modules, {"jittor": backend}), ExitStack() as stack:
        move = _select_device(SimpleNamespace(), "jittor", "npu", policy_stack=stack)
        value = _PlacedTensor()
        assert move(value) is value
        assert value.destination == "npu"
        assert changes == [{"use_cuda": 1, "use_acl": 1}]


def test_npu_guard_checks_each_tensor_backend_and_selected_index():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    from _ecosystem_runner import _assert_jittor_placement

    first, second = object(), object()
    torch = SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 1))
    for second_location, fails in ((("acl", 1), False), (("cpu", -1), True),
                                   (("acl", 0), True), (("npu", 1), True)):
        def dispatch(values):
            assert len(values) == 1
            return ("acl", 1) if values[0] is first else second_location

        backend = SimpleNamespace(core=SimpleNamespace(dispatch_context=dispatch))
        with mock.patch.dict(sys.modules, {"jittor": backend}):
            values = [("first", first), ("second", second)]
            if fails:
                with unittest.TestCase().assertRaisesRegex(RuntimeError, "second ran on"):
                    _assert_jittor_placement(torch, values, "parameter", "npu")
            else:
                _assert_jittor_placement(torch, values, "parameter", "npu")


def test_eval_restarts_only_the_original_trainable_parameters():
    class FreezingModel(_Model):
        def eval(self):
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            return self

    model = FreezingModel()
    expected = _parameter_grad_state(model)
    assert _evaluate_preserving_parameter_grads(model) == expected
    assert _parameter_grad_state(model) == expected


def test_trainer_runners_accept_npu_without_selecting_another_device():
    import sys
    from unittest import mock
    import _hf_trainer_e2e_runner as e2e
    import _hf_trainer_resume_runner as resume

    specifications = (
        (e2e, ["run", "--runtime", "jittor", "--device", "npu", "--output-dir", "unused",
               "--weights-out", "weights.npz", "--base-weights-out", "base.npz"]),
        (resume, ["run", "--runtime", "jittor", "--device", "npu", "--output-dir", "unused",
                  "--weights", "weights.npz"]),
    )
    for module, arguments in specifications:
        with mock.patch.object(module, "_run") as run:
            with mock.patch.object(sys, "argv", [module.__file__] + arguments):
                module.main()
            assert run.call_count == 1
            assert run.call_args.args[0].device == "npu"


def test_trainer_model_placement_checks_buffers_alongside_parameters():
    from types import SimpleNamespace
    from _hf_trainer_e2e_runner import _assert_model_device

    parameter = SimpleNamespace(device=SimpleNamespace(type="npu"))
    misplaced_buffer = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)],
                            named_buffers=lambda: [("position_ids", misplaced_buffer)])
    with unittest.TestCase().assertRaisesRegex(AssertionError, "position_ids"):
        _assert_model_device(None, None, model, "npu")


def test_transformers_contract_rejects_a_gradient_missing_from_both_runtimes():
    report = {
        "parameters": {"trainable": ["transformer.wte.weight"], "frozen": []},
        "parameter_grads": [],
    }
    with unittest.TestCase().assertRaisesRegex(AssertionError, "declared trainable-gradient contract"):
        _assert_declared_parameter_grads(unittest.TestCase(), "transformers_gpt2", report)


def test_encoder_contract_allows_exactly_the_unused_pooler_gradients():
    report = {
        "parameters": {"trainable": ["embeddings.weight", "pooler.dense.bias", "pooler.dense.weight"],
                       "frozen": []},
        "parameter_grads": ["embeddings.weight"],
    }
    _assert_declared_parameter_grads(unittest.TestCase(), "transformers_bert", report)
    report["parameter_grads"] = []
    with unittest.TestCase().assertRaisesRegex(AssertionError, "declared trainable-gradient contract"):
        _assert_declared_parameter_grads(unittest.TestCase(), "transformers_bert", report)


def test_cpu_trainer_bootstrap_hides_accelerators_before_import():
    import os
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    frontend = SimpleNamespace(_C=object(), npu=SimpleNamespace(
        device_count=lambda: 0, is_available=lambda: False, is_initialized=lambda: False))
    with mock.patch.dict(sys.modules, {"torch": frontend}):
        with mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}):
            jt, observed = runner._runtime("torch", device="cpu")
            assert jt is None and observed is frontend
            assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == ""
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_trainer_refuses_visible_or_initialized_npu_state():
    from types import SimpleNamespace
    import _hf_trainer_e2e_runner as runner

    for count, initialized in ((1, False), (0, True)):
        frontend = SimpleNamespace(npu=SimpleNamespace(
            device_count=lambda: count, is_available=lambda: count > 0,
            is_initialized=lambda: initialized))
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "CPU Trainer"):
            runner._cpu_runtime_evidence(frontend)


def test_cpu_trainer_rejects_native_npu_optimizer_side_effects():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    with mock.patch.dict(sys.modules, {"torch_npu": object()}):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "without torch_npu"):
            runner._cpu_runtime_evidence(SimpleNamespace())


def _npu_adapter_fixture():
    from types import ModuleType, SimpleNamespace
    from unittest import mock
    import importlib.metadata as metadata

    adapter = SimpleNamespace(patch_import_utils=lambda: None,
                              patch_flash_attention_utils=lambda: None,
                              patch_trainer=lambda: None,
                              patch_accelerate_imports=lambda: None,
                              patch_accelerate=lambda: None)
    registered = {
        "transformers.utils.import_utils": (adapter.patch_import_utils,),
        "transformers.modeling_flash_attention_utils": (adapter.patch_flash_attention_utils,),
        "transformers.trainer": (adapter.patch_trainer,),
        "accelerate.utils.imports": (adapter.patch_accelerate_imports,),
        "accelerate": (adapter.patch_accelerate,),
    }
    patcher = ModuleType("jittor.compat.module_patcher")
    patcher.registered_module_patches = lambda: registered
    package = ModuleType("jittor_adapters")
    package.transformers = adapter
    entries = [SimpleNamespace(name="jittor_transformers", value="jittor_adapters.transformers:register")]
    return metadata, mock, registered, entries, {
        "jittor.compat.module_patcher": patcher, "jittor_adapters": package,
    }


def test_npu_trainer_requires_installed_adapter_metadata_before_setup():
    import importlib.metadata as metadata
    from types import SimpleNamespace
    from unittest import mock
    import os
    import sys
    import _hf_trainer_e2e_runner as runner

    frontend = SimpleNamespace(_torch_compat_install_context=object())
    with mock.patch.dict(sys.modules, {"jittor": object(), "torch": frontend}):
        with mock.patch.dict(os.environ), mock.patch.object(
                metadata, "version", side_effect=metadata.PackageNotFoundError("jittor-torch-adapters")):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "source PYTHONPATH alone"):
                runner._runtime("jittor", device="npu")


def test_npu_trainer_accepts_only_the_registered_owned_adapter_callbacks():
    import sys
    import _hf_trainer_e2e_runner as runner

    metadata, mock, registered, entries, modules = _npu_adapter_fixture()
    with mock.patch.dict(sys.modules, modules), mock.patch.object(metadata, "version", return_value="2.0.0"):
        with mock.patch.object(metadata, "entry_points", return_value=entries):
            evidence = runner._require_npu_adapter_registration()
            assert evidence["registered_paths"] == sorted(registered)
            flash_path = "transformers.modeling_flash_attention_utils"
            flash_callback = registered.pop(flash_path)
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "not registered.*modeling_flash_attention_utils"):
                runner._require_npu_adapter_registration()
            registered[flash_path] = flash_callback
            trainer_path = "transformers.trainer"
            trainer_callback = registered.pop(trainer_path)
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "not registered.*transformers.trainer"):
                runner._require_npu_adapter_registration()
            registered[trainer_path] = trainer_callback
            registered["accelerate"] = (lambda: None,)
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "not registered.*accelerate"):
                runner._require_npu_adapter_registration()


def test_npu_trainer_rejects_missing_or_unexpected_adapter_entry_point():
    import sys
    import _hf_trainer_e2e_runner as runner

    metadata, mock, registered, entries, modules = _npu_adapter_fixture()
    with mock.patch.dict(sys.modules, modules), mock.patch.object(metadata, "version", return_value="2.0.0"):
        for values in ([], entries + entries):
            with mock.patch.object(metadata, "entry_points", return_value=values):
                with unittest.TestCase().assertRaisesRegex(RuntimeError, "owned jittor_transformers"):
                    runner._require_npu_adapter_registration()
        entries[0].value = "unrelated.adapter:register"
        with mock.patch.object(metadata, "entry_points", return_value=entries):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "owned jittor_transformers"):
                runner._require_npu_adapter_registration()


def test_npu_trainer_rejects_an_unverified_adapter_version():
    import importlib.metadata as metadata
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    with mock.patch.object(metadata, "version", return_value="99.0.0"):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "requires.*2.0.0"):
            runner._require_npu_adapter_registration()


def test_cpu_trainer_non_acl_shim_uses_build_evidence_without_initializing_npu():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    for backend in ("cpu", "cuda", "rocm", "corex"):
        npu = SimpleNamespace(device_count=mock.Mock(return_value=0),
                              is_available=mock.Mock(return_value=False),
                              is_initialized=mock.Mock(side_effect=NotImplementedError))
        native = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=0, build_config=SimpleNamespace(backend=backend, has_acl=False)),
            flags=SimpleNamespace(use_acl=0, use_cuda=0))
        frontend = SimpleNamespace(_torch_compat_install_context=object(), npu=npu)
        with mock.patch.dict(sys.modules, {"jittor": native}):
            observed = runner._cpu_runtime_evidence(frontend)
        assert observed["npu_initialized"] is False
        assert observed["npu_initialization_evidence"] == "non_acl_build"
        npu.is_initialized.assert_not_called()


def test_cpu_trainer_rejects_unproven_shim_before_any_npu_query():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    for backend, has_acl in (("acl", 1), ("acl", 0), ("cpu", 1), (None, 0),
                             ("unknown", 0), ("cpu", None)):
        npu = SimpleNamespace(
            device_count=mock.Mock(side_effect=AssertionError("must not initialize NPU")),
            is_available=mock.Mock(side_effect=AssertionError("must not initialize NPU")),
            is_initialized=mock.Mock(side_effect=NotImplementedError))
        native = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=has_acl, build_config=SimpleNamespace(backend=backend, has_acl=has_acl)),
            flags=SimpleNamespace(use_acl=0, use_cuda=0))
        frontend = SimpleNamespace(_torch_compat_install_context=object(), npu=npu)
        with mock.patch.dict(sys.modules, {"jittor": native}):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "CPU Trainer"):
                runner._cpu_runtime_evidence(frontend)
        npu.device_count.assert_not_called()
        npu.is_available.assert_not_called()
        npu.is_initialized.assert_not_called()


def test_cpu_trainer_rejects_active_shim_accelerator_flags_before_queries():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    for use_acl, use_cuda in ((1, 0), (0, 1)):
        npu = SimpleNamespace(
            device_count=mock.Mock(side_effect=AssertionError("must not query NPU")),
            is_available=mock.Mock(side_effect=AssertionError("must not query NPU")),
            is_initialized=mock.Mock(side_effect=NotImplementedError))
        native = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=0, build_config=SimpleNamespace(backend="cpu", has_acl=False)),
            flags=SimpleNamespace(use_acl=use_acl, use_cuda=use_cuda))
        frontend = SimpleNamespace(_torch_compat_install_context=object(), npu=npu)
        with mock.patch.dict(sys.modules, {"jittor": native}):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "CPU Trainer"):
                runner._cpu_runtime_evidence(frontend)
        npu.device_count.assert_not_called()
        npu.is_available.assert_not_called()
        npu.is_initialized.assert_not_called()


def test_cpu_trainer_native_still_queries_initialization_with_zero_devices():
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    initialized = mock.Mock(return_value=False)
    frontend = SimpleNamespace(_C=object(), npu=SimpleNamespace(
        device_count=lambda: 0, is_available=lambda: False, is_initialized=initialized))
    observed = runner._cpu_runtime_evidence(frontend)
    assert observed["npu_initialized"] is False
    initialized.assert_called_once_with()


def test_cpu_trainer_rejects_unproven_shim_with_conflicting_build_capability():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    for configured_acl in (True, None, "0"):
        npu = SimpleNamespace(device_count=mock.Mock(return_value=0),
                              is_available=mock.Mock(return_value=False),
                              is_initialized=mock.Mock(side_effect=NotImplementedError))
        native = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=0, build_config=SimpleNamespace(
                backend="cpu", has_acl=configured_acl)),
            flags=SimpleNamespace(use_acl=0, use_cuda=0))
        frontend = SimpleNamespace(_torch_compat_install_context=object(), npu=npu)
        with mock.patch.dict(sys.modules, {"jittor": native}):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "CPU Trainer"):
                runner._cpu_runtime_evidence(frontend)
        npu.device_count.assert_not_called()
        npu.is_available.assert_not_called()
        npu.is_initialized.assert_not_called()


def test_cpu_trainer_rejects_unproven_shim_with_missing_execution_flags():
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import _hf_trainer_e2e_runner as runner

    for flags in (SimpleNamespace(), SimpleNamespace(use_acl=0),
                  SimpleNamespace(use_cuda=0)):
        npu = SimpleNamespace(device_count=mock.Mock(return_value=0),
                              is_available=mock.Mock(return_value=False),
                              is_initialized=mock.Mock(side_effect=NotImplementedError))
        native = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=0, build_config=SimpleNamespace(
                backend="cpu", has_acl=False)), flags=flags)
        frontend = SimpleNamespace(_torch_compat_install_context=object(), npu=npu)
        with mock.patch.dict(sys.modules, {"jittor": native}):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "CPU Trainer"):
                runner._cpu_runtime_evidence(frontend)
        npu.device_count.assert_not_called()
        npu.is_available.assert_not_called()
        npu.is_initialized.assert_not_called()
