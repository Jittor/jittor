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
