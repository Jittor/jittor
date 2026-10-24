"""Public LambdaLR checkpoint/continuation contract, runnable on either Torch.

For host development only, ``--source-only`` executes the actual candidate
scheduler definitions with independent native Torch SGD/serialization. That
mode imports no Jittor and proves no candidate tensor/serialization behavior.
Run without that flag under each runtime for the full CPU checkpoint contract.
"""
import ast
import copy
import io
from pathlib import Path
import sys
import types
import unittest

import torch


SOURCE_ONLY = "--source-only" in sys.argv
Scheduler = None


def load_scheduler():
    if not SOURCE_ONLY:
        return torch.optim.lr_scheduler.LambdaLR
    if hasattr(torch, "_torch_compat_install_context") or "jittor" in str(torch.__file__):
        raise RuntimeError("source-only requires an independent native Torch interpreter")
    path = Path(__file__).resolve().parents[3] / "compat/torch/lr_scheduler.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in {"_base_lrs", "_set_lrs", "LRScheduler", "LambdaLR"}]
    namespace = {"_types": types}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["LambdaLR"]


class Scale:
    def __init__(self, factor=1.0, offset=1):
        self.factor, self.offset = factor, offset

    def __call__(self, epoch):
        return self.factor / (epoch + self.offset)


def make_optimizer(initial=(2.0, -1.0)):
    parameters = [torch.nn.Parameter(torch.tensor([value], dtype=torch.float32, device="cpu"))
                  for value in initial]
    optimizer = torch.optim.SGD([{"params": [parameters[0]], "lr": .5},
                                 {"params": [parameters[1]], "lr": .25}], lr=.5)
    return parameters, optimizer


def functions():
    # Deliberately local functions: serializing these callables is invalid.
    def warmup(epoch):
        return min((epoch + 1) / 4, 1.0)
    return [warmup, lambda epoch: 1 / (epoch + 1)]


def advance(parameters, optimizer, scheduler):
    rates_used = [float(group["lr"]) for group in optimizer.param_groups]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    return rates_used, scheduler.get_last_lr(), [float(p.item()) for p in parameters]


class TestLambdaLRCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global Scheduler
        Scheduler = load_scheduler()

    def test_functions_and_lambdas_are_not_serialized(self):
        _, optimizer = make_optimizer()
        callbacks = functions()
        callbacks[0].extra_attribute = "functions still have no serialized state"
        scheduler = Scheduler(optimizer, callbacks)
        state = scheduler.state_dict()
        self.assertEqual(state["lr_lambdas"], [None, None])
        self.assertNotIn("optimizer", state)
        self.assertIs(scheduler.lr_lambdas[0], callbacks[0])
        self.assertIs(scheduler.lr_lambdas[1], callbacks[1])

    def test_loading_function_placeholders_retains_new_constructor_functions(self):
        parameters, optimizer = make_optimizer()
        callbacks = functions()
        scheduler = Scheduler(optimizer, callbacks)
        state = scheduler.state_dict()
        state["last_epoch"] = 3
        # This is the actual native-Torch function state format, also what the
        # portable serializer produced from the previous incorrect function list.
        state["lr_lambdas"] = [None, None]
        before = copy.deepcopy(state)
        scheduler.load_state_dict(state)
        self.assertEqual(state, before)
        self.assertIs(scheduler.lr_lambdas[0], callbacks[0])
        self.assertIs(scheduler.lr_lambdas[1], callbacks[1])
        advance(parameters, optimizer, scheduler)
        self.assertEqual(scheduler.get_last_lr(), [.5, .05])

    def test_callable_object_attributes_are_copied_not_callable_objects(self):
        _, optimizer = make_optimizer()
        source = Scale(2, 3)
        scheduler = Scheduler(optimizer, [source, lambda epoch: 1])
        state = scheduler.state_dict()
        self.assertEqual(state["lr_lambdas"], [{"factor": 2, "offset": 3}, None])
        self.assertIsNot(state["lr_lambdas"][0], source.__dict__)
        source.factor = 9
        self.assertEqual(state["lr_lambdas"][0]["factor"], 2)

    def test_callable_state_restore_preserves_identity_and_input_reusability(self):
        _, optimizer = make_optimizer()
        source = Scheduler(optimizer, [Scale(2, 3), lambda epoch: 1])
        state = source.state_dict()
        # Explicit expected format keeps this a negative control even if
        # state_dict and load_state_dict are both incorrectly implemented.
        state["lr_lambdas"] = [{"factor": 2, "offset": 3}, None]
        before = copy.deepcopy(state)
        for _ in range(2):
            parameters, target_optimizer = make_optimizer()
            callback = Scale(9, 7)
            retained = lambda epoch: .5
            target = Scheduler(target_optimizer, [callback, retained])
            target.load_state_dict(state)
            self.assertIs(target.lr_lambdas[0], callback)
            self.assertIs(target.lr_lambdas[1], retained)
            self.assertEqual(callback.__dict__, {"factor": 2, "offset": 3})
            self.assertEqual(state, before)
            advance(parameters, target_optimizer, target)
            self.assertEqual(target.get_last_lr(), [.25, .125])

    def test_checkpoint_resume_matches_uninterrupted_lr_and_weight_trajectory(self):
        for callable_object in (False, True):
            with self.subTest(callable_object=callable_object):
                params, optimizer = make_optimizer()
                callbacks = [Scale(2, 3), functions()[1]] if callable_object else functions()
                scheduler = Scheduler(optimizer, callbacks)
                for _ in range(3):
                    advance(params, optimizer, scheduler)
                checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                              "parameters": [p.detach().clone() for p in params]}
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                expected = [advance(params, optimizer, scheduler) for _ in range(4)]
                buffer.seek(0)
                restored = torch.load(buffer, map_location="cpu", weights_only=False)
                new_params, new_optimizer = make_optimizer([p.item() for p in restored["parameters"]])
                new_callbacks = [Scale(99, 19), functions()[1]] if callable_object else functions()
                new_scheduler = Scheduler(new_optimizer, new_callbacks)
                new_scheduler.load_state_dict(restored["scheduler"])
                new_optimizer.load_state_dict(restored["optimizer"])
                self.assertIs(new_scheduler.lr_lambdas[0], new_callbacks[0])
                self.assertIs(new_scheduler.lr_lambdas[1], new_callbacks[1])
                actual = [advance(new_params, new_optimizer, new_scheduler) for _ in range(4)]
                self.assertEqual(actual, expected)
                self.assertEqual(new_scheduler.last_epoch, scheduler.last_epoch)
                self.assertEqual(new_scheduler._step_count, scheduler._step_count)

    def test_unknown_user_scheduler_attribute_is_retained(self):
        _, optimizer = make_optimizer()
        scheduler = Scheduler(optimizer, functions())
        scheduler.note = {"epoch_marker": 7}
        state = scheduler.state_dict()
        _, new_optimizer = make_optimizer()
        restored = Scheduler(new_optimizer, functions())
        restored.load_state_dict(state)
        self.assertEqual(restored.note, {"epoch_marker": 7})


if __name__ == "__main__":
    if SOURCE_ONLY:
        sys.argv.remove("--source-only")
    unittest.main()
