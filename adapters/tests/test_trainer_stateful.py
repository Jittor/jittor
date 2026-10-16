"""Host contracts for the explicit Trainer stateful-checkpoint integration.

These tests use controlled packages, never the native runtime or a JIT cache.
The actual Trainer/Accelerate trajectory belongs to the separate acceptance run.
"""
import ast
import importlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adapters"))
from jittor_adapters import trainer_stateful
from jittor_adapters._common import UnsupportedAdapterVersion


class _PreparedLoader:
    """Its public snapshot excludes the base loader's unconsumed lookahead."""

    use_stateful_dataloader = True

    def __init__(self, batches=8):
        self.batches = batches
        self.loads = []
        self.snapshots = 0
        self.base_dataloader = types.SimpleNamespace(
            state_dict=mock.Mock(side_effect=AssertionError("unconsumed lookahead")))

    def __len__(self):
        return self.batches

    def state_dict(self):
        self.snapshots += 1
        return {"cursor": 2, "worker_rng": ["worker-0", "worker-1"]}

    def load_state_dict(self, state):
        self.loads.append(state)


class StatefulTrainerContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.last_checkpoint = None
        self.load_calls = []
        self.base_train_calls = []
        test = self

        class FakeTrainer:
            def __init__(self, *, args=None, callbacks=None, train_dataset=None, loader=None):
                self.args = args or test.arguments()
                self.callbacks = callbacks or []
                self.train_dataset = train_dataset if train_dataset is not None else list(range(16))
                self.loader = loader if loader is not None else _PreparedLoader()
                self.accelerator = types.SimpleNamespace(
                    num_processes=1,
                    dataloader_config=types.SimpleNamespace(use_stateful_dataloader=False),
                    prepare=mock.Mock(return_value=self.loader))

            def get_train_dataloader(self):
                return self.loader

            def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
                test.base_train_calls.append((resume_from_checkpoint, trial, ignore_keys_for_eval, kwargs))
                loader = self.get_train_dataloader()
                if resume_from_checkpoint:
                    data = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())
                else:
                    data = {"global_step": 0, "epoch": 0.0}
                state = types.SimpleNamespace(**data)
                control = object()
                for callback in self.callbacks:
                    callback.on_train_begin(self.args, state, control, train_dataloader=loader)
                return "trained"

        class FakeCallback:
            def on_train_begin(self, *args, **kwargs):
                pass

        def save(value, path):
            with open(path, "wb") as handle:
                pickle.dump(value, handle)

        def load(path, **kwargs):
            test.load_calls.append((path, kwargs))
            with open(path, "rb") as handle:
                return pickle.load(handle)

        self.transformers = types.ModuleType("transformers")
        self.transformers.__version__ = "4.56.2"
        self.transformers.Trainer = FakeTrainer
        self.transformers.TrainerCallback = FakeCallback
        self.accelerate = types.ModuleType("accelerate")
        self.accelerate.__version__ = "1.10.1"
        self.torch = types.ModuleType("torch")
        self.torch.save = save
        self.torch.load = load
        utils = types.ModuleType("transformers.trainer_utils")
        utils.get_last_checkpoint = lambda output: self.last_checkpoint
        self.modules = mock.patch.dict(sys.modules, {
            "transformers": self.transformers,
            "transformers.trainer_utils": utils,
            "accelerate": self.accelerate,
            "torch": self.torch,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.Trainer = FakeTrainer

    def arguments(self, **changes):
        values = dict(
            output_dir=str(self.output), world_size=1, n_gpu=0,
            auto_find_batch_size=False, save_only_model=False,
            max_steps=4, gradient_accumulation_steps=1,
            per_device_train_batch_size=2, dataloader_num_workers=2,
            dataloader_prefetch_factor=2, dataloader_persistent_workers=False,
            dataloader_drop_last=False, seed=123, data_seed=456,
            ignore_data_skip=False,
        )
        values.update(changes)
        return types.SimpleNamespace(**values)

    def trainer(self, **kwargs):
        return trainer_stateful.stateful_trainer_class()(
            stateful_dataset_id="fixed-fixture-v1", **kwargs)

    def checkpoint(self, trainer, **changes):
        path = self.output / "checkpoint-2"
        path.mkdir(exist_ok=True)
        payload = {
            "version": 1,
            "versions": {"transformers": "4.56.2", "accelerate": "1.10.1"},
            "global_step": 2,
            "epoch": 0.25,
            "configuration": trainer.stateful_configuration(trainer.loader),
            "dataloader_state": {"cursor": 2, "worker_rng": ["a", "b"]},
            "generator_states": {},
        }
        payload.update(changes)
        self.torch.save(payload, path / trainer_stateful.STATE_FILE)
        (path / "trainer_state.json").write_text(json.dumps({
            "global_step": 2, "epoch": 0.25,
        }))
        return path, payload

    def test_import_has_no_downstream_import_or_registration_side_effect(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "accelerate", "transformers"):
                raise AssertionError("adapter import must not import its optional dependency")
            return original(name, *args, **kwargs)

        finders = list(sys.meta_path)
        with mock.patch.object(builtins, "__import__", guarded):
            importlib.reload(trainer_stateful)
        self.assertEqual(sys.meta_path, finders)

    def test_opt_in_preserves_trainer_and_framework_modules(self):
        trainer_class = dict(vars(self.Trainer))
        torch_names = dict(vars(self.torch))
        transformers_names = dict(vars(self.transformers))
        default = self.Trainer()
        adapted = self.trainer()
        self.assertFalse(default.accelerator.dataloader_config.use_stateful_dataloader)
        self.assertFalse(default.args.ignore_data_skip)
        self.assertTrue(adapted.accelerator.dataloader_config.use_stateful_dataloader)
        self.assertTrue(adapted.args.ignore_data_skip)
        self.assertEqual(dict(vars(self.Trainer)), trainer_class)
        self.assertEqual(dict(vars(self.torch)), torch_names)
        self.assertEqual(dict(vars(self.transformers)), transformers_names)

    def test_custom_trainer_is_extended_without_replacing_it(self):
        class RecordingTrainer(self.Trainer):
            def get_train_dataloader(self):
                self.recorded = True
                return super().get_train_dataloader()

        adapted = trainer_stateful.stateful_trainer_class(RecordingTrainer)(
            stateful_dataset_id="fixture")
        self.assertIsInstance(adapted, RecordingTrainer)
        self.assertIs(adapted.get_train_dataloader(), adapted.loader)
        self.assertTrue(adapted.recorded)
        with self.assertRaisesRegex(TypeError, "Trainer subclass"):
            trainer_stateful.stateful_trainer_class(object)

    def test_unverified_downstream_versions_fail_closed(self):
        for package in (self.transformers, self.accelerate):
            with self.subTest(package=package.__name__):
                original = package.__version__
                package.__version__ = "99.0.0"
                with self.assertRaises(UnsupportedAdapterVersion):
                    trainer_stateful.stateful_trainer_class()
                package.__version__ = original

    def test_dataset_identity_is_required(self):
        cls = trainer_stateful.stateful_trainer_class()
        for identity in (None, "", "  ", 123):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "stateful_dataset_id"):
                cls(stateful_dataset_id=identity)

    def test_factory_receives_owner_and_its_raw_loader_is_prepared_publicly(self):
        raw_loader = object()
        factory = mock.Mock(return_value=raw_loader)
        trainer = self.trainer(stateful_dataloader_factory=factory)
        prepared_loader = _PreparedLoader()
        trainer.accelerator.prepare.return_value = prepared_loader
        self.assertIs(trainer.get_train_dataloader(), prepared_loader)
        factory.assert_called_once_with(trainer)
        trainer.accelerator.prepare.assert_called_once_with(raw_loader)
        self.assertTrue(trainer.accelerator.dataloader_config.use_stateful_dataloader)

    def test_absent_factory_preserves_the_base_trainer_loader(self):
        trainer = self.trainer()
        self.assertIs(trainer.get_train_dataloader(), trainer.loader)
        trainer.accelerator.prepare.assert_not_called()

    def test_factory_must_be_callable(self):
        for factory in (0, "loader", object()):
            with self.subTest(factory=factory), self.assertRaisesRegex(TypeError, "factory must be callable"):
                self.trainer(stateful_dataloader_factory=factory)

    def test_factory_errors_are_not_hidden_or_replaced_by_the_default_loader(self):
        factory = mock.Mock(side_effect=RuntimeError("factory failed"))
        trainer = self.trainer(stateful_dataloader_factory=factory)
        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            trainer.get_train_dataloader()
        trainer.accelerator.prepare.assert_not_called()

    def test_factory_preparation_must_produce_a_stateful_loader(self):
        factory = mock.Mock(return_value=object())
        trainer = self.trainer(stateful_dataloader_factory=factory)
        trainer.accelerator.prepare.return_value.use_stateful_dataloader = False
        with self.assertRaisesRegex(ValueError, "did not prepare a stateful"):
            trainer.get_train_dataloader()

    def test_missing_state_never_enters_the_base_training_loop(self):
        path = self.output / "checkpoint-2"
        path.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, trainer_stateful.STATE_FILE):
            self.trainer().train(resume_from_checkpoint=str(path))
        self.assertEqual(self.base_train_calls, [])

    def test_checkpoint_schema_is_exact(self):
        trainer = self.trainer()
        path, payload = self.checkpoint(trainer)
        for invalid in ([], {**payload, "unknown": 1}, {k: v for k, v in payload.items() if k != "epoch"}):
            with self.subTest(payload=invalid):
                self.torch.save(invalid, path / trainer_stateful.STATE_FILE)
                with self.assertRaisesRegex(ValueError, "schema"):
                    trainer.train(resume_from_checkpoint=str(path))
        self.assertEqual(self.base_train_calls, [])

    def test_state_format_and_library_versions_are_checked(self):
        for changes in ({"version": 2}, {"versions": {"transformers": "5.5.3", "accelerate": "1.10.1"}}):
            with self.subTest(changes=changes):
                trainer = self.trainer()
                path, unused = self.checkpoint(trainer, **changes)
                with self.assertRaisesRegex(ValueError, "checkpoint version"):
                    trainer.train(resume_from_checkpoint=str(path))
        self.assertEqual(self.base_train_calls, [])

    def test_checkpoint_step_must_be_a_positive_integer(self):
        for step in (-1, 0, 1.5, True, "2", None):
            with self.subTest(step=step):
                trainer = self.trainer()
                path, unused = self.checkpoint(trainer, global_step=step)
                # Matching malformed metadata must still fail the state contract.
                (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": 0.25}))
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    trainer.train(resume_from_checkpoint=str(path))
                self.assertEqual(trainer.loader.loads, [])
        self.assertEqual(self.base_train_calls, [])

    def test_checkpoint_generator_states_must_be_a_dictionary(self):
        for states in (None, [], ["loader"], "loader"):
            with self.subTest(states=states):
                trainer = self.trainer()
                path, unused = self.checkpoint(trainer, generator_states=states)
                with self.assertRaisesRegex(ValueError, "generator states"):
                    trainer.train(resume_from_checkpoint=str(path))
                self.assertEqual(trainer.loader.loads, [])
        self.assertEqual(self.base_train_calls, [])

    def test_trainer_step_and_epoch_must_agree_with_loader_checkpoint(self):
        for changes in ({"global_step": 3}, {"epoch": 0.5}):
            with self.subTest(changes=changes):
                trainer = self.trainer()
                path, unused = self.checkpoint(trainer, **changes)
                with self.assertRaisesRegex(ValueError, "checkpoints disagree"):
                    trainer.train(resume_from_checkpoint=str(path))
        self.assertEqual(self.base_train_calls, [])

    def test_resume_rejects_a_completed_or_invalid_epoch(self):
        for epoch in (0.0, 1.0, 1.25):
            with self.subTest(epoch=epoch):
                trainer = self.trainer()
                path, unused = self.checkpoint(trainer, epoch=epoch)
                (path / "trainer_state.json").write_text(json.dumps({"global_step": 2, "epoch": epoch}))
                with self.assertRaisesRegex(ValueError, "unfinished first epoch"):
                    trainer.train(resume_from_checkpoint=str(path))
        self.assertEqual(self.base_train_calls, [])

    def test_configuration_changes_fail_before_restoring_worker_state(self):
        for key, value in (("dataset_id", "different-transform"), ("num_workers", 0),
                           ("batch_size", 4), ("max_steps", 3), ("data_seed", 1234)):
            with self.subTest(key=key):
                trainer = self.trainer()
                path, payload = self.checkpoint(trainer)
                payload["configuration"][key] = value
                self.torch.save(payload, path / trainer_stateful.STATE_FILE)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    trainer.train(resume_from_checkpoint=str(path))
                self.assertEqual(trainer.loader.loads, [])

    def test_resume_restores_the_public_prepared_loader_at_train_begin(self):
        trainer = self.trainer()
        path, payload = self.checkpoint(trainer)
        trainer.args.ignore_data_skip = False
        self.assertEqual(trainer.train(resume_from_checkpoint=str(path), extra="forwarded"), "trained")
        self.assertTrue(trainer.args.ignore_data_skip)
        self.assertEqual(trainer.loader.loads, [payload["dataloader_state"]])
        self.assertEqual(trainer.stateful_restored_step, 2)
        self.assertIsNone(trainer.stateful_pending_checkpoint)
        self.assertIs(trainer.stateful_loader, trainer.loader)
        self.assertEqual(self.load_calls[-1][1], {"map_location": "cpu", "weights_only": False})
        self.assertEqual(self.base_train_calls[-1][-1], {"extra": "forwarded"})
        trainer.loader.base_dataloader.state_dict.assert_not_called()

    def test_train_begin_rechecks_actual_trainer_step(self):
        trainer = self.trainer()
        unused, payload = self.checkpoint(trainer)
        trainer.stateful_pending_checkpoint = payload
        with self.assertRaisesRegex(ValueError, "global_step differs"):
            trainer.callbacks[0].on_train_begin(
                trainer.args, types.SimpleNamespace(global_step=3), object(), train_dataloader=trainer.loader)
        self.assertEqual(trainer.loader.loads, [])

    def test_epoch_must_match_consumed_accumulation_windows(self):
        for accumulation, epoch in ((1, 0.5), (2, 0.25)):
            with self.subTest(accumulation=accumulation, epoch=epoch):
                trainer = self.trainer(args=self.arguments(gradient_accumulation_steps=accumulation))
                path, unused = self.checkpoint(trainer, epoch=epoch)
                (path / "trainer_state.json").write_text(json.dumps({"global_step": 2, "epoch": epoch}))
                with self.assertRaisesRegex(ValueError, "does not match consumed batches"):
                    trainer.train(resume_from_checkpoint=str(path))
                self.assertEqual(trainer.loader.loads, [])

    def test_loader_and_nested_sampler_generators_are_saved_and_restored(self):
        class Generator:
            def __init__(self, value):
                self.value = value

            def get_state(self):
                return self.value

            def set_state(self, value):
                self.value = value

        trainer = self.trainer()
        trainer.loader.generator = Generator("loader-state")
        sampler = types.SimpleNamespace(generator=Generator("sampler-state"))
        trainer.loader.batch_sampler = types.SimpleNamespace(sampler=sampler)
        callback = trainer.callbacks[0]
        state = types.SimpleNamespace(global_step=2, epoch=0.25)
        callback.on_train_begin(trainer.args, state, object(), train_dataloader=trainer.loader)
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        callback.on_save(trainer.args, state, object())
        payload = self.torch.load(checkpoint / trainer_stateful.STATE_FILE)
        self.assertEqual(payload["generator_states"], {"loader": "loader-state", "sampler": "sampler-state"})
        trainer.loader.generator.value = "advanced-loader"
        sampler.generator.value = "advanced-sampler"
        trainer.stateful_pending_checkpoint = payload
        callback.on_train_begin(trainer.args, state, object(), train_dataloader=trainer.loader)
        self.assertEqual(trainer.loader.generator.value, "loader-state")
        self.assertEqual(sampler.generator.value, "sampler-state")

    def test_generator_configuration_change_fails_before_restoring_loader(self):
        trainer = self.trainer()
        path, unused = self.checkpoint(trainer, generator_states={"loader": "not-present"})
        with self.assertRaisesRegex(ValueError, "generators differ"):
            trainer.train(resume_from_checkpoint=str(path))
        self.assertEqual(trainer.loader.loads, [])

    def test_save_uses_consumed_boundary_and_corrects_accumulated_epoch(self):
        observer = object()
        trainer = self.trainer(args=self.arguments(gradient_accumulation_steps=2), callbacks=[observer])
        self.assertIs(trainer.callbacks[1], observer)
        callback = trainer.callbacks[0]
        control = object()
        state = types.SimpleNamespace(global_step=2, epoch=0.25)
        callback.on_train_begin(trainer.args, state, control, train_dataloader=trainer.loader)
        self.assertIs(callback.on_step_end(trainer.args, state, control), control)
        self.assertEqual(state.epoch, 0.5)
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        self.assertIs(callback.on_save(trainer.args, state, control), control)
        state_file = checkpoint / trainer_stateful.STATE_FILE
        self.assertTrue(state_file.is_file())
        self.assertFalse(state_file.with_suffix(".pt.tmp").exists())
        saved = self.torch.load(state_file)
        self.assertEqual(saved["dataloader_state"]["cursor"], 2)
        self.assertEqual(saved["epoch"], 0.5)
        self.assertEqual(trainer.loader.snapshots, 1)
        self.assertEqual(trainer.stateful_saved_checkpoints, [str(state_file)])
        trainer.loader.base_dataloader.state_dict.assert_not_called()

    def test_resume_true_resolves_the_last_checkpoint(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "no checkpoint found"):
            trainer.train(resume_from_checkpoint=True)
        path, unused = self.checkpoint(trainer)
        self.last_checkpoint = str(path)
        self.assertEqual(trainer.train(resume_from_checkpoint=True), "trained")
        self.assertEqual(self.base_train_calls[-1][0], str(path))

    def test_hyperparameter_trials_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "hyperparameter-search"):
            self.trainer().train(trial=object())
        self.assertEqual(self.base_train_calls, [])

    def test_distributed_and_multiple_device_schedules_fail_closed(self):
        for field in ("world_size", "n_gpu", "num_processes"):
            with self.subTest(field=field):
                trainer = self.trainer()
                target = trainer.accelerator if field == "num_processes" else trainer.args
                setattr(target, field, 2)
                with self.assertRaisesRegex(ValueError, "single process and device"):
                    trainer.get_train_dataloader()

    def test_schedule_requires_fixed_batches_full_checkpoint_and_positive_budget(self):
        for changes in ({"auto_find_batch_size": True}, {"save_only_model": True}, {"max_steps": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.trainer(args=self.arguments(**changes)).get_train_dataloader()

    def test_empty_tail_and_later_epoch_are_rejected_before_training(self):
        schedules = ((0, 1, 1), (7, 2, 1), (8, 1, 9), (8, 2, 5))
        for batches, accumulation, max_steps in schedules:
            with self.subTest(schedule=(batches, accumulation, max_steps)), self.assertRaises(ValueError):
                self.trainer(loader=_PreparedLoader(batches), args=self.arguments(
                    gradient_accumulation_steps=accumulation, max_steps=max_steps)).get_train_dataloader()
        for accumulation, max_steps in ((1, 8), (2, 4)):
            trainer = self.trainer(args=self.arguments(
                gradient_accumulation_steps=accumulation, max_steps=max_steps))
            self.assertIs(trainer.get_train_dataloader(), trainer.loader)

    def test_loader_requires_both_public_state_operations(self):
        for operation in ("state_dict", "load_state_dict"):
            with self.subTest(operation=operation):
                loader = _PreparedLoader()
                setattr(loader, operation, None)
                with self.assertRaisesRegex(TypeError, "state_dict/load_state_dict"):
                    self.trainer(loader=loader).get_train_dataloader()
        loader = _PreparedLoader()
        loader.use_stateful_dataloader = False
        with self.assertRaisesRegex(ValueError, "did not prepare a stateful"):
            self.trainer(loader=loader).get_train_dataloader()


_PUBLIC_IMPORTS = {
    "json", "os", "pathlib", "pathlib.Path", "torch", "accelerate", "transformers",
    "transformers.Trainer", "transformers.TrainerCallback",
    "transformers.trainer_utils.get_last_checkpoint",
    "._common.require_version",
}


def _relocatable_violations(source):
    """Check the same public-only boundary required of independent adapters.

    Imports bind aliases, so ``import torch as backend`` cannot bypass checks.
    Ordinary configuration-object mutation remains allowed.
    """
    tree = ast.parse(source)
    violations = []
    framework_names = {"jt", "jittor", "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _PUBLIC_IMPORTS:
                    violations.append((node.lineno, "import", alias.name))
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name.split(".")[0] in ("torch", "jittor"):
                    framework_names.add(bound)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = "." * node.level + (node.module or "") + "." + alias.name
                if name not in _PUBLIC_IMPORTS:
                    violations.append((node.lineno, "import", name))

    def root(node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and not node.attr.startswith("__"):
                violations.append((node.lineno, "private", node.attr))
            if isinstance(node.ctx, (ast.Store, ast.Del)) and root(node) in framework_names:
                violations.append((node.lineno, "framework-write", node.attr))
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
                and root(node) in framework_names):
            violations.append((node.lineno, "framework-write", "subscript"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            if node.func.id in ("setattr", "delattr") and root(node.args[0]) in framework_names:
                violations.append((node.lineno, "framework-write", node.func.id))
            if (node.func.id == "getattr"
                    and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str) and node.args[1].value.startswith("_")
                    and not node.args[1].value.startswith("__")):
                violations.append((node.lineno, "private", node.args[1].value))
    return violations


def _adapter_sources():
    entrypoint = Path(trainer_stateful.__file__).resolve()
    package = entrypoint.parent if entrypoint.name == "__init__.py" else entrypoint.parent / entrypoint.stem
    source_root = package.parent
    # rglob also catches new helper subpackages while this remains a module.
    return sorted(path for path in source_root.rglob("*.py")
                  if (path == entrypoint or package in path.parents)
                  and "tests" not in path.relative_to(source_root).parts)


class StatefulTrainerRelocatableContracts(unittest.TestCase):
    def test_adapter_only_uses_public_imports_and_never_mutates_framework_state(self):
        self.assertTrue(_adapter_sources())
        for path in _adapter_sources():
            with self.subTest(path=path.name):
                self.assertEqual(_relocatable_violations(path.read_text()), [])

    def test_recursive_scan_includes_helper_subpackages_and_excludes_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entrypoint = root / "trainer_stateful.py"
            helper = root / "trainer_stateful" / "nested" / "helper.py"
            excluded = root / "trainer_stateful" / "tests" / "test_fake.py"
            helper.parent.mkdir(parents=True)
            excluded.parent.mkdir(parents=True)
            for path in (entrypoint, helper, excluded):
                path.write_text("")
            with mock.patch.object(trainer_stateful, "__file__", str(entrypoint)):
                self.assertEqual(_adapter_sources(), [helper, entrypoint])

    def test_the_import_rule_rejects_native_and_private_imports(self):
        for source in ("import jittor", "from jittor.compat.torch import install",
                       "from torch._C import implementation", "from transformers.trainer import Trainer"):
            with self.subTest(source=source):
                self.assertTrue(_relocatable_violations(source))

    def test_the_private_rule_rejects_aliases_and_dynamic_access(self):
        for source in ("import torch as backend\nbackend._private()",
                       "import transformers as hf\nhf.Trainer._save_checkpoint()",
                       "self.accelerator._private()",
                       "import accelerate as acc\ngetattr(acc, '_private')"):
            with self.subTest(source=source):
                self.assertTrue(_relocatable_violations(source))

    def test_framework_mutations_cannot_hide_behind_aliases_or_call_spelling(self):
        for source in ("import torch\ntorch.__version__ = 'fake'",
                       "import torch as backend\nbackend.nn.forward += 1",
                       "import torch as backend\ndel backend.nn.forward",
                       "import torch as backend\nsetattr(backend, 'x', 1)",
                       "import torch as backend\ndelattr(backend.nn, 'x')"):
            with self.subTest(source=source):
                self.assertTrue(_relocatable_violations(source))
        self.assertTrue(_relocatable_violations("import torch\ntorch.__dict__['x'] = 1"))
        self.assertEqual(_relocatable_violations(
            "import torch\nvalue = torch.save\nconfiguration = object()\n"
            "configuration.enabled = True\nsetattr(configuration, 'limit', 1)"), [])


if __name__ == "__main__":
    unittest.main()
