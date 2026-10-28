"""Training-denominator drift contracts using source-only upstream fixtures."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"
sys.path.insert(0, str(SCRIPTS))
import registry_inventory as registry  # noqa: E402


class TrainingRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("remote", "add", "origin", "https://example.invalid/upstream.git")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.git("add", relative)
        self.git("commit", "-qm", "source fixture")

    def swift(self):
        self.write("swift/model/registry.py", 'import unavailable_tensor_runtime\n'
                   'class ModelType:\n    DENSE = "dense"\n'
                   'class ModelMeta:\n    task_type: str = "metadata only"\n'
                   'register_model(ModelMeta(ModelType.DENSE))\n')
        self.write("swift/tuners/mapping.py", 'SWIFT_MAPPING = {"LORA": Adapter}\n')
        self.write("swift/trainers/factory.py", 'class TrainerFactory:\n'
                   '    TRAINER_MAPPING = {"sft": Trainer, "ppo": Trainer}\n')
        self.write("swift/megatron/pipelines/train/rlhf.py", 'trainer_mapping = {"gkd": Trainer}\n')
        self.write("swift/cli/main.py", 'ROUTE_MAPPING: dict = {"sft": "train", "eval": "eval"}\n')
        self.write("swift/arguments/options.py", 'class Arguments:\n'
                   '    task_type: Literal["causal_lm", "seq_cls"] = None\n'
                   '    rlhf_type: Literal["ppo", "simpo"] = "ppo"\n')

    def verl(self):
        self.write("verl/trainer/ppo/core_algos.py", 'import unavailable_tensor_runtime\n'
                   'class AdvantageEstimator:\n    GAE = "gae"\n'
                   '@register_adv_est(AdvantageEstimator.GAE)\ndef advantage(): pass\n'
                   '@register_policy_loss("clip")\ndef loss(): pass\n')
        self.write("verl/trainer/main.py", '@hydra.main(config_name="ppo_trainer")\n'
                   'def train(): pass\n'
                   '@hydra.main(config_name="evaluation")\ndef evaluate(): pass\n')
        self.write("verl/trainer/distillation/losses.py",
                   '@register_distillation_loss(Settings(names=["forward_kl", "reverse_kl"]))\n'
                   'def distill(): pass\n')

    def axis(self, result, name):
        return [row for row in result["axes"] if row["axis"] == name]

    def test_swift_keeps_training_methods_separate_from_tuners_and_routes(self):
        self.swift()
        result = registry.inventory("ms-swift", self.root)
        self.assertEqual(self.axis(result, "method")[0]["members"], ["LORA"])
        self.assertEqual(self.axis(result, "rlhf_method")[0]["members"], ["ppo", "simpo"])
        self.assertEqual(self.axis(result, "trainer")[0]["count"], 2)
        self.assertEqual(self.axis(result, "megatron_trainer")[0]["count"], 1)
        self.assertEqual(self.axis(result, "entrypoint")[0]["members"], ["eval", "sft"])
        self.assertEqual(self.axis(result, "task")[0]["count"], 2)
        for source in result["sources"]:
            self.assertEqual(source["sha256"], hashlib.sha256((self.root / source["path"]).read_bytes()).hexdigest())

    def test_new_rlhf_alias_changes_its_denominator_without_inventing_a_trainer(self):
        self.swift()
        first = registry.inventory("ms-swift", self.root)
        self.write("swift/arguments/options.py", 'class Arguments:\n'
                   '    task_type: Literal["causal_lm", "seq_cls"] = None\n'
                   '    rlhf_type: Literal["ppo", "simpo", "new_alias"] = "ppo"\n')
        second = registry.inventory("ms-swift", self.root)
        self.assertEqual(self.axis(second, "rlhf_method")[0]["count"], 3)
        self.assertEqual(self.axis(first, "trainer"), self.axis(second, "trainer"))
        self.assertNotEqual(first["upstream"]["commit"], second["upstream"]["commit"])

    def test_swift_installed_source_inventory_includes_nonregistry_training_dependencies(self):
        self.swift()
        self.write("swift/dataset/prepare.py", "import unavailable_tensor_runtime\nVALUE = 1\n")
        first = registry.inventory("ms-swift", self.root)
        full = {row["path"]: row["sha256"] for row in first["package_sources"]}
        self.assertEqual(set(full), set(self.git("ls-files", "--", "swift/*.py").splitlines()))
        self.assertIn("swift/dataset/prepare.py", full)
        self.write("swift/dataset/prepare.py", "import unavailable_tensor_runtime\nVALUE = 2\n")
        second = registry.inventory("ms-swift", self.root)
        self.assertEqual(first["axes"], second["axes"])
        self.assertNotEqual(first["package_sources"], second["package_sources"])

    def test_missing_and_ambiguous_swift_training_registry_fail_closed(self):
        self.swift()
        relative = "swift/trainers/factory.py"
        for content, reason in (("class TrainerFactory: pass\n", "Missing"),
                                ('TRAINER_MAPPING={"sft": Trainer}\nTRAINER_MAPPING={"ppo": Trainer}\n', "Ambiguous")):
            self.write(relative, content)
            with self.assertRaisesRegex(ValueError, reason):
                registry.inventory("ms-swift", self.root)

    def test_dynamic_swift_methods_and_nonliteral_declarations_are_rejected(self):
        self.swift()
        self.write("swift/trainers/factory.py", "TRAINER_MAPPING = discover_trainers()\n")
        with self.assertRaisesRegex(ValueError, "Unsupported registry"):
            registry.inventory("ms-swift", self.root)
        self.write("swift/trainers/factory.py", 'TRAINER_MAPPING={"sft": Trainer}\n')
        self.write("swift/arguments/options.py", 'class Arguments:\n    task_type: str = "sft"\n')
        with self.assertRaisesRegex(ValueError, "Unsupported ms-swift Literal"):
            registry.inventory("ms-swift", self.root)

    def test_repeated_cli_axis_cannot_double_count_or_hide_a_new_registry(self):
        self.swift()
        self.write("swift/cli/another.py", 'ROUTE_MAPPING={"another": "entry"}\n')
        with self.assertRaisesRegex(ValueError, "Ambiguous registry axis"):
            registry.inventory("ms-swift", self.root)

    def test_verl_entrypoints_and_distillation_names_extend_both_existing_axes(self):
        self.verl()
        result = registry.inventory("verl", self.root)
        self.assertEqual(self.axis(result, "advantage_estimator")[0]["members"], ["gae"])
        self.assertEqual(self.axis(result, "policy_loss")[0]["members"], ["clip"])
        self.assertEqual(self.axis(result, "distillation_loss")[0]["members"], ["forward_kl", "reverse_kl"])
        self.assertEqual(self.axis(result, "entrypoint")[0]["members"],
                         ["verl/trainer/main.py:evaluate:evaluation", "verl/trainer/main.py:train:ppo_trainer"])

    def test_dynamic_verl_entrypoint_or_loss_names_cannot_shrink_the_registry(self):
        self.verl()
        for relative, content in (
            ("verl/trainer/main.py", "@hydra.main(config_name=get_name())\ndef train(): pass\n"),
            ("verl/trainer/distillation/losses.py", "@register_distillation_loss(Settings(names=discover()))\ndef loss(): pass\n"),
        ):
            with self.subTest(relative=relative):
                original = (self.root / relative).read_text()
                self.write(relative, content)
                with self.assertRaises(ValueError):
                    registry.inventory("verl", self.root)
                self.write(relative, original)

    def test_missing_or_invalid_distillation_names_are_not_a_zero_denominator(self):
        self.verl()
        for names in (None, [], [1], [""]):
            self.write("verl/trainer/distillation/losses.py",
                       "@register_distillation_loss(Settings(names=%r))\ndef loss(): pass\n" % names)
            with self.assertRaisesRegex(ValueError, "invalid"):
                registry.inventory("verl", self.root)


if __name__ == "__main__":
    unittest.main()
