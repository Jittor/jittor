"""Source-only bounded OpenMMLab denominators must not silently omit declarations."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _helpers.child_process import run_python_child

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"
sys.path.insert(0, str(SCRIPTS))
import registry_inventory as registry  # noqa: E402


class OpenMMLabRegistries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Registry test")
        self.git("config", "user.email", "registry@example.invalid")
        self.git("remote", "add", "origin", "https://example.invalid/upstream.git")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.git("add", path)
        self.git("commit", "--allow-empty", "-qm", "source fixture")

    def mmengine(self, extra=""):
        self.write("mmengine/version.py", '__version__="0.10.7"\n')
        self.write(
            "mmengine/registrations.py",
            """import unavailable_runtime
@MODELS.register_module()
class BaseDataPreprocessor: pass
@OPTIM_WRAPPERS.register_module()
class OptimWrapper: pass
@LOOPS.register_module()
class TrainLoop: pass
@WEIGHT_INITIALIZERS.register_module(name="Constant")
class Initializer: pass
"""
            + extra,
        )

    def mmcv(self, extra=""):
        self.write("mmcv/version.py", '__version__="2.2.0"\n')
        self.write(
            "mmcv/registrations.py",
            """import unavailable_runtime
@MODELS.register_module()
class ConvModule: pass
@HOOKS.register_module()
class SearchHook: pass
@TRANSFORMS.register_module()
class Transform: pass
"""
            + extra,
        )

    @staticmethod
    def members(value, axis):
        return next(row["members"] for row in value["axes"] if row["axis"] == axis)

    def test_all_conditions_aliases_literal_loops_and_compiled_declarations_remain(self):
        self.mmcv("""
from mmengine.registry import MODELS as COMPONENTS
LOCAL = COMPONENTS
for module in [nn.ReLU, nn.Sigmoid]:
    LOCAL.register_module(module=module)
if has_cuda_extension:
    @MODELS.register_module(name=["CompiledKernel", "CompiledAlias"])
    class Extension: pass
else:
    @MODELS.register_module(name="PythonAlternative")
    class Alternative: pass
""")
        result = registry.inventory("mmcv", self.root)
        self.assertEqual(
            self.members(result, "model_component"),
            [
                "CompiledAlias",
                "CompiledKernel",
                "ConvModule",
                "PythonAlternative",
                "ReLU",
                "Sigmoid",
            ],
        )
        declared = result["declaration_details"]["declarations"]
        self.assertEqual(
            {
                row["conditions"][0]["branch"]
                for row in declared
                if row["conditions"] and row["conditions"][0]["kind"] == "if"
            },
            {False, True},
        )
        for item in result["sources"]:
            self.assertEqual(
                item["sha256"], hashlib.sha256((self.root / item["path"]).read_bytes()).hexdigest()
            )

    def test_registered_function_is_not_silently_dropped(self):
        self.mmcv("@MODELS.register_module()\ndef build_component(): pass\n")
        result = registry.inventory("mmcv", self.root)
        self.assertEqual(self.members(result, "model_component"), ["ConvModule", "build_component"])

    def test_new_alias_changes_denominator_and_pinned_upstream_sha(self):
        self.mmcv()
        first = registry.inventory("mmcv", self.root)
        self.mmcv('@MODELS.register_module(name="Another")\nclass Added: pass\n')
        second = registry.inventory("mmcv", self.root)
        self.assertEqual(self.members(second, "model_component"), ["Another", "ConvModule"])
        self.assertNotEqual(first["upstream"]["commit"], second["upstream"]["commit"])

    def test_dynamic_selected_loop_is_rejected_instead_of_counting_variable_name(self):
        self.mmengine(
            "for module in discover_wrappers():\n    OPTIM_WRAPPERS.register_module(module=module)\n"
        )
        with self.assertRaisesRegex(ValueError, "Unresolved bounded"):
            registry.inventory("mmengine", self.root)

    def test_dynamic_loop_shadows_imported_class(self):
        self.mmengine(
            "from external import Wrapper\nfor Wrapper in discover():\n    OPTIM_WRAPPERS.register_module(module=Wrapper)\n"
        )
        with self.assertRaisesRegex(ValueError, "Unresolved bounded"):
            registry.inventory("mmengine", self.root)

    def test_unapplied_decorator_does_not_invent_registered_member(self):
        self.mmcv('MODELS.register_module(name="Ghost")\n')
        with self.assertRaisesRegex(ValueError, "Unresolved bounded"):
            registry.inventory("mmcv", self.root)

    def test_indirect_registration_callable_is_not_silently_omitted(self):
        self.mmcv("register = MODELS.register_module\nregister(module=Plugin)\n")
        with self.assertRaisesRegex(ValueError, "Unresolved bounded"):
            registry.inventory("mmcv", self.root)

    def test_dynamic_name_and_kwargs_fail_closed(self):
        for extra in (
            "MODELS.register_module(name=discover_name(), module=Layer)\n",
            'MODELS.register_module(name="Named", **options)\n',
        ):
            self.mmcv(extra)
            with self.assertRaisesRegex(ValueError, "Unresolved bounded"):
                registry.inventory("mmcv", self.root)

    def test_unbounded_optimizer_imports_are_disclosed_without_a_partial_denominator(self):
        self.mmengine(
            "for optimizer in dir(torch.optim):\n    OPTIMIZERS.register_module(module=optimizer)\n"
        )
        result = registry.inventory("mmengine", self.root)
        self.assertEqual(
            {row["axis"] for row in result["axes"]},
            {"model_component", "optim_wrapper", "loop", "weight_initializer"},
        )
        excluded = result["declaration_details"]["out_of_scope"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["registry"], "OPTIMIZERS")
        self.assertTrue(excluded[0]["conditions"])
        self.assertIn("Not whole-library", result["scope"])

    def test_empty_selected_axis_cannot_shrink_into_missing_axis(self):
        self.mmengine()
        self.write("mmengine/registrations.py", "@MODELS.register_module()\nclass Model: pass\n")
        with self.assertRaisesRegex(ValueError, "Empty or invalid"):
            registry.inventory("mmengine", self.root)

    def test_dirty_checkout_is_rejected(self):
        self.mmcv()
        with (self.root / "mmcv/version.py").open("a") as stream:
            stream.write("# dirty\n")
        with self.assertRaisesRegex(ValueError, "dirty"):
            registry.inventory("mmcv", self.root)

    def test_cli_runs_without_site_packages_or_tensor_runtime(self):
        self.mmengine()
        result = run_python_child(
            [
                "-S",
                str(SCRIPTS / "registry_inventory.py"),
                "--repo",
                "mmengine",
                "--checkout",
                str(self.root),
            ],
            cwd=ROOT,
            check=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(self.members(value, "optim_wrapper"), ["OptimWrapper"])
        self.assertEqual(value["upstream"]["version"], "0.10.7")
