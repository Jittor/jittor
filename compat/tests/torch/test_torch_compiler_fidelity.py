import importlib
import unittest

import jittor as torch


API_NAMES = ("torch.compile", "torch.jit.script", "torch.jit.trace")


class TestTorchCompilerFidelity(unittest.TestCase):

    def test_compiler_callables_are_independently_importable(self):
        compiler = importlib.import_module(
            "jittor.compat.torch.installers.compiler")
        for name in ("compile", "script", "trace"):
            with self.subTest(name=name):
                implementation = getattr(compiler, name)
                self.assertTrue(callable(implementation))
                self.assertEqual(implementation.__name__, name)

    def test_public_namespaces_publish_the_same_objects(self):
        compiler = importlib.import_module(
            "jittor.compat.torch.installers.compiler")
        self.assertIs(torch.compile, compiler.compile)
        self.assertIs(torch.jit.script, compiler.script)
        self.assertIs(torch.jit.trace, compiler.trace)
        self.assertIs(torch.jit.trace_module, compiler.trace)

    def test_fidelity_report_names_exactly_this_family(self):
        compiler = importlib.import_module(
            "jittor.compat.torch.installers.compiler")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        records = tuple(
            fidelity.fidelity_of(name) for name in API_NAMES
        )
        self.assertEqual(tuple(record.api for record in records), API_NAMES)
        self.assertEqual(
            tuple(record.level for record in records),
            (fidelity.Fidelity.APPROXIMATE,) * len(API_NAMES),
        )
        self.assertIs(records[0].implementation, compiler.compile)
        self.assertIs(records[1].implementation, compiler.script)
        self.assertIs(records[2].implementation, compiler.trace)
        self.assertTrue(all(record.detail for record in records))

    def test_cpu_passthrough_and_semantic_refusals_are_preserved(self):
        def add_one(value):
            return value + 1

        self.assertIs(torch.compile(add_one), add_one)
        self.assertIs(torch.jit.script(add_one), add_one)
        self.assertIs(torch.jit.trace(add_one, torch.ones(1)), add_one)
        with self.assertRaises(NotImplementedError):
            torch.compile(add_one, fullgraph=True)
        with self.assertRaises(NotImplementedError):
            torch.jit.trace(add_one, torch.ones(1), check_trace=True)
        with torch.flag_scope(use_cuda=0):
            self.assertEqual(add_one(torch.zeros(1)).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
