import importlib
import unittest

import torch
import jittor as _native_jittor


API_NAMES = ("torch.compile", "torch.jit.script", "torch.jit.trace")


class TestTorchCompilerFidelity(unittest.TestCase):

    def test_unbacked_annotation_import_and_eager_metadata(self):
        decorators = importlib.import_module("torch._dynamo.decorators")
        self.assertIs(decorators, torch._dynamo.decorators)
        tensor = torch.zeros((2, 3), device="cpu")
        self.assertIsNone(decorators.mark_unbacked(tensor, 0, hint_override=8))
        decorators.mark_unbacked(tensor, 1)
        self.assertEqual(tensor._dynamo_unbacked_indices, {0, 1})
        self.assertEqual(tensor._dynamo_hint_overrides, {0: 8})
        self.assertEqual(tensor._specialize_on, {1: []})
        self.assertTrue(tensor._has_dynamo_dim_marking)
        self.assertEqual(tuple(tensor.shape), (2, 3))
        self.assertEqual(tensor.tolist(), [[0.0] * 3] * 2)
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch._dynamo.decorators.mark_unbacked")
        self.assertIs(record.implementation, decorators.mark_unbacked)
        self.assertEqual(record.level, fidelity.Fidelity.APPROXIMATE)

    def test_unbacked_sequence_annotations_and_invalid_index(self):
        tensor = torch.zeros((2, 3), device="cpu")
        mark = torch._dynamo.decorators.mark_unbacked
        mark(tensor, [0, 1], hint_override=8)
        mark(tensor, (0,))
        self.assertEqual(tensor._dynamo_unbacked_indices, {0, 1})
        self.assertEqual(tensor._dynamo_hint_overrides, {})
        with self.assertRaisesRegex(AssertionError, "Expected index"):
            mark(tensor, "0")

    def test_unbacked_compiler_constraints_fail_before_annotation(self):
        mark = torch._dynamo.decorators.mark_unbacked
        for options in ({"strict": True}, {"specialize_on": []},
                        {"shape_id": "batch"}, {"min": 2}, {"max": 8}):
            with self.subTest(options=options):
                tensor = torch.zeros((2, 3), device="cpu")
                with self.assertRaisesRegex(NotImplementedError, "Dynamo"):
                    mark(tensor, 0, **options)
                self.assertFalse(hasattr(tensor, "_dynamo_unbacked_indices"))

    def test_compiler_callables_are_independently_importable(self):
        compiler = importlib.import_module(
            "jittor.compat.torch.installers.compiler")
        for name in ("compile", "script", "trace"):
            with self.subTest(name=name):
                implementation = getattr(compiler, name)
                self.assertTrue(callable(implementation))
                self.assertEqual(implementation.__name__, name)

    def test_inductor_cache_import_is_explicitly_unsupported_on_use(self):
        from torch._inductor.utils import fresh_cache
        compiler = importlib.import_module("jittor.compat.torch.installers.compiler")
        self.assertIs(fresh_cache, compiler._api_inductor_fresh_cache)
        with self.assertRaisesRegex(NotImplementedError, "Inductor cache isolation"):
            with fresh_cache():
                self.fail("unsupported Inductor context must never enter")
        utils = importlib.import_module("torch._inductor.utils")
        self.assertFalse(hasattr(utils, "unimplemented_analysis"))

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
        with _native_jittor.flag_scope(use_cuda=0):
            self.assertEqual(add_one(torch.zeros(1)).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
