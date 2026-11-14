"""Matrix dispatch preserves its precision policy across callbacks and errors."""
import numpy as np
import jittor as jt

from jittor._runtime.dispatch import override_kernel
from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests


class TestMatmulPolicy(JittorTestCase):
    def test_provider_error_restores_the_callers_amp_flags(self, device):
        a, b = jt.ones((2, 3)), jt.ones((3, 4))
        seen = []
        class ProviderError(ValueError):
            pass
        outer = jt.amp_flags.prefer32
        with jt.flag_scope(amp_reg=outer):
            def fail(*args):
                seen.append(jt.flags.amp_reg)
                jt.flags.amp_reg = jt.amp_flags.prefer16
                raise ProviderError("matmul provider failed")
            with override_kernel("matmul", device, fail):
                with self.assertRaisesRegex(ProviderError, "matmul provider failed"):
                    jt.matmul(a, b)
            self.assertEqual(jt.flags.amp_reg, outer)
        self.assertEqual(seen, [outer | jt.amp_flags.keep_reduce])
        np.testing.assert_array_equal(jt.matmul(a, b).numpy(), np.full((2, 4), 3.))

    def test_nested_provider_calls_restore_each_precision_scope(self, device):
        a, b = jt.ones((2, 3)), jt.ones((3, 4))
        half = jt.ones((3,), dtype="bfloat16")
        seen = []
        def provider(x, y, *flags):
            before = jt.flags.amp_reg
            # A vector contraction takes the generic reduction route and
            # clears half preferences only for this nested invocation.
            inner = jt.matmul(half, half)
            seen.append((before, jt.flags.amp_reg, str(inner.dtype)))
            return (x.unsqueeze(-1) * y.unsqueeze(0)).sum(1)
        outer = jt.amp_flags.prefer_bfloat16
        with jt.flag_scope(amp_reg=outer):
            with override_kernel("matmul", device, provider):
                result = jt.matmul(a, b)
            self.assertEqual(jt.flags.amp_reg, outer)
        expected = outer | jt.amp_flags.keep_reduce
        self.assertEqual(seen, [(expected, expected, "bfloat16")])
        np.testing.assert_array_equal(result.float32().numpy(), np.full((2, 4), 3.))
        self.assertEqual(result.location(), "device" if device == "cuda" else "cpu")

    def test_invalid_shapes_are_rejected_before_the_provider(self, device):
        seen = []
        def provider(*args):
            seen.append(args)
            raise AssertionError("invalid shapes reached a provider")
        with override_kernel("matmul", device, provider), override_kernel("batched_matmul", device, provider):
            for ash, bsh in [((2, 3), (4, 5)), ((2, 2, 3), (3, 3, 4))]:
                with self.subTest(a=ash, b=bsh):
                    a, b = jt.ones(ash), jt.ones(bsh)
                    before = jt.flags.amp_reg
                    with self.assertRaises(jt.JittorShapeError):
                        jt.matmul(a, b)
                    self.assertEqual(jt.flags.amp_reg, before)
        self.assertEqual(seen, [])


instantiate_device_type_tests(TestMatmulPolicy, globals())
