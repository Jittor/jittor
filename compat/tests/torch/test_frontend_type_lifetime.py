"""Native caches keyed by a frontend type do not keep the type alive.

The module-call, tensor-frontend policy, result-type and dtype caches used to
hold every type they had seen, so a class made at run time -- a Module defined
in a function, a Parameter subclass -- lived for the rest of the process with
everything its body closed over. They now drop an entry when its type is
collected, before the address can be reused by another type.
"""
import gc
import unittest
import weakref

import numpy as np
import torch


def _collected(make):
    ref = make()
    gc.collect()
    return ref() is None


class TestFrontendTypeLifetime(unittest.TestCase):
    def test_module_defined_in_a_function_is_collected(self):
        def make():
            offset = torch.ones(16)

            class Local(torch.nn.Module):
                def forward(self, x):
                    return x + offset

            self.assertEqual(Local()(torch.ones(16)).sum().item(), 32.0)
            return weakref.ref(Local)

        self.assertTrue(_collected(make))

    def test_parameter_subclass_is_collected(self):
        def make():
            class Local(torch.nn.Parameter):
                pass

            p = Local(torch.ones(3))
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual((p * 2).sum().item(), 6.0)
            return weakref.ref(Local)

        self.assertTrue(_collected(make))

    def test_a_new_type_never_sees_a_collected_one_s_entry(self):
        # Alternate classes whose dispatch differs: one overrides forward over
        # the builtin's execute, one does not. Collected each round, so a new
        # class may well land on the address of the previous one.
        x = torch.tensor([-1.0, 2.0])
        for i in range(40):
            if i % 2:
                class Act(torch.nn.ReLU):
                    def forward(self, x):
                        return x * 3.0
                want = [-3.0, 6.0]
            else:
                class Act(torch.nn.ReLU):
                    pass
                want = [0.0, 2.0]
            np.testing.assert_array_equal(Act()(x).numpy(), np.array(want, "float32"), str(i))
            del Act
            gc.collect()


if __name__ == "__main__":
    unittest.main()
