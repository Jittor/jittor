# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A failed `import torch` has to be catchable as an ImportError.

The ecosystem's optional-dependency idiom is

    try:
        import torch
    except ImportError:
        pass

and a bare RuntimeError walks straight through it. With jittor-torch deployed,
`import tensorboardX` therefore failed outright -- and it fails that way for any
package written the same way, *including for programs using only native Jittor*.

The exception stays a RuntimeError too: reached through an explicit `install()`
call the same failure is API misuse rather than a missing module, and callers
already catching RuntimeError keep working. Both halves are asserted, because
either one alone is a regression waiting to be reintroduced by someone
"tidying" the class up.
"""

import unittest

import pytest

pytestmark = pytest.mark.structure


class TestActivationErrorKind(unittest.TestCase):

    def setUp(self):
        from jittor.compat.torch import TorchActivationError
        self.error = TorchActivationError

    def test_it_is_an_import_error(self):
        self.assertTrue(issubclass(self.error, ImportError))

    def test_it_is_still_a_runtime_error(self):
        self.assertTrue(issubclass(self.error, RuntimeError))

    def test_the_optional_dependency_idiom_catches_it(self):
        caught = None
        try:
            raise self.error("activation failed")
        except ImportError as exc:
            caught = exc
        self.assertIsNotNone(caught, "the idiom every package uses did not catch it")

    def test_an_explicit_runtime_handler_still_catches_it(self):
        caught = None
        try:
            raise self.error("activation failed")
        except RuntimeError as exc:
            caught = exc
        self.assertIsNotNone(caught)

    def test_the_message_is_preserved(self):
        # The whole value of the actionable message is that it survives the
        # change of class; an ImportError that says nothing is not an
        # improvement over a RuntimeError that does.
        with self.assertRaises(ImportError) as ctx:
            raise self.error("redeploy the shim: python -m jittor.compat.shim deploy")
        self.assertIn("redeploy the shim", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
