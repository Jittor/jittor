# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The two helpers the package root uses to assemble itself.

They live here rather than in ``__init__.py`` because the root is required to
define nothing: ``tests/structure/test_runtime_composition_structure.py`` states
that as the architecture contract, and it holds because anything defined there
is reachable as ``jittor.<name>`` -- which is how a private helper becomes part
of the public surface by accident, and how the root grew the composition logic
that contract was written to keep out.
"""


def publish(namespace, module, names):
    """Copy ``names`` out of ``module`` and into ``namespace``.

    The target namespace is a parameter rather than the caller's ``globals()``
    so that this is a plain function of its arguments: moving it out of the root
    is the whole point, and a helper that reads its caller's frame cannot move.
    """
    for name in names:
        namespace[name] = getattr(module, name)


def make_inplace_alias(name, operation):
    """Build the ``x.foo_()`` method that assigns ``operation``'s result back.

    ``assign`` is what makes it in place; the wrapper exists so the declared
    alias table stays a table instead of 114 near-identical closures.
    """
    def inplace(self, *args, **kwargs):
        return self.assign(operation(self, *args, **kwargs))
    inplace.__name__ = name
    return inplace
