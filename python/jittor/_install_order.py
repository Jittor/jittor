# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The order jittor's runtime installers run in -- written down, and checked.

Importing jittor rewrites ``Var`` and the root namespace in a fixed sequence of
steps: native bindings, indexing, the CUDA full-reduce fast path, the backends'
``post_process``, the generated ``x.func_()`` in-place aliases, ACL's operator
swap, the MPI-free collectives, and finally the Torch compatibility layer. There
are 169 assignments of the form ``Var.x = ...`` across eight files, and **which
one wins is decided entirely by the order these steps run in**.

That order was load-bearing and undeclared. It lived in the physical arrangement
of statements in ``jittor/__init__.py``, so:

* moving one import "to group the imports together" silently changed which
  implementation of ``Var.sum`` a user gets;
* a reader had no way to tell an ordering constraint from an accident, and so no
  way to tell which rearrangements were safe;
* a step that failed to run at all -- a backend whose ``post_process`` was
  skipped, a compat composition that returned early -- left a half-installed
  runtime that still imported cleanly.

``jittor/nn/functional/softmax.py`` says it out loud: *"Backend integrations
replace the public symbol at runtime."* These patches are the **contract**, not
incidental repair work, so they get written down like one.

This module holds no jittor state and imports nothing from jittor: it is loaded
before the runtime exists.

Usage: every installer site calls :func:`record` with its declared name, and
``jittor/__init__.py`` calls :func:`verify` once the sequence should be
complete. ``tests/structure/test_install_order.py`` fails the gate if the list
here and the ``record()`` calls in the tree ever drift apart.
"""


class Step:
    """One installation step.

    ``required`` steps run on every machine. The optional ones are guarded by
    hardware or by an optional dependency, so their absence is not a fault --
    but their *position* is still fixed, and running one out of turn is.
    """

    __slots__ = ("name", "required", "why")

    def __init__(self, name, required, why):
        self.name = name
        self.required = required
        self.why = why

    def __repr__(self):
        return "Step(%r, required=%r)" % (self.name, self.required)


#: The sequence, earliest first. Each entry says why it sits where it does --
#: that is the part that was missing, not the list.
SEQUENCE = (
    Step(
        "nn.var_bindings", True,
        "jittor/nn/_bindings.py, reached by `from . import nn`. Puts the "
        "native math and complex-scalar methods on Var. First, because every "
        "later step either wraps one of these or assumes it exists."),
    Step(
        "misc.var_indexing", True,
        "jittor/misc/indexing.py. Installs __getitem__/__setitem__. After the "
        "bindings (it reuses them) and before the fast path and the compat "
        "layer, both of which index Vars while installing."),
    Step(
        "nn.full_reduce_fast_path", True,
        "jittor/nn/backends/full_reduce_cuda.py. Replaces Var.sum/Var.mean "
        "and the root jt.sum/jt.mean with the two-stage CUB reduction. Must "
        "precede the in-place alias generation (so sum_/mean_ wrap the fast "
        "path, not the binding it replaced) and the Torch wrappers (so they "
        "layer on top rather than under)."),
    Step(
        "backends.post_process", False,
        "jittor_utils.backends[*].post_process(). Hardware backends adjust "
        "flags (ACL sets amp_reg here). After the Python-side Var surface is "
        "complete, before anything reads those flags."),
    Step(
        "root.inplace_aliases", True,
        "jittor/__init__.py. Generates x.func_() for every eligible Var "
        "method by walking Var.__dict__. Order-critical in both directions: a "
        "method installed after this point gets no in-place alias, and a "
        "method replaced after this point leaves its alias bound to the OLD "
        "implementation."),
    Step(
        "acl.change_function", False,
        "jittor/extern/acl/acl_compiler.py. Swaps operator implementations "
        "for Ascend. After the aliases (it replaces functions the aliases "
        "already captured -- see the note above) and before compat."),
    Step(
        "collectives.hccl", False,
        "jittor/__init__.py. Routes Var.mpi_all_reduce/mpi_broadcast to HCCL "
        "when MPI did not provide them."),
    Step(
        "collectives.nccl", False,
        "jittor/__init__.py. The NVIDIA equivalent. MUST come after "
        "collectives.hccl: its guard is `not hasattr(Var, 'mpi_all_reduce')`, "
        "so running it first would claim the name and silently disable the "
        "HCCL route on an Ascend box."),
    Step(
        "compat.runtime_composition", True,
        "jittor/compat/runtime.py compose(). The Torch compatibility layer. "
        "Last of the patchers on purpose: it wraps whatever the native "
        "runtime ended up with, so everything above must already be final."),
    Step(
        "optim.public_exports", True,
        "jittor/optim/__init__.py _refresh_public_exports(). Re-runs the star "
        "export after compat added optimizer classes. Strictly after "
        "compat.runtime_composition -- that is the whole point of it."),
)

_BY_NAME = {step.name: step for step in SEQUENCE}

#: Steps recorded so far, in the order they actually ran.
_observed = []


class InstallOrderError(RuntimeError):
    """An installer ran out of turn, twice, or under an unknown name."""


def record(name):
    """Note that the installer called ``name`` has just run.

    Raises immediately rather than at :func:`verify` time, so the traceback
    points at the site that ran out of turn instead of at the end of the import.
    """
    step = _BY_NAME.get(name)
    if step is None:
        raise InstallOrderError(
            "unknown installer %r; add it to SEQUENCE in "
            "jittor/_install_order.py, in the position its ordering "
            "constraints require" % (name,))
    if name in _observed:
        raise InstallOrderError(
            "installer %r ran twice; installers replace attributes, so a "
            "second run rebinds them on top of the first" % (name,))
    index = SEQUENCE.index(step)
    for earlier in SEQUENCE[index + 1:]:
        if earlier.name in _observed:
            raise InstallOrderError(
                "installer %r ran after %r, but SEQUENCE puts it before. "
                "%s" % (name, earlier.name, step.why))
    _observed.append(name)


def observed():
    """The steps that have run, in order. A copy; callers may not mutate it."""
    return tuple(_observed)


def verify():
    """Check the sequence is complete and in order. Called once, at import end."""
    missing = [step.name for step in SEQUENCE
               if step.required and step.name not in _observed]
    if missing:
        raise InstallOrderError(
            "jittor finished importing without running these installers: %s. "
            "The runtime is half-patched: it will import, and some Var "
            "methods will be the ones the step was supposed to replace."
            % (", ".join(missing),))
    order = [SEQUENCE.index(_BY_NAME[name]) for name in _observed]
    if order != sorted(order):
        raise InstallOrderError(
            "installers ran out of order: %s" % (_observed,))
    return observed()


def reset_for_testing():
    """Forget what has run. Only ``tests/`` may call this."""
    del _observed[:]
