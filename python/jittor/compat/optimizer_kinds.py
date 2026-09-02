"""Which update rule an optimizer actually implements.

Two places used to answer this the same wrong way -- ``fsdp2/optimizer.py``
(to pick which sharded update to run) and ``torch/optimizers.py`` (to decide
what an optimizer's ``state`` and ``state_dict()`` look like):

    name = type(opt).__name__.lower()
    if "adamw" in name: return "adamw"
    if "adam" in name:  return "adam"
    if "sgd" in name:   return "sgd"

A substring of the class *name* is not evidence about the class's behaviour,
and it fails in both directions:

* ``SGDW``, ``MyAdamWrapper``, ``NoAdamW`` match a rule they do not implement;
* anything else -- ``Lion``, ``Adafactor``, a from-scratch optimizer -- is
  unrecognised even when it is a plain subclass of one of these.

The quiet half is the dangerous one. A subclass named ``...Adam...`` that
overrides ``step()`` with different math was recognised as "adam", and
``fsdp2/optimizer.py`` then ran *its own* base AdamW update on the sharded
parameters. The user's update rule was not applied and nothing said so; the
run just trained a different model. The loud half -- ``NotImplementedError``
for a custom optimizer -- is the one the audit recorded, because it is the one
anybody noticed.

So: identify by the class, and where running the wrong arithmetic is the
failure, also require that the class has not replaced the arithmetic.
"""

from __future__ import absolute_import

__all__ = ["kind_of", "KNOWN_KINDS"]


#: ``kind`` -> attribute name on ``jittor.optim``. The kind strings are the
#: ones both call sites already use. Resolved lazily: this module is imported
#: from ``jittor.compat`` while ``jittor`` itself may still be initialising,
#: and ``jittor.optim`` is not necessarily bound yet.
_KIND_ATTRS = (
    # AdamW before Adam only for readability -- these are looked up by class
    # identity through the MRO, not by prefix, so order does not decide
    # anything. AdamW is a sibling of Adam here, not a subclass.
    ("adamw", "AdamW"),
    ("adam", "Adam"),
    ("sgd", "SGD"),
    ("rmsprop", "RMSprop"),
    ("adan", "Adan"),
)

KNOWN_KINDS = tuple(kind for kind, _attr in _KIND_ATTRS)


def _known_classes():
    import jittor as jt

    optim = getattr(jt, "optim", None)
    if optim is None:
        return ()
    found = []
    for kind, attr in _KIND_ATTRS:
        cls = getattr(optim, attr, None)
        if isinstance(cls, type):
            found.append((kind, cls))
    return tuple(found)


def kind_of(opt, require_unmodified_step=False):
    """The update rule ``opt`` implements, or ``None`` if that is not known.

    :param require_unmodified_step: refuse a subclass that has replaced the
        update rule. Pass this wherever the answer selects *which arithmetic to
        run on the user's weights* -- being wrong there silently applies the
        base class's update instead of the one the subclass defines. Leave it
        off where the answer only describes state layout.

    ``None`` means "do not assume": the caller must refuse rather than guess.
    """
    cls = type(opt)
    known = _known_classes()
    for base in getattr(cls, "__mro__", (cls,)):
        for kind, known_cls in known:
            if base is known_cls:
                if require_unmodified_step and not _step_is_inherited(cls, base):
                    # The subclass defines its own update. Running `base`'s
                    # arithmetic here would ignore it, silently.
                    return None
                return kind
    return None


def _step_is_inherited(cls, base):
    """Whether ``cls`` still uses ``base``'s update rule.

    Compared as plain functions off the class ``__dict__`` chain, so a subclass
    that merely adds ``__init__`` or new defaults still counts as inherited --
    that is the ordinary "custom Adam subclass" people write, and it is safe.
    """
    return getattr(cls, "step", None) is getattr(base, "step", None)
