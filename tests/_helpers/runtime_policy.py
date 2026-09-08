"""Reversible policy scopes tied to unittest fixture lifetimes."""
from contextlib import ExitStack
from functools import wraps


def fixture_stack(owner, *, class_scope=False):
    stack = ExitStack()
    if not class_scope:
        owner.addCleanup(stack.close)
    elif hasattr(owner, "addClassCleanup"):
        owner.addClassCleanup(stack.close)
    else:
        # Python 3.7 has no class cleanup API. Preserve the class's real
        # teardown (including its synchronization), then close our scopes.
        previous = owner.tearDownClass
        def teardown(cls):
            try:
                previous()
            finally:
                stack.close()
        owner.tearDownClass = classmethod(teardown)
    return stack


def preserve_policy(backend, *names):
    """Capture teardown-restored settings before setUp, including failure paths."""
    def decorate(cls):
        setup = cls.setUp
        @wraps(setup)
        def scoped_setup(self):
            values = {name: getattr(backend.runtime.context, name) for name in names}
            fixture_stack(self).enter_context(backend.runtime.scope(**values))
            return setup(self)
        cls.setUp = scoped_setup
        return cls
    return decorate
