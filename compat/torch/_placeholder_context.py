"""Shared inert context behavior for explicitly unsupported frontend APIs."""


class _PlaceholderContext:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
