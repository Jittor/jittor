"""Small neural-network construction helpers."""


def skip_init(module_cls, *args, **kw):
    return module_cls(*args, **kw)
