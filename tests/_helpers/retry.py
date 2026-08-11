"""Retry decorator for tests that exercise nondeterministic tuning paths."""

from jittor import LOG


def retry(num):
    def outer(func):
        def inner(*args):
            for index in range(num):
                if index == num - 1:
                    func(*args)
                    break
                try:
                    func(*args)
                    break
                except Exception:
                    pass
                LOG.v("Retry {}".format(index))

        return inner

    return outer
