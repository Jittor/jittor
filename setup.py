"""Compatibility shim for tools that still invoke ``setup.py`` directly."""

from setuptools import find_packages, setup


# jittor.src is a data-only namespace mapped to src by pyproject.toml.
setup(packages=["jittor.src"] + find_packages(
    "python", exclude=("jittor.compat", "jittor.compat.*")
) + [
    "jittor.backends." + name for name in find_packages("backends")
])
