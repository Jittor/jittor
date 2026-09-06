"""Compatibility shim for tools that still invoke ``setup.py`` directly."""

from setuptools import find_packages, setup


setup(packages=find_packages("python") + [
    "jittor.backends." + name for name in find_packages("backends")
])
