"""Package the optional frontend without importing or compiling Jittor."""

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class BuildCompatPython(build_py):
    def find_package_modules(self, package, package_dir):
        # PEP 517 invokes setup through an in-memory wrapper, so setuptools'
        # implicit setup-script exclusion differs from direct setup.py builds.
        return [
            entry for entry in super().find_package_modules(package, package_dir)
            if entry[:2] != ("jittor.compat", "setup")
        ]


# The public torch entry and deploy both use resources/torch/__init__.py.
# Exclude its internal spelling so it is an entrypoint, not a second API owner.
packages = find_packages(".", exclude=("shim.resources.torch", "shim.resources.torch.*"))
setup(
    packages=["jittor.compat", "torch"] + ["jittor.compat." + name for name in packages],
    cmdclass={"build_py": BuildCompatPython},
)
