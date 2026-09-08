"""Compatibility shim for tools that still invoke ``setup.py`` directly."""

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from pathlib import Path


class BuildPythonWithCore(build_py):
    """Install the checkout's single native source tree as runtime data."""

    def core_files(self):
        root = Path(__file__).resolve().parent / "src"
        for source in sorted(root.rglob("*")):
            if source.is_file() and source.suffix in (".h", ".cc"):
                yield source, Path(self.build_lib) / "jittor/src" / source.relative_to(root)

    def run(self):
        super().run()
        for source, destination in self.core_files():
            self.mkpath(str(destination.parent))
            self.copy_file(str(source), str(destination))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [str(target) for _, target in self.core_files()]


setup(cmdclass={"build_py": BuildPythonWithCore}, packages=find_packages(
    "python", exclude=("jittor.compat", "jittor.compat.*")
) + [
    "jittor.backends." + name for name in find_packages("backends")
])
