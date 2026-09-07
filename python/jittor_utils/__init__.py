"""Standalone bootstrap for the build utilities' single physical owner.

Loading by path deliberately avoids importing Jittor before its compiler tools.
"""
import importlib.util as _importlib_util
import os as _os
import sys as _sys

_implementation = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)),
    "jittor", "build", "utils", "__init__.py",
)
_spec = _importlib_util.spec_from_file_location(
    __name__, _implementation,
    submodule_search_locations=[_os.path.dirname(_implementation)],
)
_module = _importlib_util.module_from_spec(_spec)
_sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
