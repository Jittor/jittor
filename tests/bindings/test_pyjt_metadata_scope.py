"""An audited metadata binding can omit policy; mixed overloads cannot."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("annotations,expected", [
    ([True], 0), ([False], 1), ([True, True], 0), ([True, False], 1),
])
def test_frontend_metadata_requires_every_overload(annotations, expected):
    path = Path(__file__).resolve().parents[2] / "python/jittor/build/pyjt_compiler.py"
    spec = importlib.util.spec_from_file_location("_pyjt_metadata_generator", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    source = "// @pyjt(Var)\nstruct VarHolder {\n"
    for index, annotated in enumerate(annotations):
        source += "// @pyjt(metadata)\n"
        if annotated:
            source += "// @attrs(frontend_metadata)\n"
        source += "int metadata(%s);\n" % ("int axis" if index else "")
    source += "};\n"
    generated = generator.compile_src(source, "metadata.h", "metadata")
    assert generated.count("PyTensorFrontendScope tensor_frontend") == expected
