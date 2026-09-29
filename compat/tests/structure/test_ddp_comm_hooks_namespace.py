"""Optional communication-hook imports must not invent collective execution."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_ddp_compression_namespace_is_importable_and_fails_closed():
    source = Path(__file__).resolve().parents[3] / "compat/torch/installers/distributed.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_install_ddp_comm_hooks")
    records = {}
    def register(namespace, prefix, names, level, detail):
        for name in names:
            records[prefix + "." + name] = (getattr(namespace, name), level)
    context = {"register_api_bindings": register,
               "Fidelity": SimpleNamespace(UNIMPLEMENTED="unimplemented")}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), context)
    algorithms, modules = SimpleNamespace(), {}
    context["_install_ddp_comm_hooks"](algorithms, modules)
    package = algorithms.ddp_comm_hooks
    prefix = "torch.distributed.algorithms.ddp_comm_hooks"
    assert package is modules[prefix] and package.__path__ == []
    assert package.default_hooks is modules[prefix + ".default_hooks"]
    assert package.powerSGD_hook is modules[prefix + ".powerSGD_hook"]
    assert len(records) == 7
    # Resolution is passive; installation/import does not execute communication.
    for full_name, (api, fidelity) in records.items():
        assert fidelity == "unimplemented"
        assert api.__module__ + "." + api.__name__ == full_name
        with pytest.raises(NotImplementedError, match="not implemented"):
            api(None, None)
    identities = {name: value[0] for name, value in records.items()}
    context["_install_ddp_comm_hooks"](algorithms, modules)
    assert identities == {name: value[0] for name, value in records.items()}
