"""Shared narrow public-import/non-mutation contract for leaf adapters."""
import ast


def boundary_errors(source):
    errors = []
    tree = ast.parse(source)
    allowed = {"jittor.compat.module_patcher", "jittor.compat.torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("jittor"):
            if node.module not in allowed or any(name.name.startswith("_") for name in node.names):
                errors.append("private framework import")
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in ("jt", "jittor", "torch"):
                if node.attr.startswith("_") or isinstance(node.ctx, (ast.Store, ast.Del)):
                    errors.append("private access or framework mutation")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr"):
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in ("jt", "jittor", "torch"):
                errors.append("framework mutation")
    return errors
