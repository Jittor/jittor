"""Shared source-boundary contract for independently published adapters."""
import ast

PUBLIC_IMPORTS = {
    "jittor", "jittor.compat.module_patcher", "jittor.compat.permissive",
    "jittor.compat.transaction", "jittor.compat.diagnostics",
    "jittor.compat.torch.gradient_task",
}


def check_source(source, filename="adapter"):
    tree = ast.parse(source, filename=filename)
    roots = {"jt", "jittor", "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("jittor", "torch"):
                    roots.add(alias.asname or alias.name)
                if alias.name.startswith("jittor") and alias.name not in PUBLIC_IMPORTS:
                    raise AssertionError(filename + ": private framework import " + alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            module = node.module or ""
            if module.startswith("jittor") and module not in PUBLIC_IMPORTS:
                raise AssertionError(filename + ": private framework import " + module)
            if module.startswith("jittor"):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        raise AssertionError(filename + ": private framework import " + alias.name)
                    if module == "jittor":
                        roots.add(alias.asname or alias.name)
    def framework(node):
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in roots
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and framework(node) and node.attr.startswith("_"):
            raise AssertionError(filename + ": private framework attribute " + node.attr)
        targets = (node.targets if isinstance(node, ast.Assign) else
                   (node.target,) if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else
                   node.targets if isinstance(node, ast.Delete) else ())
        if any(isinstance(target, ast.Attribute) and framework(target) for target in targets):
            raise AssertionError(filename + ": framework attribute mutation")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            if node.func.id in ("setattr", "delattr") and framework(node.args[0]):
                raise AssertionError(filename + ": framework attribute mutation")
            if node.func.id in ("getattr", "hasattr") and framework(node.args[0]) and len(node.args) > 1:
                if isinstance(node.args[1], ast.Constant) and str(node.args[1].value).startswith("_"):
                    raise AssertionError(filename + ": private framework attribute lookup")


def audit_adapter(package):
    sources = [path for path in package.rglob("*.py") if "tests" not in path.relative_to(package).parts]
    if not sources:
        raise AssertionError("adapter has no sources")
    for path in sources:
        check_source(path.read_text(encoding="utf8"), str(path))
