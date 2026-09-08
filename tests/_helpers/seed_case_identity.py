"""Static class/method identities used by the layout-only seed migration."""

import ast


def _generated(method):
    for decorator in method.decorator_list:
        function = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(function, "id", getattr(function, "attr", "")).lstrip("_")
        if name == "ops" or "dtypes" in name.lower():
            return True
    return False


def source_cases(source):
    """Return scoped explicit cases, parameter families and device-class aliases."""
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {}

    def class_methods(name, visiting):
        if name in methods:
            return methods[name]
        if name in visiting:
            return {}
        result = {}
        node = classes[name]
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                result.update(class_methods(base.id, visiting | {name}))
        result.update(
            (method.name, _generated(method))
            for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name.startswith("test")
        )
        methods[name] = result
        return result

    cases, families = set(), set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test"
        ):
            cases.add(node.name)
            if _generated(node):
                families.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for name, generated in class_methods(node.name, set()).items():
                identity = node.name + "::" + name
                cases.add(identity)
                if generated:
                    families.add(identity)
    devices = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and node.args
            and getattr(node.func, "id", getattr(node.func, "attr", ""))
            == "instantiate_device_type_tests"
            and isinstance(node.args[0], ast.Name)
        ):
            name = node.args[0].id
            for device in ("CPU", "CUDA", "ROCM", "NPU"):
                devices[name + device] = name
    return cases, families, devices
