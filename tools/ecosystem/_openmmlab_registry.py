"""Enumerate explicit OpenMMLab registry declarations without importing packages."""

import ast

AXES = {
    "mmengine": {
        "MODELS": "model_component",
        "OPTIM_WRAPPERS": "optim_wrapper",
        "LOOPS": "loop",
        "WEIGHT_INITIALIZERS": "weight_initializer",
    },
    "mmcv": {"MODELS": "model_component", "HOOKS": "hook", "TRANSFORMS": "transform"},
}


def declarations(root, repo, files):
    rows = []
    dynamic = []
    other = []

    def expression(node):
        return ast.dump(node, annotate_fields=True, include_attributes=False)

    for relative in sorted(files):
        path = root / relative
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text())
        registry_aliases = {name: name for name in AXES[repo]}
        symbols = {
            n.name: n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        }
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for alias in n.names:
                    symbols[alias.asname or alias.name] = alias.name
                    if alias.name in AXES[repo]:
                        registry_aliases[alias.asname or alias.name] = alias.name
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Name)
                and n.value.id in registry_aliases
            ):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        registry_aliases[target.id] = registry_aliases[n.value.id]

        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "register_module"
                and isinstance(node.value, ast.Name)
                and node.value.id in registry_aliases
            ):
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    dynamic.append(
                        {
                            "registry": registry_aliases[node.value.id],
                            "source": relative,
                            "line": node.lineno,
                            "expression": expression(node),
                            "conditions": [],
                            "members": [],
                            "reason": "indirect registry callable",
                        }
                    )

        def values(node, env):
            if isinstance(node, (ast.List, ast.Tuple)):
                return [v for item in node.elts for v in values(item, env)]
            if isinstance(node, ast.Name) and node.id in env:
                return env[node.id]
            return [node]

        def call(node, env, guards, default=None):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register_module"
            ):
                return
            if not isinstance(node.func.value, ast.Name):
                raise ValueError("Unrecognized registry owner " + relative + ":" + str(node.lineno))
            registry = registry_aliases.get(node.func.value.id, node.func.value.id)
            if registry not in AXES[repo]:
                other.append(
                    {
                        "registry": registry,
                        "source": relative,
                        "line": node.lineno,
                        "expression": expression(node),
                        "conditions": guards,
                    }
                )
                return
            if any(k.arg is None for k in node.keywords):
                dynamic.append(
                    {
                        "registry": registry,
                        "source": relative,
                        "line": node.lineno,
                        "expression": expression(node),
                        "conditions": guards,
                        "members": [],
                    }
                )
                return
            kws = {k.arg: k.value for k in node.keywords}
            name = kws.get("name", node.args[0] if node.args else None)
            if isinstance(name, ast.Constant) and name.value is None:
                name = None
            module = kws.get("module")
            # A bare register_module(name=...) only creates a decorator; no
            # member has been registered until that decorator is applied.
            if module is None and default is None:
                dynamic.append(
                    {
                        "registry": registry,
                        "source": relative,
                        "line": node.lineno,
                        "expression": expression(node),
                        "conditions": guards,
                        "members": [],
                        "reason": "unapplied registration decorator",
                    }
                )
                return
            result = []
            if name is not None:
                for value in values(name, env):
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        result.append(value.value)
                    elif isinstance(value, ast.Constant) and value.value is None and default:
                        result.append(default)
                    else:
                        result = []
                        break
            elif default:
                result = [default]
            elif module is not None:
                for value in values(module, env):
                    if isinstance(value, ast.Attribute):
                        result.append(value.attr)
                    elif isinstance(value, ast.Name) and value.id in symbols:
                        result.append(symbols[value.id])
                    else:
                        result = []
                        break
            data = {
                "registry": registry,
                "source": relative,
                "line": node.lineno,
                "expression": expression(node),
                "conditions": guards,
                "members": result,
            }
            if not result:
                dynamic.append(data)
            else:
                rows.append(data)

        def walk(node, env, guards):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    call(dec, env, guards, node.name)
                for child in node.body:
                    walk(child, env, guards)
                return
            if isinstance(node, ast.For):
                child = dict(env)
                if isinstance(node.target, ast.Name):
                    child[node.target.id] = (
                        values(node.iter, env)
                        if isinstance(node.iter, (ast.List, ast.Tuple))
                        else [node.iter]
                    )
                mark = guards + [
                    {"kind": "for", "line": node.lineno, "expression": expression(node.iter)}
                ]
                for item in node.body:
                    walk(item, child, mark)
                for item in node.orelse:
                    walk(item, env, mark)
                return
            if isinstance(node, ast.If):
                for branch, body in ((True, node.body), (False, node.orelse)):
                    for item in body:
                        walk(
                            item,
                            env,
                            guards
                            + [
                                {
                                    "kind": "if",
                                    "line": node.lineno,
                                    "branch": branch,
                                    "expression": expression(node.test),
                                }
                            ],
                        )
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    call(decorator, env, guards, node.name)
                for item in node.body:
                    walk(
                        item,
                        env,
                        guards + [{"kind": "function", "line": node.lineno, "name": node.name}],
                    )
                return
            if isinstance(node, ast.Try):
                for kind, body in (
                    [("try", node.body)]
                    + [("except", h.body) for h in node.handlers]
                    + [("else", node.orelse), ("finally", node.finalbody)]
                ):
                    for item in body:
                        walk(item, env, guards + [{"kind": kind, "line": node.lineno}])
                return
            if isinstance(node, ast.Call):
                call(node, env, guards)
            for child in ast.iter_child_nodes(node):
                walk(child, env, guards)

        walk(tree, {}, [])
    return {
        "declarations": rows,
        "unresolved": dynamic,
        "out_of_scope": other,
        "axes": {
            axis: sorted(
                {member for row in rows if row["registry"] == registry for member in row["members"]}
            )
            for registry, axis in AXES[repo].items()
        },
    }
