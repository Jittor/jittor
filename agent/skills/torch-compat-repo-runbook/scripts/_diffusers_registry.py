"""Read every conditional public model export without importing Diffusers.

The axis includes helpers, enums, mixins, functions and Flax exports. It is a
bounded model-package API axis, not a model-family or whole-library denominator.
"""
import ast


def model_exports(text):
    tree = ast.parse(text)
    owners, declarations, handled = {}, [], set()
    assigned, branch_names = set(), set()
    initialized = False

    def registry(node):
        return isinstance(node, ast.Name) and node.id == '_import_structure'

    def subscription(node):
        return isinstance(node, ast.Subscript) and registry(node.value)

    def add(target, values, guards, node, extension=False):
        if not initialized:
            raise ValueError('Diffusers model export precedes registry initialization')
        key = target.slice
        if hasattr(ast, 'Index') and isinstance(key, ast.Index):
            key = key.value
        try:
            module, names = ast.literal_eval(key), ast.literal_eval(values)
        except (ValueError, TypeError) as error:
            raise ValueError('Dynamic Diffusers model export at line %d' % node.lineno) from error
        if (not isinstance(module, str) or not module or not isinstance(names, (list, tuple))
                or not names or not all(isinstance(name, str) and name for name in names)
                or len(names) != len(set(names))):
            raise ValueError('Invalid or duplicate Diffusers model exports')
        signature = (module, tuple(names), tuple(guards))
        if any(row['_signature'] == signature for row in declarations):
            raise ValueError('Repeated Diffusers export declaration in the same branch')
        branch = (module, tuple(guards))
        if not extension and branch in assigned:
            raise ValueError('Overwritten Diffusers model export branch')
        assigned.add(branch)
        for name in names:
            if (branch, name) in branch_names:
                raise ValueError('Duplicate Diffusers export in the same branch: ' + name)
            branch_names.add((branch, name))
            if name in owners and owners[name] != module:
                raise ValueError('Ambiguous conditional Diffusers export owner: ' + name)
            owners[name] = module
        declarations.append({'module': module, 'names': list(names), 'line': node.lineno,
                             'guards': list(guards), '_signature': signature})
        handled.add(id(target))

    def walk(body, guards=()):
        nonlocal initialized
        for node in body:
            if isinstance(node, ast.If):
                condition = ast.dump(node.test, include_attributes=False)
                walk(node.body, guards + (condition,))
                walk(node.orelse, guards + ('not ' + condition,))
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1 and any(registry(t) or subscription(t) for t in node.targets):
                    raise ValueError('Aliased Diffusers model registry assignment')
                for target in node.targets:
                    if registry(target):
                        if initialized or guards or not isinstance(node.value, ast.Dict) or node.value.keys:
                            raise ValueError('Dynamic/reset Diffusers model registry')
                        initialized = True
                        handled.add(id(target))
                    elif subscription(target):
                        add(target, node.value, guards, node)
                    elif registry(node.value) or subscription(node.value):
                        raise ValueError('Aliased Diffusers model registry')
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (isinstance(call.func, ast.Attribute) and call.func.attr == 'extend'
                        and subscription(call.func.value) and len(call.args) == 1 and not call.keywords):
                    add(call.func.value, call.args[0], guards, node, extension=True)

    walk(tree.body)
    # Unknown mutations, loops/function-local declarations and aliased/dynamic
    # registration must fail instead of silently shrinking the denominator.
    for node in ast.walk(tree):
        if registry(node) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in handled:
            raise ValueError('Unsupported Diffusers model registry reset')
        if subscription(node) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in handled:
            raise ValueError('Unsupported Diffusers model registry mutation')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if registry(receiver) or (subscription(receiver) and id(receiver) not in handled):
                raise ValueError('Unsupported Diffusers model registry method')
        if (isinstance(node, ast.Call) and any(
                registry(part) for arg in [*node.args, *[k.value for k in node.keywords]] for part in ast.walk(arg))
                and not (isinstance(node.func, ast.Name) and node.func.id == '_LazyModule')):
            raise ValueError('Dynamic Diffusers model registry consumer')
    if not initialized or not owners:
        raise ValueError('Missing Diffusers model exports')
    for row in declarations:
        row.pop('_signature')
    return sorted(owners), declarations
