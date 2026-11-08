"""Enumerate TRELLIS.2 declared public namespaces without importing Torch.

Lazy export maps are separate axes, not a summed whole-library support rate.
Exported modules, pretrained loaders and sampler constructors are retained on
separate axes. Dependencies and hardware availability never remove declarations.
"""

import ast
from pathlib import PurePosixPath


def declarations(root, files):
    files = sorted(path for path in files if path.startswith('trellis2/') and path.endswith('.py'))
    trees = {path: ast.parse((root / path).read_text(encoding='utf-8'), filename=path) for path in files}
    axes = {}
    rows = []
    missing_definitions = []
    used = set()

    def literal_assignment(path, name):
        found = []
        for node in trees[path].body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                if isinstance(node.value, ast.Dict):
                    keys = [ast.literal_eval(key) for key in node.value.keys]
                    if len(keys) != len(set(keys)):
                        raise ValueError('Duplicate TRELLIS export key in ' + path)
                try:
                    found.append(ast.literal_eval(node.value))
                except (ValueError, TypeError) as error:
                    raise ValueError('Dynamic TRELLIS declaration %s in %s' % (name, path)) from error
        if len(found) != 1:
            raise ValueError('Expected one TRELLIS %s in %s' % (name, path))
        return found[0]

    def module_path(path, module, level):
        parts = list(PurePosixPath(path).parent.parts)
        if level:
            parts = parts[:len(parts) - level + 1]
        else:
            parts = []
        if module:
            parts += module.split('.')
        stem = '/'.join(parts)
        for candidate in (stem + '.py', stem + '/__init__.py'):
            if candidate in trees:
                return candidate
        return None

    def resolve(path, name, seen=()):
        if (path, name) in seen:
            return None
        seen = (*seen, (path, name))
        tree = trees[path]
        if any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name for node in tree.body):
            used.add(path)
            return path
        candidates = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                target = module_path(path, node.module, node.level)
                if target:
                    for alias in node.names:
                        if alias.name == '*' or (alias.asname or alias.name) == name:
                            resolved = resolve(target, name if alias.name == '*' else alias.name, seen)
                            if resolved:
                                candidates.append(resolved)
                                used.add(path)
        if len(set(candidates)) > 1:
            raise ValueError('Ambiguous TRELLIS export %s in %s' % (name, path))
        return candidates[0] if candidates else None

    def add(axis, member, source, symbol):
        if member in axes.setdefault(axis, []):
            raise ValueError('Duplicate TRELLIS declaration %s/%s' % (axis, member))
        axes[axis].append(member)
        rows.append({'axis': axis, 'member': member, 'source': source, 'symbol': symbol})
        used.add(source)

    for path, tree in trees.items():
        if not path.endswith('/__init__.py'):
            continue
        if not any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__attributes' for t in node.targets) for node in tree.body):
            continue
        namespace = path[len('trellis2/'):-len('/__init__.py')].replace('/', '.')
        mapping = literal_assignment(path, '__attributes')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Empty/invalid TRELLIS lazy export map: ' + path)
        for name, module in mapping.items():
            if not isinstance(name, str) or not name or name.startswith('_') or not isinstance(module, str):
                raise ValueError('Invalid TRELLIS lazy export in ' + path)
            target = module_path(path, module, 1)
            definition = resolve(target, name) if target else None
            if not definition:
                missing_definitions.append({'axis': namespace, 'member': name,
                                            'source': path, 'module': module})
            add(namespace, name, path, (definition or target or module) + ':' + name)
        modules = literal_assignment(path, '__submodules')
        if not isinstance(modules, list) or not all(isinstance(name, str) and name for name in modules):
            raise ValueError('Invalid TRELLIS submodule declaration in ' + path)
        for name in modules:
            if not module_path(path, name, 1):
                raise ValueError('Unresolved TRELLIS submodule %s in %s' % (name, path))
            add('exported_modules', namespace + '.' + name, path, name)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                add('package_api', namespace + '.' + node.name, path, node.name)
    required = {'models', 'pipelines', 'datasets', 'trainers', 'renderers', 'representations', 'modules.sparse'}
    if not required <= axes.keys():
        raise ValueError('Missing TRELLIS public registries: ' + str(sorted(required - axes.keys())))
    sampler_path = 'trellis2/pipelines/samplers/__init__.py'
    if sampler_path not in trees:
        raise ValueError('Missing TRELLIS sampler exports')
    for node in trees[sampler_path].body:
        if not isinstance(node, ast.ImportFrom):
            raise ValueError('Unrecognized TRELLIS sampler export declaration')
        for alias in node.names:
            if alias.name == '*' or alias.name.startswith('_'):
                raise ValueError('Unresolved TRELLIS sampler export')
            name = alias.asname or alias.name
            definition = resolve(sampler_path, name)
            if not definition:
                raise ValueError('Unresolved TRELLIS sampler: ' + name)
            add('samplers', name, sampler_path, definition + ':' + alias.name)
    return {'axes': {axis: sorted(members) for axis, members in sorted(axes.items())},
            'declarations': sorted(rows, key=lambda row: (row['axis'], row['member'])),
            'sources': sorted(trees), 'unresolved': [], 'missing_definitions': missing_definitions,
            'scope': 'All source-declared lazy public export maps, sampler constructors, exported submodules and package loaders. Separate bounded API axes, not all TRELLIS internals or pretrained model variants; no execution is established by enumeration.'}
