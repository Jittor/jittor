"""Read bounded vLLM registries without importing downstream code.

Membership is a source declaration, never evidence of a runnable engine. Runtime
plugins, conditional registrations and inherited vLLM models remain explicit.
"""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


VLLM_MODELS = 'vllm/model_executor/models/registry.py'
OMNI_MODELS = 'vllm_omni/model_executor/models/registry.py'
OMNI_DIFFUSION = 'vllm_omni/diffusion/registry.py'
OMNI_PIPELINES = 'vllm_omni/config/pipeline_registry.py'


def _value(node, names):
    """Interpret only literal registry expressions and filtered .items() maps."""
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError('Unresolved registry name: ' + node.id)
        return names[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_value(item, names) for item in node.elts)
    if isinstance(node, ast.Dict):
        result, explicit = {}, set()
        for key, value in zip(node.keys, node.values):
            if key is None:
                unpack = _value(value, names)
                if not isinstance(unpack, dict):
                    raise ValueError('Registry unpacking requires a mapping')
                result.update(unpack)
            else:
                key = _value(key, names)
                if key in explicit:
                    raise ValueError('Duplicate explicit registry key: ' + str(key))
                explicit.add(key)
                result[key] = _value(value, names)
        return result
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return _value(node.left, names) == _value(node.comparators[0], names)
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        call = generator.iter
        if (generator.is_async or not isinstance(call, ast.Call) or call.args or call.keywords
                or not isinstance(call.func, ast.Attribute) or call.func.attr != 'items'):
            raise ValueError('Unsupported registry comprehension')
        mapping = _value(call.func.value, names)
        if not isinstance(mapping, dict):
            raise ValueError('Registry comprehension requires a mapping')

        def bind(target, value, local):
            if isinstance(target, ast.Name):
                local[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
                for item, part in zip(target.elts, value):
                    bind(item, part, local)
            else:
                raise ValueError('Unsupported registry comprehension target')

        result = {}
        for pair in mapping.items():
            local = dict(names)
            bind(generator.target, pair, local)
            if all(_value(condition, local) for condition in generator.ifs):
                result[_value(node.key, local)] = _value(node.value, local)
        return result
    raise ValueError('Dynamic/unsupported registry expression: ' + ast.dump(node, include_attributes=False))


def _mapping(tree, name, symbolic_values=False):
    assignments = []
    for item in ast.walk(tree):
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                assignments.append(item)
    if len(assignments) != 1 or assignments[0] not in tree.body:
        raise ValueError('Conditional, repeated or missing registry declaration: ' + name)
    names = {}
    found = []
    for node in tree.body:
        if found:
            break
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == name:
                if symbolic_values:
                    if not isinstance(node.value, ast.Dict) or any(k is None for k in node.value.keys):
                        raise ValueError('Dynamic pipeline registry')
                    keys = [_value(key, {}) for key in node.value.keys]
                    if len(keys) != len(set(keys)):
                        raise ValueError('Duplicate pipeline registry key')
                    result = dict(zip(keys, [ast.unparse(value) for value in node.value.values]))
                else:
                    result = _value(node.value, names)
                found.append((node, result))
            if isinstance(node.value, ast.Dict) and target.id.endswith('_MODELS'):
                names[target.id] = _value(node.value, names)
    if len(found) != 1 or not isinstance(found[0][1], dict) or not found[0][1]:
        raise ValueError('Expected one nonempty registry: ' + name)
    # Do not overlook a second declaration hidden in an import/device branch.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node not in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                raise ValueError('Conditional or nested replacement of registry: ' + name)
    return found[0]


def declarations(root, files, repo):
    """Return separate bounded axes; ``repo`` is vllm or vllm-omni."""
    if repo == 'vllm':
        specs = [(VLLM_MODELS, '_VLLM_MODELS', 'model_family', False)]
        prefix = 'vllm/'
    elif repo == 'vllm-omni':
        specs = [(OMNI_MODELS, '_OMNI_MODELS', 'model_family', False),
                 (OMNI_DIFFUSION, '_DIFFUSION_MODELS', 'diffusion_model', False),
                 (OMNI_PIPELINES, 'OMNI_PIPELINES', 'pipeline_model_type', True)]
        prefix = 'vllm_omni/'
    else:
        raise ValueError('Unsupported vLLM repository: ' + repo)
    files = set(files)
    for path, *_ in specs:
        if path not in files:
            raise ValueError('Missing upstream registry: ' + path)
    selected = {path for path, *_ in specs}
    if repo == 'vllm':
        selected.update(path for path in files if path.startswith('vllm/model_executor/layers/')
                        and path.endswith('.py'))
    trees = {path: ast.parse((root / path).read_text(encoding='utf-8'), filename=path)
             for path in sorted(selected)}
    axes, rows, dynamic, missing = {}, [], [], []
    used_sources = set(trees)

    def add(axis, member, source, symbol, line, conditions=()):
        if not isinstance(member, str) or not member:
            raise ValueError('Invalid registry member in ' + source)
        if member in axes.setdefault(axis, []):
            raise ValueError('Duplicate registry member: ' + axis + '/' + member)
        axes[axis].append(member)
        rows.append({'axis': axis, 'member': member, 'source': source,
                     'symbol': symbol, 'line': line, 'conditions': list(conditions),
                     'target_source': source, 'definition_checked': axis == 'custom_op'})

    def target_path(module):
        stem = module.replace('.', '/')
        return next((path for path in (stem + '.py', stem + '/__init__.py') if path in files), None)

    for path, name, axis, symbolic in specs:
        node, mapping = _mapping(trees[path], name, symbolic)
        for member, target in mapping.items():
            if not symbolic and (not isinstance(target, tuple)
                                 or len(target) != (2 if repo == 'vllm' else 3)
                                 or not all(isinstance(part, str) and part for part in target)):
                raise ValueError('Invalid model registration: ' + member)
            add(axis, member, path, target if symbolic else ':'.join(target), node.lineno)
            if symbolic:
                imports = {alias.asname or alias.name: item.module for item in trees[path].body
                           if isinstance(item, ast.ImportFrom) and not item.level for alias in item.names}
                module = imports.get(target)
                resolved = target_path(module) if module else path
            elif repo == 'vllm':
                module = target[0] if target[0].startswith('vllm.') else 'vllm.model_executor.models.' + target[0]
                resolved = target_path(module)
            else:
                module = ('vllm_omni.diffusion.models.' if axis == 'diffusion_model'
                          else 'vllm_omni.model_executor.models.') + target[0] + '.' + target[1]
                resolved = target_path(module)
            rows[-1]['target_source'] = resolved
            rows[-1]['definition_checked'] = False
            if resolved:
                used_sources.add(resolved)
            else:
                missing.append({'axis': axis, 'member': member, 'symbol': rows[-1]['symbol']})
    for path, tree in trees.items():
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

        def context(node):
            result = []
            while node in parents:
                child, node = node, parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.append('function:' + node.name)
                elif isinstance(node, ast.If):
                    condition = ast.unparse(node.test)
                    result.append('conditional:' + ('not (' + condition + ')' if child in node.orelse else condition))
            return list(reversed(result))

        if repo == 'vllm' and path.startswith('vllm/model_executor/layers/'):
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == 'register'
                            and isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == 'CustomOp'):
                        if len(decorator.args) != 1:
                            raise ValueError('Unresolved CustomOp decorator in ' + path)
                        add('custom_op', _value(decorator.args[0], {}), path,
                            node.name, decorator.lineno, context(node))
        literal_maps = {name for source, name, _, _ in specs if source == path}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                mutations = [target for target in targets if isinstance(target, ast.Subscript)
                             and isinstance(target.value, ast.Name) and target.value.id in literal_maps]
                if mutations and not any(part.startswith('function:') for part in context(node)):
                    raise ValueError('Module-level registry mutation needs an explicit extractor in ' + path)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in literal_maps
                    and node.func.attr in ('update', 'setdefault', 'pop', 'clear')
                    and not any(part.startswith('function:') for part in context(node))):
                raise ValueError('Module-level registry mutation needs an explicit extractor in ' + path)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                    'register_model', 'register_pipeline'):
                dynamic.append({'source': path, 'line': node.lineno, 'kind': 'runtime_api',
                                'call': node.name, 'context': context(node)})
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
                if name in ('register_model', 'register_pipeline', 'load_general_plugins'):
                    dynamic.append({'source': path, 'line': node.lineno, 'kind': 'runtime_call',
                                    'call': ast.unparse(node), 'context': context(node)})
    return {'axes': {axis: sorted(members) for axis, members in sorted(axes.items())},
            'declarations': sorted(rows, key=lambda row: (row['axis'], row['member'])),
            'sources': sorted(used_sources), 'unresolved': [], 'dynamic_sites': dynamic,
            'missing_targets': missing,
            'inherited_registry': 'vllm:_VLLM_MODELS (requires a separately pinned vllm inventory)'
            if repo == 'vllm-omni' else None,
            'scope': prefix + ' source-declared built-in registries only; axes are not additive. '
            'Runtime plugins and registrations are open-ended and excluded from these bounded denominators. '
            'CustomOp declarations describe layer components, not model or engine execution; '
            'enumeration establishes no device support, model weights, numerical parity or performance.'}


def inventory(repo, checkout):
    """Bind the host-only declarations to a clean upstream commit and sources."""
    root = Path(checkout).expanduser().resolve()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

    sha = git('rev-parse', 'HEAD')
    if not re.fullmatch('[0-9a-f]{40}', sha):
        raise ValueError('Expected a full upstream SHA')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Dirty upstream checkout; denominator does not match its SHA')
    value = declarations(root, git('ls-files', '--', '*.py').splitlines(), repo)
    value.update({'schema_version': 1, 'repo': repo, 'upstream_sha': sha,
                  'upstream_remote': git('remote', 'get-url', 'origin'),
                  'source_sha256': {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                                    for path in value['sources']}})
    return value


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', choices=('vllm', 'vllm-omni'), required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[4]
    checkout = args.checkout.expanduser().resolve()
    if any(destination == root or root in destination.parents for root in (source_root, checkout)):
        raise ValueError('Write generated inventories outside both source checkouts')
    value = inventory(args.repo, checkout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'repo': value['repo'], 'upstream_sha': value['upstream_sha'],
                      'axes': {axis: len(members) for axis, members in value['axes'].items()}}))


if __name__ == '__main__':
    main()
