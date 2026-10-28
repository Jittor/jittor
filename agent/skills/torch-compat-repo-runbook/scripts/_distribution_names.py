"""Read finite setup.py distribution names without importing build dependencies."""

import ast
import hashlib
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _environment_test(node):
    require(isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq)),
            'Unresolved distribution-name condition')
    call = node.left
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == 'os'
            and call.func.attr == 'getenv' and len(call.args) == 2 and not call.keywords,
            'Unresolved distribution-name environment condition')
    try:
        environment, default = (ast.literal_eval(arg) for arg in call.args)
        value = ast.literal_eval(node.comparators[0])
    except (ValueError, TypeError) as error:
        raise ValueError('Nonliteral distribution-name environment condition') from error
    require(all(isinstance(item, str) for item in (environment, default, value))
            and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', environment),
            'Invalid distribution-name environment condition')
    return {'environment': environment, 'default': default,
            'operator': '==' if isinstance(node.ops[0], ast.Eq) else '!=', 'value': value}


def name_cases(expression):
    """Keep every branch; unsupported dynamic names fail rather than disappear."""
    node = ast.parse(expression, mode='eval').body

    def walk(value, conditions):
        if isinstance(value, ast.IfExp):
            condition = _environment_test(value.test)
            return (walk(value.body, conditions + [dict(condition, result=True)])
                    + walk(value.orelse, conditions + [dict(condition, result=False)]))
        try:
            name = ast.literal_eval(value)
        except (ValueError, TypeError) as error:
            raise ValueError('Unresolved distribution name') from error
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?', name),
                'Invalid declared distribution name')
        return [{'name': name, 'conditions': conditions}]

    return walk(node, [])


def setup_distribution_names(checkout):
    path = checkout / 'setup.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename='setup.py')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and (isinstance(node.func, ast.Name) and node.func.id == 'setup'
                  or isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == 'setuptools' and node.func.attr == 'setup')]
    require(len(calls) == 1, 'Expected one unambiguous distribution setup declaration')
    names = [keyword.value for keyword in calls[0].keywords if keyword.arg == 'name']
    require(len(names) == 1 and not any(keyword.arg is None for keyword in calls[0].keywords),
            'Missing or dynamic distribution setup name')
    expression = ast.unparse(names[0])
    return {'source': {'path': 'setup.py', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'expression': expression, 'cases': name_cases(expression),
            'scope': 'All finite setup name branches; build variants do not certify compiled operators'}


def validate_name_declaration(declaration, sources):
    require(isinstance(declaration, dict) and isinstance(declaration.get('source'), dict),
            'Missing source-derived distribution-name declaration')
    source = declaration['source']
    require(source.get('path') == 'setup.py' and isinstance(source.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', source['sha256'])
            and sum(row == source for row in sources) == 1,
            'Distribution-name declaration source hash mismatch')
    require(isinstance(declaration.get('expression'), str), 'Missing distribution-name expression')
    cases = name_cases(declaration['expression'])
    require(declaration.get('cases') == cases, 'Incomplete or mismatched distribution-name branches')
    return sorted({row['name'] for row in cases})
