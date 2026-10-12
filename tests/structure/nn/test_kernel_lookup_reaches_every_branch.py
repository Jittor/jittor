# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""If one arm of a branch asks for an accelerated kernel, its sibling must too.

`conv_transpose` asked `select_kernel("conv_transpose2d", ...)` in its
`groups == 1` branch and not in its grouped one, so every depthwise transposed
convolution took the eight-dimensional reindex lowering while cuDNN's adapter
sat there taking a `groups` argument nobody passed it. Nothing was wrong with
the numbers, so no test caught it; what was wrong was a question never asked.
On the MiniMax-H3 audio VAE that one missed lookup was 79.8% of the decode.

Two earlier drafts of this file got the rule wrong, in opposite directions, and
both failures are worth keeping written down:

*Comparing line numbers* -- "a return must have a lookup above it" -- would
have passed the very defect it is named for. `conv_transpose`'s lookup is in
the `groups == 1` branch, which is written *first*, so the grouped branch's
unguarded `return y` sat below it and looked covered.

*Demanding that a lookup dominate every return* is sound but flags the shape
every dispatch site in the tree legitimately has: `if <the kernel applies>:
result = try_dispatch(...); if result is not None: return result`, then the
fallback. There the un-asking path is the point -- the guard is the caller
saying the kernel does not cover this case. Eleven sites, all correct, and an
exemption list that long is a gate nobody reads.

What distinguishes the defect from the idiom is *a sibling*. A guarded fast
path has no else: control rejoins and the fallback serves both conditions.
`conv_transpose` had two arms doing the same kind of work where only one asked.
So: when an `if` body performs a lookup, its else -- written, or implicit in
the statements after a body that always returns -- must perform one too, or
raise, or hand the work to a function that asks.

`try_dispatch(op, ...)` counts as a lookup: it is `select_kernel` plus the call
(`python/jittor/_runtime/dispatch.py`), and it is how most of the tree spells
this. Excluding it would leave attention, embedding and split outside a scan
that appeared to cover them.

Deliberately structural rather than behavioural: the defect produces correct
output, so only the shape of the code shows it.
"""
import ast
import functools
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3] / "python" / "jittor"

#: The two spellings of "ask the backend whether it has this op".
_LOOKUP_NAMES = frozenset({"select_kernel", "try_dispatch"})

_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


@functools.lru_cache(maxsize=None)
def _parsed(path):
    """One parse of the package per session -- a full pass is ~4s of it."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:  # a file written for a Python this build is not
        return None


def _functions(path):
    tree = _parsed(path)
    return [] if tree is None else [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _is_lookup_call(node):
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Name):
        return function.id in _LOOKUP_NAMES
    if isinstance(function, ast.Attribute):
        return function.attr in _LOOKUP_NAMES
    return False


def _own_nodes(node):
    """Descendants of `node`, stopping at a nested function's boundary.

    A lookup inside a closure says nothing about the enclosing function's own
    branches, and a `return` inside one returns from the closure.
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(child))


def _asks(nodes):
    return any(_is_lookup_call(node) for parent in nodes
               for node in (parent, *_own_nodes(parent)))


def _always_exits(body):
    """Whether every path through `body` returns, raises, or leaves a loop."""
    for statement in body:
        if isinstance(statement, _EXITS):
            return True
        if isinstance(statement, ast.If) and statement.orelse:
            if _always_exits(statement.body) and _always_exits(statement.orelse):
                return True
        if isinstance(statement, (ast.With, ast.AsyncWith)) \
                and _always_exits(statement.body):
            return True
        if isinstance(statement, ast.Try):
            handled = all(_always_exits(handler.body)
                          for handler in statement.handlers)
            if handled and _always_exits(statement.body + statement.orelse):
                return True
            if _always_exits(statement.finalbody):
                return True
    return False


def _only_raises(body):
    """Whether `body` cannot return a value at all -- it exists to reject."""
    if not body:
        return False
    returns = [node for statement in body
               for node in (statement, *_own_nodes(statement))
               if isinstance(node, ast.Return) and node.value is not None]
    return not returns and _always_exits(body)


def _called_names(body):
    for statement in body:
        for node in (statement, *_own_nodes(statement)):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                yield node.func.id
            elif isinstance(node.func, ast.Attribute):
                yield node.func.attr


def _asking_function_names(root):
    """Names of functions that ask, directly or through another that does.

    `_pool2d` hands `op == "mean"` to `avg_pool2d`, which hands it to
    `_avg_pool_nd`, which asks; two links, so a single step would not see it.
    Matching on the bare name across the tree rather than resolving the import
    biases toward missing a defect over inventing one, which is the right
    direction for a gate that has to be believed when it does fire.
    """
    asks, calls = set(), {}
    for path in sorted(root.rglob("*.py")):
        for node in _functions(path):
            if _asks(node.body):
                asks.add(node.name)
            calls.setdefault(node.name, set()).update(_called_names(node.body))
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            if name not in asks and callees & asks:
                asks.add(name)
                changed = True
    return asks


def _sibling_of(statement, following):
    """The else arm of `statement`: written, or implicit after an exiting body.

    With no else and a body that falls through, control rejoins and what
    follows serves both conditions -- that is the guarded fast path, and it has
    no sibling. With a body that always exits, the statements after the `if`
    *are* the else, just spelled without the keyword; the defect reads the same
    either way, so the check has to see through the spelling.
    """
    if statement.orelse:
        return statement.orelse
    # Nothing after an exiting body is not a sibling, it is the end of the
    # function; falling off it is a different bug than this one.
    if following and _always_exits(statement.body):
        return list(following)
    return None


def _blocks(statement):
    """The statement lists `statement` owns, each visited exactly once."""
    for name in ("body", "orelse", "finalbody"):
        block = getattr(statement, name, None)
        if isinstance(block, list):
            yield block
    for handler in getattr(statement, "handlers", None) or []:
        yield handler.body
    for case in getattr(statement, "cases", None) or []:
        yield case.body


def _branches_that_ask_alone(function, asking_names):
    """Every `if` in `function` that asks in one arm and not in its sibling."""
    found = []

    def scan(body):
        for index, statement in enumerate(body):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                continue  # scanned as its own function
            if isinstance(statement, ast.If):
                sibling = _sibling_of(statement, body[index + 1:])
                if _asks(statement.body) and sibling is not None \
                        and not _asks(sibling) \
                        and not _only_raises(sibling) \
                        and not set(_called_names(sibling)) & asking_names:
                    found.append(statement)
            for block in _blocks(statement):
                scan(block)

    scan(function.body)
    return sorted(found, key=lambda node: node.lineno)


class TestKernelLookupReachesEveryBranch(unittest.TestCase):
    def test_no_branch_asks_for_a_kernel_alone(self):
        asking_names = _asking_function_names(_ROOT)
        offenders = []
        for path in sorted(_ROOT.rglob("*.py")):
            for function in _functions(path):
                offenders.extend(
                    "%s:%d %s() asks for a kernel in one arm of this branch "
                    "and not in the other"
                    % (path.relative_to(_ROOT), node.lineno, function.name)
                    for node in _branches_that_ask_alone(function, asking_names))
        self.assertEqual(offenders, [], "\n".join(
            ["one arm of these branches consults the backend and its sibling "
             "silently takes the Python lowering; ask on both arms, or make "
             "the sibling delegate to something that asks:"] + offenders))

    def test_the_check_has_something_to_check(self):
        # A structural test that silently matches nothing passes forever. Both
        # spellings have to be reaching real call sites, the delegation closure
        # has to be finding the indirect askers, and the branch walk has to be
        # reaching conditional lookups -- the shape the check exists to judge.
        conditional = 0
        direct = set()
        for path in sorted(_ROOT.rglob("*.py")):
            for function in _functions(path):
                if _asks(function.body):
                    direct.add(function.name)
                conditional += sum(
                    1 for node in _own_nodes(function)
                    if isinstance(node, ast.If) and _asks(node.body))
        self.assertGreater(len(direct), 20, "no inline lookup call sites were "
                                            "found; the scan is looking in the "
                                            "wrong place")
        self.assertGreater(conditional, 5, "no lookup under a branch was "
                                           "scanned; the check is walking past "
                                           "the shape it exists to judge")
        self.assertGreater(
            len(_asking_function_names(_ROOT) - direct), 5,
            "the delegation closure found no indirect askers, so a sibling "
            "that hands the work to an asking function would be flagged")

    def test_it_catches_the_defect_it_was_written_for(self):
        # `conv_transpose` before the fix, reduced. The lookup is in the first
        # branch and the unguarded return is textually below it, which is why
        # this is not a line-number check.
        written_else = (
            "def conv_transpose(x, w, groups):\n"
            "    if groups == 1:\n"
            "        kernel = select_kernel('conv_transpose2d', x, w)\n"
            "        if kernel is not None:\n"
            "            return kernel(x, w)\n"
            "        return lower(x, w)\n"
            "    else:\n"
            "        return lower_grouped(x, w, groups)\n")
        # The same defect with the `else` left off. The first arm returns on
        # every path, so what follows is its sibling however it is spelled.
        implicit_else = written_else.replace(
            "    else:\n        return lower_grouped(x, w, groups)\n",
            "    return lower_grouped(x, w, groups)\n")
        for label, source in (("written else", written_else),
                              ("implicit else", implicit_else)):
            function, = ast.parse(source).body
            with self.subTest(label):
                self.assertEqual(
                    [node.lineno for node
                     in _branches_that_ask_alone(function, set())], [2],
                    "the grouped branch went unnoticed")

    def test_it_passes_the_guarded_fast_path(self):
        # The idiom, which must not fire: the body falls through, so the
        # fallback below serves both conditions and there is no sibling.
        source = (
            "def var_getitem(x, slices, return_x=None):\n"
            "    if return_x is None:\n"
            "        result = try_dispatch('tensor.getitem', x, slices)\n"
            "        if result is not None:\n"
            "            return result\n"
            "    return _native_var_getitem(x, slices)\n")
        function, = ast.parse(source).body
        self.assertEqual(_branches_that_ask_alone(function, set()), [])

    def test_it_follows_an_elif_chain(self):
        # An `elif` is an `if` in the outer `else`, so a chain where only the
        # first arm asks has to be reported at the arm that is actually wrong
        # -- the innermost one -- and not just at the top.
        source = (
            "def f(x, kind):\n"
            "    if kind == 'a':\n"
            "        k = select_kernel('op', x)\n"
            "        return k(x)\n"
            "    elif kind == 'b':\n"
            "        k = select_kernel('op', x)\n"
            "        return k(x)\n"
            "    else:\n"
            "        return slow(x)\n")
        function, = ast.parse(source).body
        self.assertEqual(
            [node.lineno for node in _branches_that_ask_alone(function, set())],
            [5], "the chain should be clean at the top -- the outer else does "
                 "ask -- and reported at the `elif` whose else does not")

    def test_it_passes_a_sibling_that_delegates(self):
        # `_pool2d` hands `op == "mean"` to `avg_pool2d`, which asks two calls
        # later. Reversed here so the asking arm comes first.
        source = (
            "def _pool2d(x, op):\n"
            "    if op != 'mean':\n"
            "        fast = try_dispatch('nn.pool2d', x, op)\n"
            "        if fast is not None:\n"
            "            return fast\n"
            "        return lower(x, op)\n"
            "    return avg_pool2d(x)\n")
        function, = ast.parse(source).body
        self.assertEqual(
            _branches_that_ask_alone(function, {"avg_pool2d"}), [])
        self.assertEqual(
            [node.lineno for node in _branches_that_ask_alone(function, set())],
            [2], "the delegation allowance is what cleared this, not the walk")


if __name__ == "__main__":
    unittest.main()
