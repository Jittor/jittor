import re
from functools import lru_cache


WINDOWS_PACKED_DATA_PATCH_VERSION = "windows-preprocessed-v1"


_WINDOWS_NODE_CALLBACK_WRAPPERS = {
    "release_forward_liveness": "_jt_release_forward_liveness",
    "release_backward_liveness": "_jt_release_backward_liveness",
    "release_pending_liveness": "_jt_release_pending_liveness",
    "own_forward_liveness": "_jt_own_forward_liveness",
    "own_backward_liveness": "_jt_own_backward_liveness",
    "own_pending_liveness": "_jt_own_pending_liveness",
}


def _split_args(src: str):
    args = []
    start = 0
    depth = 0
    for i, ch in enumerate(src):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(src[start:i])
            start = i + 1
    args.append(src[start:])
    return args


def decode_source(source: str) -> str:
    char_map = {}
    kept_lines = []
    for line in source.splitlines():
        if line.startswith("#define x"):
            _, name, value = line.split(" ", 2)
            char_map[name] = value
            continue
        if line.startswith("#define XX") or line.startswith("#define _XX"):
            continue
        if line.startswith("_P(") and line.endswith(")"):
            kept_lines.append(line[3:-1])
            continue
        kept_lines.append(line)
    body = "\n".join(kept_lines)

    @lru_cache(maxsize=None)
    def decode_expr(expr: str) -> str:
        out = []
        i = 0
        n = len(expr)
        while i < n:
            if expr.startswith("_P(", i):
                depth = 1
                k = i + 3
                while k < n and depth:
                    if expr[k] == "(":
                        depth += 1
                    elif expr[k] == ")":
                        depth -= 1
                    k += 1
                if depth == 0:
                    out.append(decode_expr(expr[i + 3 : k - 1]))
                    i = k
                    continue
            if expr.startswith("XX", i):
                j = i + 2
                while j < n and expr[j].isdigit():
                    j += 1
                k0 = j
                while k0 < n and expr[k0].isspace():
                    k0 += 1
                if k0 < n and expr[k0] == "(":
                    depth = 1
                    k = k0 + 1
                    while k < n and depth:
                        if expr[k] == "(":
                            depth += 1
                        elif expr[k] == ")":
                            depth -= 1
                        k += 1
                    if depth == 0:
                        inner = expr[k0 + 1 : k - 1]
                        out.append(
                            "".join(decode_expr(arg.strip()) for arg in _split_args(inner))
                        )
                        i = k
                        continue
            if expr[i] == "x":
                j = i + 1
                while j < n and expr[j].isdigit():
                    j += 1
                token = expr[i:j]
                if token in char_map:
                    out.append(char_map[token])
                    i = j
                    continue
            out.append(expr[i])
            i += 1
        return "".join(out)

    return decode_expr(body)


def patch_windows_source(source: str) -> str:
    wrappers = (
        "\n"
        "static void _jt_release_forward_liveness(Node* node) { node->release_forward_liveness(); }\n"
        "static void _jt_release_backward_liveness(Node* node) { node->release_backward_liveness(); }\n"
        "static void _jt_release_pending_liveness(Node* node) { node->release_pending_liveness(); }\n"
        "static void _jt_own_forward_liveness(Node* node) { node->own_forward_liveness(); }\n"
        "static void _jt_own_backward_liveness(Node* node) { node->own_backward_liveness(); }\n"
        "static void _jt_own_pending_liveness(Node* node) { node->own_pending_liveness(); }\n\n"
    )
    pattern = re.compile(
        r"\(void\(\s*\*\s*\)\(Node\s*\*\s*\)\)\s*&\s*Node\s*::\s*"
        r"(release_forward_liveness|release_backward_liveness|release_pending_liveness|"
        r"own_forward_liveness|own_backward_liveness|own_pending_liveness)"
    )
    patched = pattern.sub(
        lambda m: _WINDOWS_NODE_CALLBACK_WRAPPERS[m.group(1)],
        source,
    )
    if patched == source:
        return source

    anchor = "static vector < pair < Node * , void( * )(Node * ) >>"
    insert_at = patched.find(anchor)
    if insert_at < 0:
        namespace_match = re.search(r"\bnamespace\s+jittor\s*\{", patched)
        if namespace_match is None:
            return patched
        insert_at = namespace_match.end()
    return patched[:insert_at] + wrappers + patched[insert_at:]


def prepare_windows_source(source: str) -> str:
    return patch_windows_source(decode_source(source))
