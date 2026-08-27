"""Parser helpers for tuner diagnostic output."""


def simple_parser(value):
    """Turn a ``{name:value, ...}`` tuner line into a dict.

    Splitting the whole line on ``:`` and quoting each fragment breaks on the
    compile-flag entries the compiler adds, whose names contain both a colon and
    spaces -- ``{FLAGS: -O3 :1, order0:0, ...}`` becomes
    ``{"FLAGS": "-O3" :1, ...}``, which is not valid Python. Values here are
    always numbers or bracketed lists, so split each entry at its *last* colon
    and keep everything before it as the name.
    """
    body = value.strip()
    if body.startswith("{"):
        body = body[1:]
    if body.endswith("}"):
        body = body[:-1]

    entries = []
    depth = 0
    current = ""
    for character in body:
        if character in "[{(":
            depth += 1
        elif character in "]})":
            depth -= 1
        if character == "," and depth == 0:
            entries.append(current)
            current = ""
            continue
        current += character
    entries.append(current)

    parsed = {}
    for entry in entries:
        if not entry.strip():
            continue
        name, _, raw = entry.rpartition(":")
        parsed[name.strip()] = eval(raw.strip())
    return parsed


def tuner_choices(value):
    """The tuner's own choices, without the compile flags it also records.

    ``FLAGS:``-prefixed entries carry compiler switches an operator asked for
    (``-O3`` to keep float32 comparisons IEEE, for instance). They are not
    search dimensions, so a test asserting on what the search picked should not
    have to enumerate them.
    """
    return {
        name: choice
        for name, choice in simple_parser(value).items()
        if not name.startswith("FLAGS:")
    }
