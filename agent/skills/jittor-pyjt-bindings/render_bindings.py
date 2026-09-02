#!/usr/bin/env python3
"""Render the pyjt bindings for a few headers without building jittor.

`pyjt_compiler.py` only needs `jittor_utils`, not the compiled core, so the
generator can be run on its own.  Point this at two checkouts (or at the same
one before and after an edit) and diff the output to see exactly what a change
to the generator does to the emitted C++ -- seconds instead of a rebuild.

    python render_bindings.py <repo>/python <out_dir> [<gen_dir>]

<gen_dir> is a cache `gen/` directory; if given, `jit_op_maker.h` is taken from
there and prefixed with `var_holder.h`, the way compiler.py does it, so the
Var methods and the type object end up in the output too.  Find one with:

    find "$JITTOR_HOME/.cache/jittor" -name jit_op_maker.h
"""
import importlib.util
import os
import sys


def load_generator(python_dir):
    sys.path.insert(0, python_dir)
    path = os.path.join(python_dir, "jittor", "pyjt_compiler.py")
    spec = importlib.util.spec_from_file_location("_pyjt_compiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    python_dir = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    gen_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None

    pyjt_compiler = load_generator(python_dir)
    src_root = os.path.join(python_dir, "jittor")
    os.makedirs(out, exist_ok=True)

    headers = []
    for dirpath, _, filenames in os.walk(os.path.join(src_root, "src")):
        for name in filenames:
            if name.endswith(".h"):
                headers.append((os.path.join(dirpath, name), None))
    if gen_dir:
        # compiler.py compiles jit_op_maker.h together with var_holder.h
        headers = [(h, p) for h, p in headers
                   if os.path.basename(h) != "var_holder.h"]
        headers.append((os.path.join(gen_dir, "jit_op_maker.h"),
                        os.path.join(src_root, "src", "var_holder.h")))

    written = 0
    for header, prefix in sorted(headers):
        if not os.path.exists(header):
            continue
        with open(header, encoding="utf8") as f:
            src = f.read()
        if "@pyjt" not in src and prefix is None:
            continue
        if prefix:
            with open(prefix, encoding="utf8") as f:
                src = f.read() + src
        base = os.path.basename(header).split(".")[0]
        dst = os.path.join(out, "pyjt_" + base + ".cc")
        if pyjt_compiler.compile_single(header, dst, src):
            written += 1
    print(f"wrote {written} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
