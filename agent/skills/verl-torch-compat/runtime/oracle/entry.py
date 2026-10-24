"""Explicit, fail-closed entry for the official Torch PPO driver/Ray workers."""
import json
import os
import runpy
import sys

import sitecustomize as bootstrap


def main():
    # sitecustomize exceptions are normally swallowed by Python. These checks
    # run explicitly, so a partial startup can never execute a worker payload.
    bootstrap._install_runtime_guards()
    bootstrap.require_clean_venv()
    import torch

    bootstrap.require_official_torch(torch)
    arguments = sys.argv[1:]
    while arguments and arguments[0] in ("-u", "-B", "-s"):
        arguments.pop(0)
    if not arguments:
        raise SystemExit("oracle entry requires a script, -m module, or -c program")
    if arguments[0] == "--check-only":
        print(json.dumps({"torch_version": torch.__version__, "torch_origin": torch.__file__,
                          "executable": sys.executable, "jittor_loaded": False,
                          "user_site_disabled": bool(sys.flags.no_user_site)}))
        return
    if arguments[0] == "-m" and len(arguments) >= 2:
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    elif arguments[0] == "-c" and len(arguments) >= 2:
        sys.argv = ["-c"] + arguments[2:]
        exec(compile(arguments[1], "<oracle-command>", "exec"), {"__name__": "__main__"})
    elif arguments[0].startswith("-"):
        raise SystemExit("unsupported oracle Python option: " + arguments[0])
    else:
        sys.argv = arguments
        # Mirror Python's script-directory import behavior without inheriting
        # arbitrary caller paths or another framework's site-packages.
        sys.path.insert(0, os.path.dirname(os.path.abspath(arguments[0])))
        runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    main()
