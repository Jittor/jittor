# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Remove parts of the Jittor cache.

The subcommands are generated from ``jittor_utils.CACHE_GROUPS``, which is the
same description ``find_cache_path()`` builds the tree against. They used to be
a second, hand-written copy of the layout, and it had drifted: two of the five
paths it deleted no longer existed, and a third deleted the bundled CUDA
toolkit as a side effect of cleaning the compiled products.
"""

import os, sys, shutil

import jittor_utils as jit_utils


def callback(func, path, exc_info):
    print("remove \"%s\" failed." % path)


def remove(path):
    if os.path.isdir(path):
        print("remove \"%s\" recursive." % path)
        shutil.rmtree(path, onerror=callback)
    elif os.path.isfile(path):
        print("remove \"%s\"." % path)
        try:
            os.remove(path)
        except OSError:
            callback(None, path, None)


def clean(group):
    """Remove one group. ``all`` is every group plus the root itself."""
    if group == "all":
        remove(jit_utils.cache_root())
        return
    paths = jit_utils.cache_group_paths(group)
    if not paths:
        print("nothing to remove for \"%s\"." % group)
    for path in paths:
        remove(path)


GROUPS = tuple(name for name, _ in jit_utils.CACHE_GROUPS) + ("all",)


def print_help(status=1):
    print("Usage: %s -m jittor_utils.clean_cache [%s]"
          % (sys.executable, "|".join(GROUPS)))
    for name, description in jit_utils.CACHE_GROUPS:
        print("  %-8s %s" % (name, description))
    print("  %-8s %s" % ("all", "the whole cache"))
    sys.exit(status)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # `clean_cache help` is documented in docs/guides/debugging.md, and asking
    # for help is not an error.
    if argv and all(name in ("help", "-h", "--help") for name in argv):
        print_help(0)
    if not argv or any(name not in GROUPS for name in argv):
        print_help()
    for name in argv:
        clean(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
