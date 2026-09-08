# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The jit cache tables must not keep views into storage they do not own.

``utils/jit_cache_map.h`` replaced ``string_view_map``, whose keys were
``string_view``s into a ``vector<string>`` it appended to.  A string of 15
characters or fewer keeps its characters *inside* the string object with
libstdc++'s small-string optimisation, so every reallocation of that vector
moved those characters and freed the block they had been in -- and left every
key already in the hash map pointing into freed memory.

That is not something a comment can be trusted to keep fixed, so it is checked:
the header is compiled on its own under ``-fsanitize=address`` and the scenario
is run.  The second case compiles the pattern that used to be there and asserts
that ASan *does* report it, because a memory checker that would pass either way
proves nothing about the first case.

The header is deliberately dependent on nothing but ``common.h`` and the one
``jit_cache_size`` flag, which is what makes compiling it standalone possible;
the case below supplies that one symbol.  If this file ever stops building,
the reason is almost certainly a new ``#include`` in the header.

Run::  python -m pytest tests/codegen/test_jit_cache_map_asan.py
"""

import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python/jittor/src"

#: The scenario, once against the real header and once against the pattern it
#: replaced.  Short keys and enough of them to force the holder vector to
#: reallocate several times; the churn afterwards reuses and overwrites what
#: was freed, so a stale key reads scrambled bytes instead of bytes that happen
#: to have survived.
CURRENT = r"""
#include <cstdio>
#include <string>
#include <vector>
#include "utils/jit_cache_map.h"

// The header's only symbol dependency; op.cc owns it in a real build.
namespace jittor { int jit_cache_size = 4096; }

int main() {
    jittor::jit_cache_map<int> table;
    table.capacity = 1u << 20;
    const int n = 4096;
    for (int i = 0; i < n; i++)
        table[std::string("k") + std::to_string(i)] = i;
    std::vector<std::vector<char> > churn;
    for (int i = 0; i < 64; i++) churn.push_back(std::vector<char>(1 << 16, (char)0xa5));
    int missing = 0;
    for (int i = 0; i < n; i++) {
        int* found = table.find(std::string("k") + std::to_string(i));
        if (!found || *found != i) missing++;
    }
    printf("entries=%zu missing=%d\n", table.size(), missing);
    jittor::jit_cache_map<int> bounded;
    bounded.capacity = 8;
    for (int i = 0; i < 1000; i++)
        bounded[std::string("k") + std::to_string(i)] = i;
    printf("bounded=%zu\n", bounded.size());
    return (missing != 0 || bounded.size() > 8) ? 1 : 0;
}
"""

PREDECESSOR = r"""
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <experimental/string_view>

using std::string;
using std::vector;
using std::experimental::string_view;

// python/jittor/src/utils/string_view_map.h as of f5a0ec5a1, verbatim.
template<class T>
struct string_view_map {
    typedef typename std::unordered_map<string_view, T> umap_t;
    typedef typename umap_t::iterator iter_t;
    umap_t umap;
    vector<string> holder;

    iter_t find(string_view sv) { return umap.find(sv); }
    iter_t begin() { return umap.begin(); }
    iter_t end() { return umap.end(); }
    const T& at(string_view sv) { return umap.at(sv); }
    size_t size() { return umap.size(); }

    T& operator[](string_view sv) {
        auto iter = find(sv);
        if (iter != end()) return iter->second;
        holder.emplace_back(sv);
        string_view nsv = holder.back();
        return umap[nsv];
    }
};

int main() {
    string_view_map<int> table;
    const int n = 4096;
    for (int i = 0; i < n; i++)
        table[string("k") + std::to_string(i)] = i;
    int missing = 0;
    for (int i = 0; i < n; i++) {
        auto iter = table.find(string("k") + std::to_string(i));
        if (iter == table.end() || iter->second != i) missing++;
    }
    printf("entries=%zu missing=%d\n", table.size(), missing);
    return 0;
}
"""


def _compile_and_run(source, extra_includes=()):
    """Build `source` with ASan and run it; None if ASan is unavailable."""
    compiler = os.environ.get("CXX", "g++")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        source_path = scratch / "case.cc"
        source_path.write_text(source, encoding="utf-8")
        binary = scratch / "case"
        command = [compiler, "-std=c++14", "-g", "-O0", "-fsanitize=address",
                   str(source_path), "-o", str(binary)]
        for include in extra_includes:
            command += ["-I" + str(include)]
        build = subprocess.run(command, capture_output=True, text=True,
                               timeout=300)
        if build.returncode != 0:
            # No libasan, or no <experimental/string_view>: report that rather
            # than failing, but never quietly pass the first case.
            return None, build.stdout + build.stderr
        # Leaks are not what this looks for, and the predecessor case leaks by
        # construction.
        environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        run = subprocess.run([str(binary)], capture_output=True, text=True,
                             timeout=300, env=environment)
        return run, run.stdout + run.stderr


class TestJitCacheMapUnderAsan(unittest.TestCase):
    def test_the_current_table_is_clean_under_asan(self):
        run, output = _compile_and_run(
            CURRENT, (SRC, sysconfig.get_path("include")))
        if run is None:
            self.skipTest("could not build with -fsanitize=address:\n" + output)
        self.assertNotIn("AddressSanitizer", output)
        self.assertIn("missing=0", output)
        self.assertIn("bounded=8", output)
        self.assertEqual(run.returncode, 0, output)

    def test_the_predecessor_pattern_is_what_asan_catches(self):
        """The check above would pass on a broken table if ASan were inert."""
        run, output = _compile_and_run(PREDECESSOR)
        if run is None:
            self.skipTest("could not build with -fsanitize=address:\n" + output)
        self.assertIn("AddressSanitizer", output)
        self.assertIn("heap-use-after-free", output)
        # ASan stops at the first report, so the counting loop is never
        # reached. Built without the checker the same program reports
        # `missing=2017` of 4096 short keys on this box -- the defect is not
        # only visible to a sanitizer.
        self.assertNotEqual(run.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()
