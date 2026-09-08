# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Bounds on the jit kernel search.

The searcher compiles and times one kernel per combination of the candidates a
tuner offers, so the cost is the product of the per-key choice counts.
ReorderTuner offered a choice for every range, which makes that product N!:
at ten ranges plus a few splits it is over three billion kernels.  Nothing
stopped it -- ``Searcher::timeout`` existed but no code read it.
"""
import re
import time
import unittest

import jittor as jt
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser


class TestJitSearchBounds(unittest.TestCase):
    def test_reorder_candidates_are_bounded(self):
        """Ten dimensions plus three splits is thirteen ranges.

        Unbounded that is order0..order12, a product of 3.6e9 combinations.
        """
        a = jt.ones([2] * 10)
        a.sync()
        options = {"split0": 2, "split1": 2, "split2": 2, "_search_bounds": 1}
        with jt.flag_scope(use_cuda=0):
            with jt.log_capture_scope(log_v=0, log_vprefix="tuner_manager=100",
                                      compile_options=options) as logs:
                (a + a).sync()
        found = find_log_with_re(
            logs, "Run tuner reorder: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(found) == 1, logs
        candidates = simple_parser(found[0][1])
        product = 1
        for choices in candidates.values():
            product *= len(choices)
        # the keys it stops at, spelled out so a change of bound is visible.
        # Unbounded this is order0..order12.
        assert sorted(candidates) == ["order%d" % i for i in range(6)], \
            (sorted(candidates), product)
        limit = getattr(jt.flags, "jit_search_max_candidates", None)
        assert limit, "flag jit_search_max_candidates is missing"
        assert product <= limit, (product, limit, sorted(candidates))

    def test_search_timeout_stops_the_search(self):
        """``jit_search_timeout`` is read.

        Each candidate compiles a fresh kernel, which is far more than a second,
        so a one second budget stops after the first one.  The unique tag keeps
        the JIT cache from turning those compiles into no-ops.
        """
        tag = int(time.time()) % 1000000
        a = jt.ones((80, 80, 80))
        a.sync()
        with jt.flag_scope(use_cuda=0):
            with jt.log_capture_scope(
                    log_v=0, log_vprefix="jit_searcher=1000",
                    jit_search_kernel=1, jit_search_timeout=1,
                    compile_options={"compile_shape": 1, "_search_bounds": tag}) as logs:
                (a + a).sync()
        tried = find_log_with_re(
            logs, r"Choices\(([0-9.eE+-]+)ms, best [0-9.eE+-]+\) (.*)$")
        stopped = [l["msg"] for l in logs if "search stopped after" in l["msg"]]
        assert stopped, ("the search ran all %d candidates without stopping"
                         % len(tried))
        matched = re.search(r"stopped after (\d+) of (\d+)", stopped[0])
        assert matched, stopped[0]
        assert int(matched.group(1)) == len(tried), (matched.groups(), len(tried))
        assert int(matched.group(1)) < int(matched.group(2)), matched.groups()


if __name__ == "__main__":
    unittest.main()
