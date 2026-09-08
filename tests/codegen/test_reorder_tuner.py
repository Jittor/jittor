# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser, tuner_choices

gid = 0

class TestReorderTuner(unittest.TestCase):
    def test(self):
        a = jt.ones((8,8,8))
        a.data
        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100"
        ) as logs:
            b = a + a
            b.data
        
        logs = find_log_with_re(logs, 
            "Run tuner reorder: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1
        assert logs[0][0] == "1", "confidence of reorder should be 1"
        candidates = simple_parser(logs[0][1])
        assert candidates == {
            "order0":[0,], "order1":[0,1,], "order2":[0,1,2,]
        }

    def test_with_split(self):
        a = jt.ones((8,8,8))
        a.data
        global gid
        gid+=1
        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            compile_options={
                "split0": 4, "split1": 4, "split2": 4,
                "test_reorder_tuner":gid
            }
        ) as logs:
            b = a + a
            b.data
        
        logs = find_log_with_re(logs, 
            "Run tuner reorder: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1
        assert logs[0][0] == "1", "confidence of reorder should be 1"
        candidates = simple_parser(logs[0][1])
        assert candidates == {
            "order0":[0,], "order1":[0,1,], "order2":[0,1,2,], 
            "order3":[0,1,2,], "order4":[0,1,2,], "order5":[0,1,2,], 
        }, candidates

    def test_searcher(self):
        a = jt.ones((80,80,80))
        a.data
        global gid
        gid+=1
        with jt.log_capture_scope(
            log_v=0, log_vprefix="jit_searcher=1000",
            jit_search_kernel=1, 
            compile_options={
                "compile_shape":1,
                "test_reorder_tuner":gid
            }
        ) as logs:
            b = a + a
            b.data
        choices = find_log_with_re(
            logs,
            r"Choices\(([0-9.eE+-]+)ms, best [0-9.eE+-]+\) (.*)$",
        )
        assert len(choices) == 6, (choices, logs)
        measured = [(float(elapsed), tuner_choices(raw))
                    for elapsed, raw in choices]
        observed_orders = set()
        for _, choice in measured:
            assert choice["compile_shape"] == 1
            assert choice["test_reorder_tuner"] == gid
            observed_orders.add(
                (choice["order0"], choice["order1"], choice["order2"])
            )
        assert observed_orders == {
            (0, order1, order2)
            for order1 in range(2)
            for order2 in range(3)
        }

        best_logs = find_log_with_re(
            logs,
            r"Best choices\(([0-9.eE+-]+)ms\): (.*)$",
        )
        assert len(best_logs) == 1
        best_time = float(best_logs[0][0])
        best = tuner_choices(best_logs[0][1])
        chosen_times = [elapsed for elapsed, choice in measured if choice == best]
        assert len(chosen_times) == 1, (best, measured)
        min_time = min(elapsed for elapsed, _ in measured)
        tolerance = max(1e-4, min_time * 1e-5)
        assert abs(chosen_times[0] - min_time) <= tolerance
        assert abs(best_time - chosen_times[0]) <= tolerance



if __name__ == "__main__":
    unittest.main()
