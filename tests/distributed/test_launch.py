import unittest

from jittor.distributed.launch import _wait_processes


class _FakeLog:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, codes):
        self.codes = list(codes)
        self.last_code = None
        self.terminated = False

    def poll(self):
        if self.terminated:
            return -15
        if self.codes:
            self.last_code = self.codes.pop(0)
        return self.last_code

    def terminate(self):
        self.terminated = True


class TestDistributedLaunch(unittest.TestCase):
    def test_rank_failure_terminates_pending_siblings(self):
        slow = _FakeProcess([None])
        failed = _FakeProcess([3])
        slow_log = _FakeLog()
        failed_log = _FakeLog()

        return_code = _wait_processes(
            [(slow, slow_log), (failed, failed_log)], poll_interval=0
        )

        self.assertEqual(return_code, 3)
        self.assertTrue(slow.terminated)
        self.assertTrue(slow_log.closed)
        self.assertTrue(failed_log.closed)


if __name__ == "__main__":
    unittest.main()
