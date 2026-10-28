"""Public profiler enum discovery used by downstream configuration code."""

import unittest

import torch


class TestProfilerEnums(unittest.TestCase):
    def test_activity_members_have_names_values_and_identity(self):
        activity = torch.profiler.ProfilerActivity
        expected = {"CPU": 0, "XPU": 1, "MTIA": 4, "CUDA": 2, "HPU": 3, "PrivateUse1": 5}
        self.assertEqual(set(activity.__members__), set(expected))
        for name, value in expected.items():
            member = activity.__members__[name]
            self.assertIs(member, getattr(activity, name))
            self.assertEqual(member.name, name)
            self.assertEqual(member.value, value)
            self.assertEqual(activity(value), member)
        # Activity discovery describes enum names, not available device hardware.
        devices = {
            name.lower(): member
            for name, member in activity.__members__.items()
            if name not in ("CPU", "PrivateUse1")
        }
        self.assertIs(devices["cuda"], activity.CUDA)
        self.assertNotIn("cpu", devices)
        self.assertNotIn("privateuse1", devices)

    def test_action_members_and_schedule_share_enum_identity(self):
        action = torch.profiler.ProfilerAction
        expected = {"NONE": 0, "WARMUP": 1, "RECORD": 2, "RECORD_AND_SAVE": 3}
        self.assertEqual(
            {name: member.value for name, member in action.__members__.items()}, expected
        )
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
        for step, name in enumerate(("NONE", "WARMUP", "RECORD", "RECORD_AND_SAVE", "NONE")):
            self.assertIs(schedule(step), action.__members__[name])


if __name__ == "__main__":
    unittest.main()
