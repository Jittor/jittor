"""Standard-library tests for the wheel contents gate."""

from __future__ import print_function

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


_SCRIPT = Path(__file__).resolve().with_name("check_wheel_contents.py")
_SPEC = importlib.util.spec_from_file_location("_check_wheel_contents_test", _SCRIPT)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


class TestWheelContents(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.base_members = {
            checker.REQUIRED_MEMBERS[0]: b"unpack raw\n",
            checker.REQUIRED_MEMBERS[1]: b"flash attention\n",
            "jittor/base.py": b"BASE = 1\n",
        }

    @staticmethod
    def _digest(content):
        return hashlib.sha256(content).hexdigest()

    def _wheel(self, name, members):
        path = self.root / name
        items = members.items() if isinstance(members, dict) else members
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as archive:
                for member, content in items:
                    archive.writestr(member, content)
        return path

    def _hashed_list(self, name, members):
        path = self.root / name
        lines = ["# candidate SHA-256 and exact wheel path"]
        lines.extend(
            "{}  {}".format(self._digest(content), member)
            for member, content in sorted(members.items())
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _run(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = checker.main([str(argument) for argument in arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_hash_manifest_accepts_an_unchanged_wheel(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        candidate = self._wheel("candidate.whl", self.base_members)
        baseline = self.root / "baseline.txt"

        status, _stdout, stderr = self._run(
            ["manifest", old_wheel, "--output", baseline]
        )
        self.assertEqual(status, 0, stderr)

        entries = checker._read_hashed_path_list(baseline, "test baseline")
        self.assertEqual(len(entries), len(self.base_members))

        status, stdout, stderr = self._run(
            ["compare", candidate, "--baseline", baseline]
        )
        self.assertEqual(status, 0, stderr)

        self.assertIn("content changed: 0", stdout)

    def test_default_stage_allowlists_pin_candidate_hashes(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        baseline = self.root / "baseline.txt"
        self.assertEqual(
            self._run(["manifest", old_wheel, "--output", baseline])[0], 0
        )

        changed_members = dict(self.base_members)
        changed_members["jittor/base.py"] = b"BASE = 2\n"
        changed_members["jittor/new.py"] = b"NEW = True\n"
        candidate = self._wheel("candidate.whl", changed_members)
        additions = self._hashed_list(
            "additions.txt", {"jittor/new.py": changed_members["jittor/new.py"]}
        )
        content_changes = self._hashed_list(
            "content-changes.txt",
            {"jittor/base.py": changed_members["jittor/base.py"]},
        )

        patches = (
            mock.patch.object(checker, "DEFAULT_BASELINE", baseline),
            mock.patch.object(checker, "DEFAULT_ADDITION_ALLOWLIST", additions),
            mock.patch.object(
                checker, "DEFAULT_CONTENT_CHANGE_ALLOWLIST", content_changes
            ),
        )
        with patches[0], patches[1], patches[2]:
            status, stdout, stderr = self._run(["compare", candidate])
            self.assertEqual(status, 0, stderr)
            self.assertIn("added: 1 (approved: 1, unexpected: 0)", stdout)
            self.assertIn("content changed: 1 (approved: 1, unexpected: 0)", stdout)

            reverted = self._wheel("reverted.whl", self.base_members)
            status, _stdout, stderr = self._run(["compare", reverted])
            self.assertEqual(status, 1)
            self.assertIn("approved wheel addition is absent", stderr)
            self.assertIn("approved wheel content change is not present", stderr)

            tampered_members = dict(changed_members)
            tampered_members["jittor/new.py"] = b"tampered addition\n"
            tampered = self._wheel("tampered-addition.whl", tampered_members)
            status, _stdout, stderr = self._run(["compare", tampered])
            self.assertEqual(status, 1)
            self.assertIn("addition hash does not match approval", stderr)

            tampered_members = dict(changed_members)
            tampered_members["jittor/base.py"] = b"tampered change\n"
            tampered = self._wheel("tampered-change.whl", tampered_members)
            status, _stdout, stderr = self._run(["compare", tampered])
            self.assertEqual(status, 1)
            self.assertIn("content-change hash does not match approval", stderr)

    def test_custom_old_wheel_supports_explicit_hashed_allowlists(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        changed_members = dict(self.base_members)
        changed_members["jittor/base.py"] = b"BASE = approved\n"
        changed_members["jittor/approved.py"] = b"APPROVED = True\n"
        candidate = self._wheel("candidate.whl", changed_members)
        additions = self._hashed_list(
            "additions.txt",
            {"jittor/approved.py": changed_members["jittor/approved.py"]},
        )
        content_changes = self._hashed_list(
            "content.txt", {"jittor/base.py": changed_members["jittor/base.py"]}
        )

        status, _stdout, stderr = self._run(
            [
                "compare",
                candidate,
                "--old-wheel",
                old_wheel,
                "--addition-allowlist",
                additions,
                "--content-change-allowlist",
                content_changes,
            ]
        )
        self.assertEqual(status, 0, stderr)

    def test_explicit_stage_hash_supersedes_the_default_stage_hash(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        baseline = self.root / "baseline.txt"
        self.assertEqual(
            self._run(["manifest", old_wheel, "--output", baseline])[0], 0
        )

        stage_one = self._hashed_list(
            "stage-one.txt", {"jittor/base.py": b"BASE = stage_one\n"}
        )
        stage_two_members = dict(self.base_members)
        stage_two_members["jittor/base.py"] = b"BASE = stage_two\n"
        stage_two = self._hashed_list(
            "stage-two.txt", {"jittor/base.py": stage_two_members["jittor/base.py"]}
        )
        candidate = self._wheel("candidate.whl", stage_two_members)

        patches = (
            mock.patch.object(checker, "DEFAULT_BASELINE", baseline),
            mock.patch.object(checker, "DEFAULT_ADDITION_ALLOWLIST", self._hashed_list(
                "empty-additions.txt", {}
            )),
            mock.patch.object(checker, "DEFAULT_CONTENT_CHANGE_ALLOWLIST", stage_one),
        )
        with patches[0], patches[1], patches[2]:
            status, stdout, stderr = self._run(
                ["compare", candidate, "--content-change-allowlist", stage_two]
            )
        self.assertEqual(status, 0, stderr)
        self.assertIn("content changed: 1 (approved: 1, unexpected: 0)", stdout)

    def test_unreviewed_secret_env_addition_fails(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        members = dict(self.base_members)
        members["jittor/secret.env"] = b"TOKEN=not-secret-test-data\n"
        candidate = self._wheel("candidate.whl", members)

        status, _stdout, stderr = self._run(
            ["compare", candidate, "--old-wheel", old_wheel]
        )
        self.assertEqual(status, 1)
        self.assertIn("added without approval: jittor/secret.env", stderr)

    def test_unpack_content_tampering_fails(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        members = dict(self.base_members)
        members[checker.REQUIRED_MEMBERS[0]] = b"tampered\n"
        candidate = self._wheel("candidate.whl", members)

        status, _stdout, stderr = self._run(
            ["compare", candidate, "--old-wheel", old_wheel]
        )
        self.assertEqual(status, 1)
        self.assertIn("content changed without approval", stderr)
        self.assertIn(checker.REQUIRED_MEMBERS[0], stderr)

    def test_removal_requires_an_explicit_allowance(self):
        old_wheel = self._wheel("old.whl", self.base_members)
        members = dict(self.base_members)
        del members["jittor/base.py"]
        candidate = self._wheel("candidate.whl", members)

        status, _stdout, stderr = self._run(
            ["compare", candidate, "--old-wheel", old_wheel]
        )
        self.assertEqual(status, 1)
        self.assertIn("removed without approval", stderr)

        status, _stdout, stderr = self._run(
            [
                "compare",
                candidate,
                "--old-wheel",
                old_wheel,
                "--allow-removal",
                "jittor/base.py",
            ]
        )
        self.assertEqual(status, 0, stderr)

        unchanged = self._wheel("unchanged.whl", self.base_members)
        status, _stdout, stderr = self._run(
            [
                "compare",
                unchanged,
                "--old-wheel",
                old_wheel,
                "--allow-removal",
                "jittor/base.py",
            ]
        )
        self.assertEqual(status, 1)
        self.assertIn("approved wheel removal is not present", stderr)

    def test_missing_required_member_fails(self):
        members = dict(self.base_members)
        del members[checker.REQUIRED_MEMBERS[0]]
        candidate = self._wheel("candidate.whl", members)
        reference = self._wheel("reference.whl", members)

        status, _stdout, stderr = self._run(
            ["compare", candidate, "--old-wheel", reference]
        )
        self.assertEqual(status, 1)
        self.assertIn("required wheel member is missing", stderr)

    def test_duplicate_traversal_and_pollution_are_rejected(self):
        cases = (
            (
                "duplicate.whl",
                list(self.base_members.items()) + [("jittor/base.py", b"again\n")],
                "duplicate member",
            ),
            (
                "traversal.whl",
                list(self.base_members.items()) + [("../escape.py", b"escape\n")],
                "parent traversal member",
            ),
            (
                "pollution.whl",
                list(self.base_members.items())
                + [("jittor/__pycache__/bad.pyc", b"bytecode\n")],
                "polluting wheel member",
            ),
        )
        reference = self._wheel("reference.whl", self.base_members)
        for name, members, message in cases:
            with self.subTest(name=name):
                candidate = self._wheel(name, members)
                status, _stdout, stderr = self._run(
                    ["compare", candidate, "--old-wheel", reference]
                )
                self.assertEqual(status, 1)
                self.assertIn(message, stderr)


if __name__ == "__main__":
    unittest.main()
