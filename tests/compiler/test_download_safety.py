"""What Jittor does when it fetches a third-party binary.

Three habits this pins down, each of which reached beyond Jittor itself:

* the import used to run ``ssl._create_default_https_context =
  ssl._create_unverified_context``, which disarms certificate verification for
  *every* use of stdlib ssl in the process, not just Jittor's downloads;
* ``tarfile.extractall`` follows ``../`` in member names and writes outside the
  directory it was given;
* a checksum mismatch raised but kept the corrupt file, so every later run
  recomputed the same hash of the same bytes and failed the same way.
"""

import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from jittor_utils import manifest, misc


REPO = Path(__file__).resolve().parents[2]


class TestNoGlobalTlsDowngrade(unittest.TestCase):

    def test_the_repository_does_not_disarm_certificate_verification(self):
        offenders = []
        for path in (REPO / "python").rglob("*.py"):
            text = path.read_text(encoding="utf8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if "_create_unverified_context" in line and \
                        not line.lstrip().startswith("#"):
                    offenders.append(f"{path}:{number}")
        self.assertEqual(offenders, [],
                         "a process-wide TLS downgrade is back")

    def test_importing_jittor_leaves_ssl_alone(self):
        import ssl
        import jittor  # noqa: F401
        self.assertIs(ssl._create_default_https_context,
                      ssl.create_default_context)


class TestArchiveExtraction(unittest.TestCase):

    def _tar_with_escape(self, path):
        with tarfile.open(path, "w") as tar:
            payload = b"owned"
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    def test_a_member_cannot_write_outside_the_target(self):
        with tempfile.TemporaryDirectory() as d:
            archive = os.path.join(d, "evil.tar")
            self._tar_with_escape(archive)
            target = os.path.join(d, "target")
            os.makedirs(target)
            with tarfile.open(archive) as tar:
                try:
                    misc.safe_tar_extractall(tar, target)
                except Exception:
                    pass  # refusing outright is also a correct answer
            self.assertFalse(os.path.exists(os.path.join(d, "escaped.txt")),
                             "the archive wrote outside the target directory")


class TestManifest(unittest.TestCase):

    def test_every_asset_has_a_url_a_name_and_a_digest(self):
        for asset in manifest.offline_assets():
            self.assertTrue(asset.url.startswith("https://"), asset)
            self.assertTrue(asset.filename, asset)
            algorithm, digest = manifest.digest_of(asset)
            if asset.key == "mnist":
                continue  # upstream publishes no checksum for these
            self.assertIn(algorithm, ("sha256", "md5"), asset)
            self.assertTrue(digest, asset)

    def test_sha256_wins_when_both_are_recorded(self):
        asset = manifest.CUTT
        self.assertTrue(asset.sha256 and asset.md5)
        self.assertEqual(manifest.digest_of(asset)[0], "sha256")

    def test_the_offline_package_covers_what_an_install_downloads(self):
        """pack_offline.py's own list was missing msvc.zip and every jtcuda."""
        names = {asset.filename for asset in manifest.offline_assets()}
        self.assertIn("msvc.zip", names)
        self.assertIn("cuda12.2_cudnn8_linux.tgz", names)
        self.assertIn(manifest.CUTT.filename, names)
        self.assertIn(manifest.NCCL.filename, names)


class TestDigestChecking(unittest.TestCase):

    def test_algorithm_is_chosen_by_length(self):
        self.assertEqual(misc.digest_algorithm("a" * 64), "sha256")
        self.assertEqual(misc.digest_algorithm("a" * 32), "md5")
        self.assertIsNone(misc.digest_algorithm(None))
        with self.assertRaises(ValueError):
            misc.digest_algorithm("abc")

    def test_a_corrupt_file_is_removed_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            payload = b"an error page, not the archive you asked for"
            source = os.path.join(d, "server.bin")
            with open(source, "wb") as f:
                f.write(payload)
            expected = hashlib.sha256(b"the real thing").hexdigest()
            with self.assertRaises(RuntimeError) as caught:
                misc.download_url_to_local(
                    Path(source).as_uri(), "downloaded.bin", d, expected)
            message = str(caught.exception)
            self.assertIn("sha256", message)
            self.assertIn(expected, message)
            self.assertFalse(os.path.exists(os.path.join(d, "downloaded.bin")))
            self.assertFalse(os.path.exists(
                os.path.join(d, "downloaded.bin.part")),
                "the partial download was left behind")

    def test_a_good_file_is_installed(self):
        with tempfile.TemporaryDirectory() as d:
            payload = b"the real thing"
            source = os.path.join(d, "server.bin")
            with open(source, "wb") as f:
                f.write(payload)
            digest = hashlib.sha256(payload).hexdigest()
            misc.download_url_to_local(
                Path(source).as_uri(), "downloaded.bin", d, digest)
            target = os.path.join(d, "downloaded.bin")
            self.assertTrue(os.path.isfile(target))
            with open(target, "rb") as f:
                self.assertEqual(f.read(), payload)
            # A second call is a no-op rather than a re-download.
            os.remove(source)
            misc.download_url_to_local(
                Path(source).as_uri(), "downloaded.bin", d, digest)


if __name__ == "__main__":
    unittest.main()
