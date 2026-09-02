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


class TestMklCheckedByLoadingIt(unittest.TestCase):
    """Installing MKL used to compile *and run* an upstream example.

    On the import path, with the user's compiler, and with
    ``assert 0 == os.system(...)`` as the whole diagnostic. Loading the
    library and resolving the symbol jittor's own operators call answers the
    same question -- is this archive usable from this process -- without
    building and executing a third-party program.
    """

    def test_a_missing_library_is_named(self):
        from jittor.compile_extern import check_mkl_usable
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as caught:
                check_mkl_usable(d)
            self.assertIn(d, str(caught.exception))

    def test_a_library_without_the_symbol_is_rejected(self):
        from jittor.compile_extern import check_mkl_usable
        with tempfile.TemporaryDirectory() as d:
            lib = os.path.join(d, "lib")
            os.makedirs(lib)
            # A real, loadable shared object that simply is not oneDNN.
            import ctypes.util
            source = ctypes.util.find_library("m")
            self.assertIsNotNone(source, "libm not found")
            import shutil
            found = None
            for root in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
                         "/lib64", "/usr/lib"):
                candidate = os.path.join(root, source)
                if os.path.isfile(candidate):
                    found = candidate
                    break
            if found is None:
                self.skipTest("no plain shared object to stand in for oneDNN")
            shutil.copy(found, os.path.join(lib, "libmkldnn.so"))
            with self.assertRaises(RuntimeError) as caught:
                check_mkl_usable(d)
            self.assertIn("dnnl_sgemm", str(caught.exception))

    def test_the_real_one_passes(self):
        from jittor.compile_extern import check_mkl_usable
        import jittor_utils as jit_utils
        root = os.path.join(jit_utils.home(), ".cache", "jittor", "mkl")
        installed = [name for name in sorted(os.listdir(root))
                     if name.startswith("dnnl")
                     and os.path.isdir(os.path.join(root, name))] \
            if os.path.isdir(root) else []
        if not installed:
            self.skipTest("MKL is not installed in this cache")
        check_mkl_usable(os.path.join(root, installed[0]))


class TestCutlassIsGone(unittest.TestCase):
    """Downloaded on every CUDA machine; referenced by nothing."""

    def test_the_build_no_longer_fetches_or_sets_up_cutlass(self):
        """Unrelated mentions elsewhere (the vLLM shim names CUTLASS kernels)
        are not this: what had to go is the download and the setup hook."""
        import jittor.compile_extern as compile_extern
        for name in ("install_cutlass", "setup_cutlass", "use_cutlass",
                     "cutlass_ops"):
            self.assertFalse(hasattr(compile_extern, name),
                             f"compile_extern.{name} is back")
        source = (REPO / "python" / "jittor" / "compile_extern.py").read_text(
            encoding="utf8")
        self.assertNotIn("cutlass", source.lower())

    def test_nccl_is_not_fetched_before_the_conditions_are_checked(self):
        """The device-count and MPI checks used to come after the download."""
        source = (REPO / "python" / "jittor" / "compile_extern.py").read_text(
            encoding="utf8")
        body = source[source.index("def install_nccl("):]
        body = body[:body.index("\ndef ")]
        self.assertLess(body.index("get_device_count"),
                        body.index("download_url_to_local"))
        self.assertLess(body.index("inside_mpi"),
                        body.index("download_url_to_local"))


if __name__ == "__main__":
    unittest.main()
