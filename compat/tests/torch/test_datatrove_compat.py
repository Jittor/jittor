"""DataTrove CPU regressions through the independent Jittor Torch frontend."""

import importlib.util
import json
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np


_HAS_DATATROVE = importlib.util.find_spec("datatrove") is not None
_REQUIRE_DATATROVE = os.environ.get("JITTOR_REQUIRE_DATATROVE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
if _REQUIRE_DATATROVE and not _HAS_DATATROVE:
    raise ModuleNotFoundError("JITTOR_REQUIRE_DATATROVE=1 but datatrove is not installed")


def _keep_even_document(document):
    """Top-level callable so the real multiprocess executor can serialize it."""

    return int(document.id) % 2 == 0


def _prime_multiprocess_temp_directory():
    """Avoid multiprocess 0.70.19's Python 3.13 child-only tempfile bug."""

    import multiprocess.util

    previous_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = os.environ.get("DATATROVE_MULTIPROCESS_TMPDIR", "/tmp")
        return multiprocess.util.get_temp_dir()
    finally:
        tempfile.tempdir = previous_tempdir


@unittest.skipUnless(_HAS_DATATROVE, "datatrove is not installed")
class TestDataTroveCompat(unittest.TestCase):
    def setUp(self):
        import torch

        if os.environ.get("JITTOR_DATATROVE_RUNTIME") == "oracle":
            self.assertTrue(hasattr(torch, "_C"), "oracle must be an official PyTorch build")
            self.assertFalse(
                hasattr(torch, "_torch_compat_install_context"),
                "oracle resolves to the Jittor shim",
            )
            return

        import jittor
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner

        self.assertTrue(
            namespace_owner(torch) is jittor
            and activation_status(jittor).active
            and torch.Tensor is not jittor.Var,
            "DataTrove must run through the independent Jittor Torch frontend",
        )
        self.assertEqual(jittor.flags.use_cuda, 0, "DataTrove regression is CPU-only")

    def test_file_dataset_reads_tokens_and_builds_eos_positions(self):
        import torch
        from datatrove.utils.dataset import DatatroveFileDataset

        tokens = np.array([10, 11, 2, 20, 21, 30, 2, 40, 41, 42], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "fixture.ds"
            tokens.tofile(dataset_path)
            dataset = DatatroveFileDataset(
                str(dataset_path),
                seq_len=4,
                token_size=2,
                return_positions=True,
                positions_from_eos_token_id=2,
            )

            self.assertEqual(len(dataset), 2)
            samples = [dataset[index] for index in range(len(dataset))]

            for sample in samples:
                for name in ("input_ids", "positions"):
                    self.assertIsInstance(sample[name], torch.Tensor)
                    self.assertEqual(sample[name].dtype, torch.int64)
                    self.assertEqual(sample[name].device.type, "cpu")

        self.assertEqual(
            [sample["input_ids"].tolist() for sample in samples],
            [[10, 11, 2, 20, 21], [30, 2, 40, 41, 42]],
        )
        self.assertEqual(
            [sample["positions"].tolist() for sample in samples],
            [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]],
        )

    def test_local_jsonl_filter_pipeline(self):
        from datatrove.pipeline.filters import LambdaFilter
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "output"
            input_directory.mkdir()
            input_file = input_directory / "documents.jsonl"
            input_file.write_text(
                "".join(
                    json.dumps({"id": str(index), "text": f"document {index}"}) + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )

            reader = JsonlReader(str(input_directory), compression=None)
            filter_step = LambdaFilter(lambda document: int(document.id) % 2 == 0)
            writer = JsonlWriter(
                str(output_directory),
                output_filename="${rank}.jsonl",
                compression=None,
            )
            data = reader.run(data=None, rank=0, world_size=1)
            data = filter_step.run(data, rank=0, world_size=1)
            data = writer.run(data, rank=0, world_size=1)
            deque(data, maxlen=0)
            kept = list(
                JsonlReader(str(output_directory), compression=None).run(
                    data=None,
                    rank=0,
                    world_size=1,
                )
            )

        self.assertEqual([document.id for document in kept], ["0", "2", "4"])

    def test_local_executor_multiworker_jsonl_pipeline(self):
        from datatrove.executor.local import LocalPipelineExecutor
        from datatrove.pipeline.filters import LambdaFilter
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter

        self.assertTrue(Path(_prime_multiprocess_temp_directory()).is_dir())

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "output"
            logging_directory = root / "logs"
            input_directory.mkdir()
            for index in range(8):
                (input_directory / ("%02d.jsonl" % index)).write_text(
                    json.dumps({"id": str(index), "text": "document %d" % index}) + "\n",
                    encoding="utf-8",
                )

            executor = LocalPipelineExecutor(
                pipeline=[
                    JsonlReader(str(input_directory), compression=None),
                    LambdaFilter(_keep_even_document),
                    JsonlWriter(
                        str(output_directory),
                        output_filename="${rank}.jsonl",
                        compression=None,
                    ),
                ],
                tasks=4,
                workers=2,
                logging_dir=str(logging_directory),
                start_method="forkserver",
            )
            stats = executor.run()
            kept = list(
                JsonlReader(str(output_directory), compression=None).run(
                    data=None,
                    rank=0,
                    world_size=1,
                )
            )

            self.assertEqual(sorted(document.id for document in kept), ["0", "2", "4", "6"])
            filter_stats = stats.stats[1]
            self.assertEqual(int(filter_stats["total"].total), 8)
            self.assertEqual(int(filter_stats["forwarded"].total), 4)
            self.assertEqual(int(filter_stats["dropped"].total), 4)
            for rank in range(4):
                self.assertTrue((logging_directory / "completions" / ("%05d" % rank)).is_file())


if __name__ == "__main__":
    unittest.main()
