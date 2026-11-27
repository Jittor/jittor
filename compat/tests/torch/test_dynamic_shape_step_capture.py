"""Coordinate counts used by Python must be guarded before replay."""
import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.cuda
def test_boolean_index_from_basic_views_replays_with_live_coordinates():
    require_accelerator("cuda")

    def forward(coordinates):
        keep = coordinates[0] != coordinates[1]
        return coordinates[:, keep].sum(dim=0)

    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        compiled = torch.compile(forward, mode="reduce-overhead")
        for index in range(8):
            row = np.arange(32, dtype=np.int64)
            other = row + ((row + index) % 2)
            coordinates = torch.tensor(np.stack([row, other]), device="cuda")
            actual = compiled(coordinates)
            assert actual.is_cuda
            np.testing.assert_array_equal(actual.cpu().numpy(), (row + other)[row != other])

        assert compiled.refused is None
        assert compiled.stats["graph"], compiled.graph_refused

        row = np.arange(32, dtype=np.int64)
        changed = torch.tensor(np.stack([row, row + 1]), device="cuda")
        np.testing.assert_array_equal(compiled(changed).cpu().numpy(), row * 2 + 1)
        assert "dynamic prefix changed" in compiled.last_invalidated


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_integer_shape_reads_guard_the_source_storage(device):
    if device == "cuda":
        require_accelerator("cuda")
    widths = torch.tensor([1, 2], device=device)

    def forward(source):
        width = (widths[0] + widths[1]) // 2
        if width > 1:
            source = source * 2
        return torch.nn.functional.pad(source, (width, 0))

    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        compiled = torch.compile(forward, mode="reduce-overhead")
        for number in range(8):
            source = torch.full((2,), float(number), device=device)
            np.testing.assert_array_equal(compiled(source).cpu().numpy(), [0, number, number])
        assert compiled.refused is None
        assert compiled.stats["replayed"]
        if device == "cuda":
            assert compiled.stats["graph"], compiled.graph_refused

        widths[0].add_(2)
        source = torch.full((2,), 3.0, device=device)
        np.testing.assert_array_equal(compiled(source).cpu().numpy(), [0, 0, 6, 6])
        assert "scalar dependency" in compiled.last_invalidated


def test_read_of_private_argument_remains_dynamic():
    def forward(source, width):
        return torch.nn.functional.pad(source, (width[0], 0))

    with torch.no_grad():
        compiled = torch.compile(forward, mode="reduce-overhead")
        for width in (1, 1, 2):
            result = compiled(torch.ones(2, device="cpu"), torch.tensor([width], device="cpu"))
            np.testing.assert_array_equal(result.numpy(), [0] * width + [1, 1])
        assert "read a value back to the host" in compiled.refused
        assert compiled.stats["replayed"] == 0


@pytest.mark.cuda
def test_integer_shape_reads_from_migrated_module_buffers():
    require_accelerator("cuda")

    class Padding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("kernel", torch.tensor(5, device="cpu"))
            self.register_buffer("stride", torch.tensor(2, device="cpu"))
            self.register_buffer("padding", self.kernel - self.stride)

        def forward(self, source):
            length = source.shape[-1]
            frames = ((length - self.kernel + self.padding) / self.stride).ceil().long()
            extra = frames * self.stride + self.kernel - self.padding - length
            if extra > self.padding:
                source = source * 2
            return torch.nn.functional.pad(source, (self.padding, extra))

    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        module = Padding().to("cuda").eval()
        compiled = torch.compile(module.forward, mode="reduce-overhead")
        for number in range(7):
            source = torch.full((5,), float(number), device="cuda")
            actual = compiled(source)
            assert actual.is_cuda
            np.testing.assert_array_equal(actual.cpu().numpy(), [0] * 3 + [number] * 5 + [0])
        assert compiled.refused is None
        assert compiled.stats["graph"], compiled.graph_refused


def test_read_of_foreign_storage_remains_dynamic():
    array = np.array([1], dtype=np.int64)
    width = torch.from_numpy(array)

    def forward(source):
        return torch.nn.functional.pad(source, (width[0], 0))

    with torch.no_grad():
        compiled = torch.compile(forward, mode="reduce-overhead")
        source = torch.ones(2, device="cpu")
        for number in (1, 1, 2):
            array[0] = number
            np.testing.assert_array_equal(compiled(source).numpy(), [0] * number + [1, 1])
        assert "read a value back to the host" in compiled.refused


@pytest.mark.cuda
@pytest.mark.parametrize("record", [False, True])
def test_unique_count_and_masked_pooling_guard_optimizer_replay(record):
    require_accelerator("cuda")

    def make_step(weight, optimizer):
        def step(tokens, values):
            eos = tokens == 1
            if len(torch.unique_consecutive(eos.sum(1))) > 1:
                raise ValueError("Unequal EOS counts")
            pooled = (values * weight)[eos, :].view(tokens.shape[0], -1, 2)[:, -1, :]
            loss = pooled.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach()
        return step

    reference_weight = torch.nn.Parameter(torch.tensor([1., 2.], device="cuda"))
    candidate_weight = torch.nn.Parameter(reference_weight.detach().clone())
    reference_opt = torch.optim.AdamW([reference_weight], lr=.001)
    candidate_opt = torch.optim.AdamW([candidate_weight], lr=.001)
    eager = make_step(reference_weight, reference_opt)
    captured = jt.capture_step(make_step(candidate_weight, candidate_opt), record=record)
    values = torch.arange(48, dtype=torch.float32, device="cuda").reshape(3, 8, 2) / 48

    def feed(rows):
        return torch.tensor(rows, dtype=torch.int64, device="cuda")

    def compare(tokens):
        expected, actual = eager(tokens, values), captured(tokens, values)
        np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-6)
        np.testing.assert_allclose(candidate_weight.detach().cpu().numpy(),
                                   reference_weight.detach().cpu().numpy(), rtol=2e-6)

    with jt.flag_scope(backend_fallback="error"):
        for i in range(8):
            compare(feed([[int(j == (i + row) % 8) for j in range(8)] for row in range(3)]))
        assert captured.refused is None
        assert captured.stats["replayed"]
        if record:
            assert captured.stats["graph"], captured.graph_refused

        # The total stays at three so only the count-of-runs guard, rather
        # than the compaction size for masked indexing, sees this change.
        invalid = feed([[1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        before = candidate_weight.detach().cpu().numpy().copy()
        with pytest.raises(ValueError, match="Unequal EOS counts"):
            captured(invalid, values)
        np.testing.assert_array_equal(candidate_weight.detach().cpu().numpy(), before)

        # The step can continue with a new valid number of positions.
        for i in range(8):
            compare(feed([[int((j + i + row) % 8 < 2) for j in range(8)] for row in range(3)]))
        opt_state = candidate_opt.state[candidate_weight]
        assert opt_state["step"].item() == 16
