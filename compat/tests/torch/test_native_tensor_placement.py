"""Explicit Tensor placement survives global policy and mixed native graphs."""

from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child
import numpy as np
import pytest


def _cuda_runtime():
    import jittor as jt
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("accelerator prerequisite: CUDA runtime unavailable")
    return jt.flag_scope(use_cuda=1, auto_flush_ops=0)


def _cpu(value):
    assert value.placement_backend == 0
    assert value.device_id == -1
    assert value.device.type == "cpu"
    assert value.location() == "cpu"
    assert not getattr(value, "_jittor_torch_force_cpu", False)


def test_str_enum_device_uses_its_public_value():
    from enum import Enum
    import torch

    class OffloadDeviceEnum(str, Enum):
        cpu = "cpu"

    value = torch.tensor([1.0, 2.0], device=OffloadDeviceEnum.cpu)
    value.sync()
    _cpu(value)
    np.testing.assert_array_equal(value.numpy(), [1.0, 2.0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_new_preserves_torch_frontend_for_writable_views(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device=device)
    batch = source.new(torch.Size([2, 3, 4])).fill_(255)
    assert isinstance(batch, torch.Tensor) and batch.device == source.device
    assert batch.dtype == source.dtype and not batch.requires_grad
    first, _ = batch
    first[..., :2, :2].copy_(source)
    expected = np.full((2, 3, 4), 255, dtype=np.int64)
    expected[0, :2, :2] = [[1, 2], [3, 4]]
    np.testing.assert_array_equal(batch.cpu().numpy(), expected)
    assert source.new(2, 3).shape == (2, 3)
    assert source.new().shape == (0,)
    np.testing.assert_array_equal(source.new((5, 6)).cpu().numpy(), [5, 6])
    assert source.new(source).data_ptr() == source.data_ptr()
    with pytest.raises(TypeError, match="matching dtype and device"):
        source.new(source.float())


def test_cpu_clamp_bounds_do_not_allocate_cuda_memory():
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    with _cuda_runtime(), forbid_backend_fallbacks(), torch.no_grad():
        source = torch.from_numpy(np.array([-1., 0.5, 2.], dtype=np.float32))
        source.sync()
        _cpu(source)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = source.clamp(0, 1)
        result.sync()
        _cpu(result)
        np.testing.assert_array_equal(result.numpy(), [0., 0.5, 1.])
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == baseline


@pytest.mark.parametrize("spatial_shape", [(6, 8), (4, 6, 8)])
def test_cpu_grid_operations_do_not_allocate_cuda_memory(spatial_shape):
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    dimensions = len(spatial_shape)
    centers = [2 * (np.arange(n, dtype=np.float32) + 0.5) / n - 1
               for n in spatial_shape]
    positions = np.stack(np.meshgrid(*centers, indexing="ij")[::-1], axis=-1)[None]
    values = np.linspace(-1, 1, 2 * np.prod(spatial_shape), dtype=np.float32)
    values = values.reshape((1, 2, *spatial_shape))
    theta = np.eye(dimensions, dimensions + 1, dtype=np.float32)[None]

    with _cuda_runtime(), forbid_backend_fallbacks(), torch.no_grad():
        image, grid, transform = map(torch.from_numpy, (values, positions, theta))
        for source in (image, grid, transform):
            source.sync()
            _cpu(source)

        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        generated = torch.nn.functional.affine_grid(transform, tuple(image.shape), align_corners=False)
        sampled = torch.nn.functional.grid_sample(image, grid, align_corners=False)
        for result, expected in ((generated, positions), (sampled, values)):
            result.sync()
            _cpu(result)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == baseline


def test_lazy_cpu_factories_views_reductions_and_native_selection():
    import jittor as jt
    import torch
    with _cuda_runtime():
        x = torch.tensor([[-2., 1.], [4., -3.]], device="cpu", requires_grad=True)
        assert x.location() == "none"
        transposed = x.t()
        reduced = ((transposed + 2) * transposed).sum()
        matrix = torch.matmul(x, x)
        random = torch.rand((2, 3), device="cpu")
        ones = x.new_ones(2, 3)
        sort = torch.argsort(x, dim=-1)
        jt.sync_all(True)
        for value in (x, transposed, reduced, matrix, random, ones, sort):
            _cpu(value)
        np.testing.assert_array_equal(matrix.numpy(), [[8., -5.], [-20., 13.]])
        np.testing.assert_array_equal(ones.numpy(), np.ones((2, 3)))
        np.testing.assert_array_equal(sort.numpy(), [[0, 1], [1, 0]])
        dx, = torch.autograd.grad(reduced, x)
        dx.sync()
        _cpu(dx)
        np.testing.assert_array_equal(dx.numpy(), [[-2., 4.], [10., -4.]])


def test_cpu_and_cuda_segments_share_one_submission_without_migration():
    import jittor as jt
    import torch
    with _cuda_runtime():
        cpu = torch.tensor([1., 2., 3.], device="cpu")
        gpu = torch.tensor([4., 5., 6.], device="cuda")
        host_result = ((cpu + 1) * 2).sum()
        device_result = ((gpu + 1) * 2).sum()
        host_like = torch.zeros_like(gpu, device="cpu")
        gpu_like = torch.ones_like(gpu)
        jt.sync_all(True)
        for value in (cpu, host_result, host_like):
            _cpu(value)
        for value in (gpu, device_result, gpu_like):
            assert value.placement_backend == 1
            assert value.device.type == "cuda" and value.location() == "device"
        assert host_result.item() == 18
        assert device_result.item() == 36
        with pytest.raises(RuntimeError, match="same backend and device"):
            _ = cpu + gpu


def test_explicit_copy_roundtrip_and_gradient_keep_source_placement():
    import jittor as jt
    import torch
    with _cuda_runtime():
        x = torch.tensor([2., 3.], device="cpu", requires_grad=True)
        gpu = x.to("cuda")
        host = (gpu * gpu).cpu()
        dx, = torch.autograd.grad(host.sum(), x)
        jt.sync_all(True)
        _cpu(x)
        _cpu(host)
        _cpu(dx)
        assert gpu.location() == "device"
        assert gpu.placement_backend == 1
        np.testing.assert_array_equal(host.numpy(), [4., 9.])
        np.testing.assert_array_equal(dx.numpy(), [4., 6.])
        assert x.cpu() is x
        assert gpu.cuda() is gpu
        copied = x.to("cpu", copy=True)
        copied.sync()
        _cpu(copied)
        assert copied is not x
        if jt.core.get_device_count() >= 2:
            other_gpu = x.cuda(1)
            first_gpu = other_gpu.to("cuda:0")
            roundtrip = first_gpu.cpu()
            copied_grad, = torch.autograd.grad((roundtrip * roundtrip).sum(), x)
            jt.sync_all(True)
            assert other_gpu.device_id == 1 and other_gpu.location() == "device"
            assert first_gpu.device_id == 0 and first_gpu.location() == "device"
            _cpu(roundtrip)
            _cpu(copied_grad)
            np.testing.assert_array_equal(copied_grad.numpy(), [4., 6.])


@pytest.mark.cuda
def test_to_device_keyword_accepts_accelerator_index():
    import jittor as jt
    import torch

    with _cuda_runtime():
        host = torch.tensor([1.5, -2.0], device="cpu")
        moved = host.to(device=0, dtype=torch.float16)
        squared = moved.square()
        jt.sync_all(True)
        _cpu(host)
        assert moved.device == torch.device("cuda:0") and moved.dtype == torch.float16
        assert squared.device == moved.device and squared.location() == "device"
        np.testing.assert_array_equal(squared.cpu().numpy(), [2.25, 4.0])


@pytest.mark.cuda
def test_a_device_var_parked_on_the_host_is_moved_back_by_to_device():
    """Asking for the device must not accept host storage for a device Var.

    Explicit host migration parks a device Var in host memory while
    ``device_id`` keeps naming the device it belongs to. Scalar reads now
    preserve device storage, so they cannot establish this precondition. Such a Var is not on its device: an extension handed
    it reads a host pointer and fails its own ``is_cuda()`` check, so
    ``.to("cuda:N")``/``.cuda()`` have to move it back rather than trust the
    matching index.
    """
    import jittor as jt
    import torch
    with _cuda_runtime():
        bounds = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda:0")
        jt.sync_all(True)
        assert bounds.device_id == 0 and bounds.location() == "device"

        # Explicitly establish host storage without changing logical placement.
        assert int(bounds[1]) == 4
        bounds.migrate_to_cpu()
        assert bounds.device_id == 0 and bounds.location() == "cpu"

        back = bounds.to("cuda:0")
        back.sync()
        assert back.device_id == 0 and back.location() == "device"
        np.testing.assert_array_equal(back.numpy(), [0, 4, 8])

        assert int(bounds[1]) == 4
        bounds.migrate_to_cpu()
        assert bounds.location() == "cpu"
        shown = bounds.cuda()
        shown.sync()
        assert shown.device_id == 0 and shown.location() == "device"
        np.testing.assert_array_equal(shown.numpy(), [0, 4, 8])

        bounds.migrate_to_cpu()
        assert bounds.location() == "cpu"
        dtype_only = bounds.to(torch.int32)
        dtype_only.sync()
        assert dtype_only.device_id == 0 and dtype_only.location() == "device"
        np.testing.assert_array_equal(dtype_only.numpy(), [0, 4, 8])


def test_explicit_cuda_ignores_disabled_default_while_native_follows_runtime():
    import jittor as jt
    import torch
    with _cuda_runtime():
        gpu = torch.tensor([3., 4.], device="cuda")
        gpu.sync()
        with jt.flag_scope(use_cuda=0):
            result = gpu + 2
            result.sync()
            assert result.placement_backend == 1
            assert result.device.type == "cuda" and result.location() == "device"
            native = jt.array([1., 2.])
            assert type(native) is jt.Var and native.placement_backend == -1
            native.sync()
            assert native.location() == "cpu"
        native_result = native + 1
        native_result.sync()
        assert native_result.placement_backend == -1
        assert native_result.location() == "device"
        np.testing.assert_array_equal(native_result.numpy(), [2., 3.])


def test_numpy_and_parameter_cpu_ownership_under_cuda_default():
    import jittor as jt
    import torch
    from jittor.compat.torch import frontend
    assert torch.nn.Parameter.__new__ is frontend.parameter_new
    assert torch.nn.Parameter.__init__ is frontend.parameter_init
    with _cuda_runtime():
        array = np.array([2**40 + 1, 2**40 + 2], dtype=np.int64)
        value = torch.from_numpy(array)
        parameter = torch.nn.Parameter(torch.tensor([1., 2.], device="cpu"))
        copied = parameter.cuda().cpu()
        jt.sync_all(True)
        _cpu(value)
        _cpu(parameter)
        _cpu(copied)
        np.testing.assert_array_equal((value + 1).numpy(), array + 1)
        assert value.location() == "cpu"


def test_published_scalar_operands_keep_backend_and_both_gradients():
    import jittor as jt
    import torch
    with _cuda_runtime():
        gpu_scalar = torch.tensor(2., device="cuda", requires_grad=True)
        host_vector = torch.tensor([3., 4.], device="cpu")
        for left, right in ((gpu_scalar, host_vector), (host_vector, gpu_scalar)):
            with pytest.raises(RuntimeError, match="same backend and device"):
                _ = left * right
        gpu_scalar.sync()
        assert gpu_scalar.placement_backend == 1 and gpu_scalar.location() == "device"
        for reverse in (False, True):
            scalar = torch.tensor(2., device="cpu", requires_grad=True)
            vector = torch.tensor([3., 4.], device="cuda", requires_grad=True)
            result = vector * scalar if reverse else scalar * vector
            ds, dv = torch.autograd.grad(result.sum(), (scalar, vector))
            jt.sync_all(True)
            _cpu(scalar)
            _cpu(ds)
            assert vector.location() == result.location() == dv.location() == "device"
            np.testing.assert_array_equal(result.numpy(), [6., 8.])
            assert ds.item() == 7
            np.testing.assert_array_equal(dv.numpy(), [2., 2.])
        for reverse in (False, True):
            cpu = torch.tensor(2., device="cpu", requires_grad=True)
            cuda = torch.tensor(3., device="cuda", requires_grad=True)
            result = cuda * cpu if reverse else cpu * cuda
            dc, dg = torch.autograd.grad(result, (cpu, cuda))
            jt.sync_all(True)
            _cpu(cpu)
            _cpu(dc)
            assert cuda.placement_backend == result.placement_backend == dg.placement_backend == 1
            assert result.item() == 6 and dc.item() == 3 and dg.item() == 2
        base = torch.tensor([1., 2.], device="cpu", requires_grad=True)
        reduced = base.sum()
        vector = torch.tensor([3., 4.], device="cuda", requires_grad=True)
        result = reduced * vector
        db, dv = torch.autograd.grad(result.sum(), (base, vector))
        zero = torch.zeros((), device="cpu")
        with_zero = vector + zero
        jt.sync_all(True)
        for value in (base, reduced, db, zero):
            _cpu(value)
        assert result.location() == dv.location() == with_zero.location() == "device"
        np.testing.assert_array_equal(db.numpy(), [7., 7.])
        np.testing.assert_array_equal(dv.numpy(), [3., 3.])
        np.testing.assert_array_equal(with_zero.numpy(), [3., 4.])



def test_host_scalar_operands_queued_behind_device_work_keep_their_values():
    # A 0-d host operand of a CUDA op is copied up through one reused pageable
    # bounce buffer rather than waiting for the stream to drain
    # (backends/cuda/runtime/driver.cc); each copy must be staged before the
    # next one overwrites the buffer, while the matmuls still hold the stream.
    import jittor as jt
    import torch
    with _cuda_runtime():
        a = torch.full((1024, 1024), 1 / 1024, device="cuda")
        y = a
        for _ in range(16):
            y = y @ a
        scales = [torch.tensor(float(i), device="cpu") for i in range(64)]
        outs = [y * scale for scale in scales]
        del scales
        jt.sync_all(True)
        base = y.numpy()
        for i, out in enumerate(outs):
            np.testing.assert_allclose(out.numpy(), base * i, rtol=1e-6)


def test_frequency_factories_use_native_placement_for_following_operations():
    import torch
    with _cuda_runtime():
        for name in ("fftfreq", "rfftfreq"):
            reference = getattr(np.fft, name)(8, 0.5)
            for device in ("cpu", "cuda"):
                value = getattr(torch.fft, name)(8, d=0.5, device=device)
                result = value * 2 + 1
                result.sync()
                assert value.device.type == result.device.type == device
                assert value.location() == result.location() == ("cpu" if device == "cpu" else "device")
                np.testing.assert_array_equal(result.numpy(), reference * 2 + 1)


_NUMPY_CODE_ON_EXPLICIT_CUDA = """
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1
gpu = torch.tensor([1., 2., 3., 4.], device="cuda")
gpu.sync()

def forward(np_, data):
    np_.copyto(data["outputs"][0], data["inputs"][0] * 2)

with jt.flag_scope(use_cuda=0):
    doubled = jt.numpy_code(gpu.shape, gpu.dtype, [gpu], forward)
    assert doubled.placement_backend == 1
    try:
        got = doubled.numpy()
    except RuntimeError as error:
        assert "CuPy" in str(error), error
        # Still pending: leaving the scope would run it again.
        del doubled
        print("REFUSED")
    else:
        np.testing.assert_array_equal(got, [2., 4., 6., 8.])
        print("RAN")
"""


def test_a_numpy_code_operator_on_explicit_cuda_inputs_gets_device_arrays():
    # The operator follows its explicitly placed inputs onto CUDA under
    # use_cuda=0, as `gpu + 2` does above, so its callback must be handed
    # CuPy -- or refuse without it. It used to be chosen by use_cuda: numpy
    # over device memory, and the first read was a segfault (the torch.linalg
    # tests after a `set_default_device("cuda")`). In a child, since the
    # failure is the process.
    import jittor as jt
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("accelerator prerequisite: CUDA runtime unavailable")
    finished = run_python_child(["-c", _NUMPY_CODE_ON_EXPLICIT_CUDA],
                                env={"JITTOR_TORCH_SHIM": "1"}, timeout=900,
                                merge_stderr=True)
    tail = finished.stdout[-3000:]
    assert finished.returncode == 0, tail
    assert {"REFUSED", "RAN"} & set(finished.stdout.splitlines()), tail

@pytest.mark.parametrize("source,ambient", [("cpu", 1), ("cuda", 0)])
@pytest.mark.parametrize("by_row", [False, True])
def test_unique_counts_inherit_explicit_input_placement(source, ambient, by_row):
    import jittor as jt
    import torch
    raw = np.array([[1, 2], [2, 3], [1, 2], [4, 5]], dtype=np.int64)
    with _cuda_runtime():
        x = torch.tensor(raw, dtype=torch.int64, device=source)
        x.sync()
        with jt.flag_scope(use_cuda=ambient):
            values, inverse, counts = torch.unique(x, dim=0 if by_row else None, return_inverse=True, return_counts=True)
            jt.sync_all(True)
        want = np.unique(raw, return_inverse=True, return_counts=True, axis=0 if by_row else None)
        for value in (values, inverse, counts):
            assert value.device.type == source
        np.testing.assert_array_equal(values.numpy(), want[0])
        np.testing.assert_array_equal(inverse.numpy().reshape(want[1].shape), want[1])
        np.testing.assert_array_equal(counts.numpy(), want[2])



def test_parameter_data_slice_copies_cpu_checkpoint_weight_to_cuda():
    import jittor as jt
    import torch

    with _cuda_runtime(), jt.flag_scope(backend_fallback="error"):
        parameter = torch.nn.Parameter(torch.zeros((4, 3), dtype=torch.bfloat16, device="cuda"))
        before = torch.arange(12, dtype=torch.float32, device="cpu").reshape(4, 3)
        weight = before.to(torch.bfloat16)
        data = parameter.data
        data[1:3, ...] = weight[1:3, ...]
        jt.sync([parameter, data, weight], device_sync=True)

        assert parameter.requires_grad and not data.requires_grad
        assert parameter.device.type == "cuda" and parameter.location() == "device"
        assert data.device == parameter.device and data.location() == "device"
        _cpu(weight)
        expected = [[0., 0., 0.], [3., 4., 5.], [6., 7., 8.], [0., 0., 0.]]
        np.testing.assert_array_equal(parameter.cpu().float().numpy(), expected)
        np.testing.assert_array_equal(data.cpu().float().numpy(), expected)
        np.testing.assert_array_equal(weight.float().numpy(), before.numpy())



def test_cuda_boolean_index_with_trailing_ellipsis_matches_mask_assignment():
    import jittor as jt
    import torch

    with _cuda_runtime(), jt.flag_scope(backend_fallback="error"):
        value = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 2, 2)
        mask = torch.tensor([True, False, True], dtype=torch.bool, device="cuda")
        value[mask, ...] = 0
        jt.sync([value], device_sync=True)

        assert value.location() == "device"
        np.testing.assert_array_equal(value[mask, ...].cpu().numpy(), np.zeros((2, 2, 2)))
        np.testing.assert_array_equal(value.cpu().numpy(), [
            [[0, 0], [0, 0]], [[4, 5], [6, 7]], [[0, 0], [0, 0]],
        ])
