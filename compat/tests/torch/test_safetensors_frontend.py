"""Optional safetensors I/O uses the selected frontend without narrowing."""

import textwrap

import pytest

from _helpers.child_process import run_python_child


def test_safetensors_frontend_dtype_device_and_numpy_owner(tmp_path):
    pytest.importorskip("safetensors")
    result = run_python_child(["-c", textwrap.dedent("""
        import sys
        import numpy as np
        import jittor as jt
        from jittor.compat.shim import activate
        torch = activate(auto_scan_extensions=False, build_extensions=False,
                         configure_cuda=False, local_home=False, verbose=False)["torch"]
        import safetensors
        import safetensors.torch as st
        import safetensors.numpy as sn
        values = {"wide": torch.tensor([2**45, 2**45+1], dtype=torch.int64),
                  "bf16": torch.tensor([1.25, 2.5], dtype=torch.bfloat16),
                  "scalar": torch.tensor(3.25)}
        path = sys.argv[1] + "/tensors.safetensors"
        st.save_file(values, path)
        loaded = st.load_file(path, device="cpu")
        for key, value in loaded.items():
            assert type(value) is torch.Tensor
            assert value.dtype == values[key].dtype
            assert value.is_cpu
            np.testing.assert_array_equal(value.numpy(), values[key].numpy())
        assert tuple(loaded["scalar"].shape) == ()
        assert st.load(st.save(values))["bf16"].dtype is torch.bfloat16
        with safetensors.safe_open(path, framework="pt", device="cpu") as reader:
            assert reader.get_dtype("bf16") == "BF16"
            sliced = reader.get_slice("wide")[1:]
            assert type(sliced) is torch.Tensor and sliced.dtype is torch.int64
            np.testing.assert_array_equal(sliced.numpy(), [2**45+1])
        numpy_path = sys.argv[1] + "/numpy.safetensors"
        sn.save_file({"wide": np.array([2**45], dtype=np.int64)}, numpy_path)
        native = sn.load_file(numpy_path)["wide"]
        assert type(native) is np.ndarray and native.dtype == np.int64
        print("SAFETENSORS_FRONTEND_OK")
    """), str(tmp_path)], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "SAFETENSORS_FRONTEND_OK" in result.stdout
