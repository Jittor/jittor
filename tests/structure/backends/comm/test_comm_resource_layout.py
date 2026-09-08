"""Communication resources have a single packaged owner outside Python."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_legacy_runtime_resource_trees_are_absent():
    assert not (ROOT / "python/jittor/src").exists()
    assert not (ROOT / "python/jittor/extern").exists()
    for backend in ("mpi", "nccl", "hccl"):
        owner = ROOT / "backends/comm" / backend
        for directory in ("inc", "src", "ops"):
            assert (owner / directory).is_dir()
    compiler = (ROOT / "python/jittor/build/compiler.py").read_text()
    assert "os.path.join(jittor_path, 'extern')" not in compiler


def test_nccl_build_declares_sdk_and_stream_adapter_dependencies():
    compiler = (ROOT / "python/jittor/build/compile_extern.py").read_text()
    nccl = compiler.split("def setup_nccl(", 1)[1].split("def setup_hccl(", 1)[0]
    assert "+ cuda_sdk_flags + cuda_link_flags" in nccl
    wrapper = (ROOT / "backends/comm/nccl/src/nccl_wrapper.cc").read_text()
    assert '#include "stream_compat.h"' in wrapper
    mpi = (ROOT / "backends/comm/mpi/inc/mpi_wrapper.h").read_text()
    assert '#include "core/common.h"' in mpi
