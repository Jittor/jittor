"""CUDA source ownership and generated shared-template composition, without JIT."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from _helpers.op_registration_generator import load_op_registration_generator as _load_generator


ROOT = Path(__file__).resolve().parents[4]
JITTOR = ROOT / "python/jittor"
KERNELS = ROOT / "backends/cuda/kernels/core"


@pytest.mark.parametrize("name", ["where", "candidate", "transpose"])
def test_separate_cuda_kernel_is_the_registered_accelerator_source(tmp_path, name):
    source = _load_generator(tmp_path)([str(ROOT / "src/ops/composite" / (name + "_op.h"))])
    assert json.dumps(str(KERNELS / (name + "_op.cc"))) in source
    assert "::backend_mask" in source
    assert "__global__" in (KERNELS / (name + "_op.cc")).read_text()
    assert "__global__" not in (ROOT / "src/ops/composite" / (name + "_op.cc")).read_text()


@pytest.mark.parametrize("name", ["getitem", "setitem"])
def test_shared_indexing_source_has_real_backend_prefix_and_stable_publication(tmp_path, name):
    generator = _load_generator(tmp_path)
    header = str(ROOT / "src/ops/composite" / (name + "_op.h"))
    registration = generator([header])
    composed = tmp_path / "backend_sources" / (name + "_cuda.cc")
    assert json.dumps(str(composed)) in registration
    source = composed.read_text()
    prefix = KERNELS / (name + "_prefix.cc")
    shared = ROOT / "src/ops/composite" / (name + "_op.cc")
    assert '#line 1 ' + json.dumps(str(prefix)) in source
    assert '#line 1 ' + json.dumps(str(shared)) in source
    assert prefix.read_text() in source and shared.read_text() in source
    before = composed.stat()
    assert generator([header]) == registration
    after = composed.stat()
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert not list(composed.parent.glob("*.tmp"))


def test_shared_indexing_composition_is_atomic_and_failure_preserves_old_source(tmp_path, monkeypatch):
    generator = _load_generator(tmp_path / "cache")
    fake_jittor = tmp_path / "jittor"
    core = fake_jittor / "src/ops/composite"
    core.mkdir(parents=True)
    backend = tmp_path / "backend"
    kernels = backend / "kernels/core"
    kernels.mkdir(parents=True)
    for suffix in (".h", ".cc"):
        (core / ("getitem_op" + suffix)).write_text(
            (ROOT / "src/ops/composite" / ("getitem_op" + suffix)).read_text())
    prefix = kernels / "getitem_prefix.cc"
    prefix.write_text("// first backend prefix\n")
    generator.__globals__["jittor_path"] = str(fake_jittor)
    generator.__globals__["core_root"] = lambda path: str(fake_jittor / "src")
    generator.__globals__["backend_root"] = lambda path, name: str(backend)
    header = str(core / "getitem_op.h")
    generator([header])
    composed = tmp_path / "cache/backend_sources/getitem_cuda.cc"
    original = composed.read_text()
    inode = composed.stat().st_ino
    prefix.write_text("// second backend prefix\n")
    generator([header])
    assert composed.stat().st_ino != inode
    assert composed.read_text() != original
    original = composed.read_text()
    prefix.write_text("// unpublished backend prefix\n")

    def reject_publish(source, destination):
        raise PermissionError("publication rejected")

    monkeypatch.setattr(os, "replace", reject_publish)
    with pytest.raises(PermissionError, match="publication rejected"):
        generator([header])
    assert composed.read_text() == original
    assert not list(composed.parent.glob("*.tmp"))


def test_external_operator_does_not_inherit_a_builtin_cuda_source(tmp_path):
    custom = tmp_path / "getitem_op.h"
    custom.write_text((ROOT / "src/ops/composite/getitem_op.h").read_text())
    source = _load_generator(tmp_path / "cache")([str(custom)], export="custom_indexing")
    assert "backend_sources" not in source
    assert "_prefix.cc" not in source
    assert "indexing_codegen" not in source
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("cuda", [False, True])
def test_indexing_registration_only_links_backend_codegen_in_accelerator_build(tmp_path, cuda):
    source = tmp_path / "registration.cc"
    source.write_text('''
#include "ops/composite/getitem_op.h"
#include "ops/composite/op_registration.h"
void register_test() {
    jittor::register_op_definition<jittor::GetitemOp>(
        {"getitem", "shared.cc", ""});
}
''')
    output = tmp_path / "registration.o"
    command = [os.environ.get("CXX", "g++"), "-std=c++14", "-c", str(source),
               "-I", str(ROOT / "src"), "-o", str(output)]
    if cuda:
        command += ["-DHAS_CUDA", "-DIS_CUDA"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    symbols = subprocess.run(["nm", "-C", "-u", str(output)], capture_output=True,
                             text=True, check=True, timeout=10).stdout
    assert ("GetitemOp::configure_accelerator_codegen" in symbols) is cuda


def test_indexing_cuda_algorithms_have_one_backend_owner():
    for name in ("getitem", "setitem"):
        source = (ROOT / "src/ops/composite" / (name + "_op.cc")).read_text()
        for token in ("__global__", "__device__", "cuda_loop_schedule", "cudaMemcpy", "cuda_atomic"):
            assert token not in source
    backend = (KERNELS / "indexing_codegen.cc").read_text()
    assert "void cuda_loop_schedule(" in (KERNELS / "indexing_schedule_codegen.cc").read_text()
    assert '"static __global__ void"' in backend
    prefix = (KERNELS / "setitem_prefix.cc").read_text()
    for token in ("atomicAdd", "cuda_atomic_max_rmw", "cuda_atomic_min_rmw", "cuda_atomic_mul"):
        assert token in prefix
    assert "@is_def(indexing_backend_" not in prefix
    for operation in ("void", "add", "maximum", "minimum", "multiply"):
        assert "@strcmp(@OP," + operation + ")==0" in prefix
