"""Host edits must not invalidate kernels; template/header/flag edits must.

Each observation uses a fresh process: dlopen and the in-memory kernel cache
must not hide disk invalidation. Only the temporary custom op is edited.
"""

import json

import pytest

import jittor as jt
from _helpers import capability
from _helpers.child_process import run_child_script


HEADER = r"""
#pragma once
#include "core/op.h"
namespace jittor {
struct JitincrementalOp : Op {
    Var* x, * y;
    JitincrementalOp(Var* x);
    const char* name() const override { return "jitincremental"; }
    DECLARE_jit_run;
};
}
"""
PREFIX = r"""
#include "core/var.h"
#include "jitincremental_op.h"
#include "incremental_value.h"
namespace jittor {
"""
HOST = r"""
JitincrementalOp::JitincrementalOp(Var* x) : x(x) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::element);
    y = create_output(x->shape, x->dtype());
}
void JitincrementalOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
}
"""
KERNEL = r"""
void JitincrementalOp::jit_run() {
    auto* xp = x->ptr<T>();
    auto* yp = y->ptr<T>();
    index_t num = y->num;
    for (index_t i=0; i<num; i++) {
        yp[i] = xp[i] + 1 + ::jittor_incremental_value();
    }
}
"""
DEPENDENCY = r"""
#pragma once
#ifdef __CUDACC__
__host__ __device__
#endif
inline int jittor_incremental_value() { return 0 + FLAG_DELTA; }
"""


@pytest.mark.parametrize("use_cuda", [0, 1], ids=["cpu", "cuda"])
def test_companion_rebuild_boundaries(tmp_path, use_cuda):
    if use_cuda and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    header = tmp_path / "jitincremental_op.h"
    host = tmp_path / "jitincremental_op.cc"
    kernel = tmp_path / "jitincremental_op.jit.cc"
    dependency = tmp_path / "incremental_value.h"
    header.write_text(HEADER)
    host.write_text(PREFIX + "#ifndef JIT\n" + HOST + "#else\n" + KERNEL + "#endif\n}\n")
    dependency.write_text(DEPENDENCY)

    def execute(expected, *, flag=0, explicit=False):
        filenames = [str(header), str(host)]
        if explicit:
            filenames.append(str(kernel))
        script = f"""
import hashlib, json
from pathlib import Path
import numpy as np
import jittor as jt
jt.flags.use_cuda = {use_cuda}
module = jt.compile_custom_ops({filenames!r}, extra_flags=' -DFLAG_DELTA={flag} ', return_module=True)
with jt.flag_scope(lazy_execution=1, enable_tuner=0):
    x = jt.array(np.arange(16, dtype=np.float32))
    x.sync()
    got = (module.ops.jitincremental(x) * 2).numpy()
np.testing.assert_array_equal(got, (np.arange(16) + {expected}) * 2)
cache = Path(jt.compiler.cache_path)
suffix = jt.compiler.build_config.kernel_source_suffix if {use_cuda} else '.cc'
marker = {'#define JIT_cuda 1' if use_cuda else '#define JIT_cpu 1'!r}
products = sorted(p for p in (cache / 'jit').glob('*jitincremental*.so')
                  if p.with_suffix(suffix).exists()
                  and marker in p.with_suffix(suffix).read_text())
assert products, 'no custom kernel was compiled'
def signature(path):
    return [path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()]
sources = [p.with_suffix(suffix) for p in products]
assert all(p.exists() for p in sources), sources
print('INCREMENTAL_RESULT=' + json.dumps({{
    'kernels': {{p.name: signature(p) for p in products}},
    'host_object': signature(cache / 'obj_files' / 'jitincremental_op.cc.o'),
    'library': signature(Path(module.__file__)),
    'source': '\\n'.join(p.read_text() for p in sources),
}}))
"""
        result = run_child_script(script, directory=tmp_path, name="incremental_probe",
                                  without_torch_mode=True, text=True)
        assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
        rows = [line.partition("=")[2] for line in result.stdout.splitlines()
                if line.startswith("INCREMENTAL_RESULT=")]
        assert len(rows) == 1, result.stdout
        return json.loads(rows[0])

    legacy = execute(1)
    assert "jitincremental_op.cc" in legacy["source"]
    # Discover a companion even when callers keep the original file list.
    kernel.write_text(PREFIX + KERNEL.replace("+ 1 +", "+ 2 +") + "}\n")
    first = execute(2)
    assert "jitincremental_op.jit.cc" in first["source"]
    line = kernel.read_text().split("void JitincrementalOp::jit_run", 1)[0].count("\n") + 1
    assert f'#line {line} "{kernel}"' in first["source"]
    assert execute(2) == first
    host.write_text("// A host-only line shift.\n" + host.read_text())
    second = execute(2, explicit=True)
    # A comment recompiles the host object. Its bytes may remain identical,
    # in which case the content-addressed linker correctly reuses the library.
    assert second["host_object"] != first["host_object"]
    assert second["kernels"] == first["kernels"]
    assert second["source"] == first["source"]

    kernel.write_text(kernel.read_text().replace("+ 2 +", "+ 3 +"))
    third = execute(3, explicit=True)
    assert third["kernels"] != second["kernels"]
    assert third["host_object"] == second["host_object"]
    assert third["library"] == second["library"]
    dependency.write_text(DEPENDENCY.replace("return 0 +", "return 7 +"))
    fourth = execute(10, explicit=True)
    assert fourth["kernels"] != third["kernels"]
    fifth = execute(11, flag=1, explicit=True)
    assert fifth["kernels"] != fourth["kernels"]

    # Removing the companion must restore the combined-source registration.
    kernel.unlink()
    restored = execute(8)
    assert "jitincremental_op.jit.cc" not in restored["source"]
    assert restored["kernels"] != fourth["kernels"]



def test_native_module_with_only_free_function_bindings(tmp_path):
    header = tmp_path / "binding_only.h"
    source = tmp_path / "binding_only.cc"
    header.write_text("""
#pragma once
namespace jittor {
// @pyjt(increment)
int increment(int value);
}
""")
    source.write_text("""
#include "binding_only.h"
namespace jittor {
int increment(int value) { return value + 1; }
}
""")
    module = jt.compile_custom_ops(
        [str(header), str(source)], extra_flags=f' -I"{tmp_path}" ',
        gen_name_="test_binding_only", return_module=True)
    assert module.increment(41) == 42
    assert not hasattr(module.ops, "increment")
