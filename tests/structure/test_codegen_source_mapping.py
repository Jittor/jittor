"""Generated backend composition keeps the original C++ source locations."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_jit_annotation_respects_existing_source_line_directives(tmp_path):
    source = (ROOT / "python/jittor/src/op_compiler.cc").read_text()
    helpers = source[source.index("static string line_directive_path("):
                     source.index("DECLARE_FLAG(string, jittor_path);")]
    harness = r'''
#include <cassert>
#include <iomanip>
#include <sstream>
#include <string>
using string = std::string;
template<class T> string S(T value) { return std::to_string(value); }
HELPERS
#undef assert
#define assert(condition) do { if (!(condition)) return __LINE__; } while (0)
int main() {
    auto plain = annotate_jit_run_lines("// first\nvoid X::jit_run() {\n}\n", "/core/op.cc");
    assert(plain.find("#line 2 \"/core/op.cc\"") != string::npos);
    auto composed = annotate_jit_run_lines(
        "#line 10 \"/core/shared op.cc\"\n// first\nvoid X::jit_run() {\n}\n"
        "#line 7 \"/backend/helper.cc\"\nvoid Y::jit_run() {\n}\n",
        "/cache/composed.cc");
    assert(composed.find("#line 11 \"/core/shared op.cc\"") != string::npos);
    assert(composed.find("#line 7 \"/backend/helper.cc\"\nvoid Y::jit_run") != string::npos);
    assert(composed.find("/cache/composed.cc") == string::npos);
}
'''.replace("HELPERS", helpers)
    path = tmp_path / "source_mapping.cc"
    path.write_text(harness)
    binary = tmp_path / "source_mapping"
    compiled = subprocess.run(["g++", "-std=c++14", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
