"""KV cache updates must retain the previous cache producer as an input."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'backends/acl/kernels/ops/flashattention_op.py'


def test_memcpy_wrapper_preserves_cache_dependency_and_storage_target():
    calls = []
    class Function:
        def __call__(self, *args):
            return self.execute(*args)
    def recording_code(name, inputs, **kwargs):
        assert inputs[2].dense
        calls.append((name, inputs, kwargs))
        return kwargs['outputs']
    namespace = {
        'jt': SimpleNamespace(Function=Function),
        'flashattention_cmd': recording_code,
        'attribute_program': lambda name, values, **kwargs: (name, values),
        'code_program': lambda fragments: fragments,
    }
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'KVCacheMemcpyACL']
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    class Cache:
        dense = False
        def _storage_is_contiguous(self):
            return self.dense
        def contiguous(self):
            return 'materialized-cache'
        def update(self, value):
            assert value == 'materialized-cache'
            self.dense = True
    key, value, cache = object(), object(), Cache()
    result = namespace['KVCacheMemcpyACL'](128, [1, 129])(key, value, cache)
    assert result is cache
    name, inputs, kwargs = calls.pop()
    assert name == 'KVCacheMemcpy'
    # outputs= requests storage aliasing but cannot keep a lazy initializer or
    # an earlier partial update alive. A real input edge owns that dependency.
    assert inputs == [key, value, cache]
    assert kwargs['outputs'] == [cache]
    # Check the production wrapper forwards both blocks' slots unchanged.
    _, attributes = kwargs['attr_code'][1]
    assert attributes == {'blockSize': 128, 'slots': [1, 129]}


def test_native_memcpy_preserves_unwritten_cache_for_shared_and_separate_buffers(tmp_path):
    import os
    import shlex
    import subprocess
    source = (ROOT / 'backends/acl/kernels/native/flashattention_op_acl.cc').read_text()
    marker = 'void KVCacheMemcpyOpRunner::executeOp('
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 0
    for end in range(opening, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            break
    method = source[start:end + 1]
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using std::string; using std::vector;
using AclOpRegistry = std::map<string, int>;
struct Attr { virtual ~Attr() = default; };
struct KVCacheMemcpyAttr : Attr { int64_t blockSize; vector<int64_t> slots; };
struct Var {
    void* mem_ptr; int64_t size;
    bool is_contiguous() const { return true; }
};
constexpr int ACL_SUCCESS = 0, ACL_MEMCPY_DEVICE_TO_DEVICE = 1;
int aclstream = 3;
vector<size_t> copies;
int aclrtMemcpyAsync(void* dst, size_t capacity, const void* src,
                     size_t bytes, int kind, int stream) {
    assert(capacity >= bytes && kind == 1 && stream == 3);
    copies.push_back(bytes); std::memcpy(dst, src, bytes); return 0;
}
#define CHECK assert
std::ostringstream diagnostic;
#define LOGf diagnostic
struct KVCacheMemcpyOpRunner {
    string name = "KVCacheMemcpy";
    std::unique_ptr<Attr> op_attr;
    vector<Var*> in_, out_;
    vector<vector<int64_t>> inputShapes, outputShapes;
    int ret;
    void executeOp(AclOpRegistry::const_iterator&);
};
'''
    checks = r'''
int main() {
    for (bool shared : {false, true}) {
        vector<float> previous(1024, 0), output(1024, -999);
        previous[8] = 13; // an earlier update in a row not touched this time
        vector<float> key{1, 2, 3, 4}, value{5, 6, 7, 8};
        Var k{key.data(), 16}, v{value.data(), 16}, old{previous.data(), 4096};
        Var next{shared ? previous.data() : output.data(), 4096};
        KVCacheMemcpyOpRunner runner;
        auto attr = new KVCacheMemcpyAttr(); attr->blockSize = 128; attr->slots = {1, 129};
        runner.op_attr.reset(attr);
        runner.in_ = {&k, &v, &old}; runner.out_ = {&next};
        runner.inputShapes = {{2,1,2}, {2,1,2}, {2,2,128,1,2}};
        runner.outputShapes = {{2,2,128,1,2}};
        AclOpRegistry registry; auto iterator = registry.cend();
        copies.clear(); runner.executeOp(iterator);
        vector<float> expected(1024, 0); expected[8] = 13;
        expected[2]=1; expected[3]=2; expected[258]=5; expected[259]=6;
        expected[514]=3; expected[515]=4; expected[770]=7; expected[771]=8;
        auto* actual = static_cast<float*>(next.mem_ptr);
        for (size_t i=0; i<expected.size(); ++i) assert(actual[i] == expected[i]);
        assert(copies.size() == (shared ? 4 : 5));
        if (!shared) assert(copies.front() == 4096);
    }
}
'''
    unit = tmp_path / 'kv_cache_memcpy.cc'
    unit.write_text(preamble + method + checks)
    executable = tmp_path / 'kv_cache_memcpy'
    result = subprocess.run([*shlex.split(os.environ.get('CXX', 'g++')), '-std=c++14',
                             str(unit), '-o', str(executable)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
