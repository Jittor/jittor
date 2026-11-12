"""Execute the real host generator; this is not Ascend device acceptance."""
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'backends/acl/src/acl_fused_ascendc.cc'


@pytest.fixture(scope='module')
def generator(tmp_path_factory):
    text = SOURCE.read_text()
    def section(start, end):
        return text[text.index(start):text.index(end, text.index(start))]
    source = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using std::string;
using std::vector;
using int64 = int64_t;
using LaunchFn = void (*)(void**, uint32_t, uint32_t, void*);
enum NanoString { ns_float32, ns_float16, ns_bfloat16, ns_add, ns_subtract,
    ns_multiply, ns_divide, ns_maximum, ns_minimum, ns_abs, ns_negative, ns_cast };
string S(int n) { return std::to_string(n); }
'''
    source += section('constexpr int kMaxNodes', 'typedef void (*LaunchFn)')
    source += section('enum NodeKind', 'Var *locate(')
    source += section('const char *binary_intrinsic', 'bool var_is_plain(')
    source += section('int64 gcd64(', '// ---- ccec toolchain')
    source += r'''
int main(int argc, char** argv) {
    Plan p;
    auto dtype = static_cast<NanoString>(std::atoi(argv[1]));
    int period = std::atoi(argv[2]);
    p.input_period = {0, period};
    p.input_dtype = {ns_float32, dtype};
    p.output_dtype = {ns_float32};
    p.widen = dtype != ns_float32;
    p.wide_dtype = dtype;
    p.align = period > 1 ? period : 16;
    Node n{};
    n.kind = kBinary; n.ns = ns_add; n.nin = 2;
    n.src[0] = -1; n.src[1] = -2; n.out_mat = 0;
    n.dtype = ns_float32;
    p.nodes.push_back(n);
    int tile = 0;
    std::cout << generate_source(p, tile);
}
'''
    directory = tmp_path_factory.mktemp('ascendc-generator')
    cpp = directory / 'generator.cc'
    binary = directory / 'generator'
    cpp.write_text(source)
    subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17',
                    str(cpp), '-o', str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize('dtype', [0, 1, 2], ids=['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('period', [0, 1, 16, 256], ids=['dense', 'scalar', 'row16', 'row256'])
def test_generated_buffers_are_initialized_and_fit_ub(generator, dtype, period):
    source = subprocess.check_output([str(generator), str(dtype), str(period)], text=True)
    assert source, 'valid broadcast plan was declined'
    tile = int(re.search(r'JT_TILE = (\d+)', source)[1])
    declared = set(re.findall(r'(?:TBuf|TQue)<[^;]+> (\w+);', source))
    initialized = set()
    total = 0
    for match in re.finditer(r'pipe.InitBuffer\((\w+), (.*?)\);', source):
        name, args = match.groups()
        assert name not in initialized
        initialized.add(name)
        sizes = args.split(', ')
        values = []
        for size in sizes:
            factors = size.replace('JT_TILE', str(tile)).split(' * ')
            assert all(factor.isdigit() for factor in factors)
            value = 1
            for factor in factors:
                value *= int(factor)
            values.append(value)
        total += values[0] if len(values) == 1 else values[0] * values[1]
    used = set(re.findall(r'\b(\w+)\.(?:Get|AllocTensor)<', source))
    assert used <= initialized, f'unallocated buffers: {used - initialized}'
    assert declared == initialized
    assert total <= 160 * 1024, f'generated buffers exceed UB budget: {total}'
