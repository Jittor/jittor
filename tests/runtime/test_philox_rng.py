import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_philox4x32_10_matches_random123_known_answers():
    # Random123 upstream KAT vectors:
    # https://raw.githubusercontent.com/DEShawResearch/random123/main/tests/kat_vectors
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        pytest.skip("a C++14 compiler is required for the Philox host contract")
    source = r'''
#include <cstdint>
#include <iomanip>
#include <iostream>
#include "utils/philox.h"

static void emit(uint64_t seed, uint64_t counter_low, uint64_t counter_high) {
    auto value = jittor::philox4x32_10(seed, counter_low, counter_high);
    std::cout << std::hex << std::setfill('0')
              << std::setw(8) << value.x0 << " "
              << std::setw(8) << value.x1 << " "
              << std::setw(8) << value.x2 << " "
              << std::setw(8) << value.x3 << "\n";
}

static void emit_mul(uint64_t left, uint64_t right) {
    auto value = jittor::philox_mul_wide(left, right);
    std::cout << std::hex << std::setfill('0')
              << std::setw(16) << value.low << " "
              << std::setw(16) << value.high << "\n";
}

int main() {
    emit(0, 0, 0);
    emit(UINT64_MAX, UINT64_MAX, UINT64_MAX);
    emit(UINT64_C(0x299f31d0a4093822),
         UINT64_C(0x85a308d3243f6a88),
         UINT64_C(0x0370734413198a2e));
    emit_mul(0, UINT64_MAX);
    emit_mul(UINT64_MAX, UINT64_MAX);
    emit_mul(UINT64_C(0x0123456789abcdef),
             UINT64_C(0xfedcba9876543210));
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-philox-kat-") as directory:
        directory = Path(directory)
        unit = directory / "philox_kat.cc"
        executable = directory / "philox_kat"
        unit.write_text(source)
        compiled = subprocess.run(
            [compiler, "-std=c++14", "-O2", "-Wall", "-Wextra",
             "-I", str(ROOT / "src"), str(unit), "-o", str(executable)],
            text=True, capture_output=True)
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        completed = subprocess.run([str(executable)], text=True, capture_output=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[:3] == [
        "6627e8d5 e169c58d bc57ac4c 9b00dbd8",
        "408f276d 41c83b0e a20bc7c6 6d5451fd",
        "d16cfe09 94fdcceb 5001e420 24126ea1",
    ]
    factors = [
        (0, (1 << 64) - 1),
        ((1 << 64) - 1, (1 << 64) - 1),
        (0x0123456789abcdef, 0xfedcba9876543210),
    ]
    mask = (1 << 64) - 1
    expected_products = [
        "%016x %016x" % ((left * right) & mask, (left * right) >> 64)
        for left, right in factors
    ]
    assert lines[3:] == expected_products
