"""Observe the OpenMP runtime directly; no tensor computation."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def test_thread_limit_tracks_native_openmp_control(tmp_path):
    source = tmp_path / "thread_limit.cc"
    source.write_text(r"""
#include <cassert>
#include <omp.h>
#include "runtime/threading.h"
int main() {
    assert(jittor::runtime_openmp_max_threads() == 2);
    omp_set_num_threads(3);
    assert(jittor::runtime_openmp_max_threads() == 3);
    omp_set_num_threads(5);
    assert(jittor::runtime_openmp_max_threads() == 5);
}
""", encoding="utf-8")
    binary = tmp_path / "thread_limit"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++14", "-fopenmp",
                    "-I" + str(ROOT / "src"), str(source),
                    str(ROOT / "src/runtime/threading.cc"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], env=dict(os.environ, OMP_NUM_THREADS="2"), check=True)
