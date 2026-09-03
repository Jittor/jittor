// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <limits>
#ifdef __linux__
#include <sched.h>
#endif

#include "parallel_compiler.h"

namespace jittor {

JIT_TEST(parallel_compiler_worker_limits) {
    ASSERTop(parse_parallel_compile_cpu_max("max 100000"),==,
        std::numeric_limits<int>::max());
    ASSERTop(parse_parallel_compile_cpu_max("200000 100000"),==,2);
    ASSERTop(parse_parallel_compile_cpu_max("150000 100000"),==,2);
    ASSERTop(parse_parallel_compile_cpu_max("50000 100000"),==,1);
    ASSERTop(parse_parallel_compile_cpu_max("invalid 100000"),==,
        std::numeric_limits<int>::max());

    int workers = parallel_compile_worker_count(128);
    ASSERTop(workers,>=,1);
    ASSERTop(workers,<=,128);
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CHECKop(sched_getaffinity(0, sizeof(affinity), &affinity),==,0);
    ASSERTop(workers,<=,CPU_COUNT(&affinity));
#endif
}

} // namespace jittor
