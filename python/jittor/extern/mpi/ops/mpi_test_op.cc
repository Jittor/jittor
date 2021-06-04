// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mpi_warper.h"

#include "var.h"
#include "mpi_test_op.h"
#include "utils/str_utils.h"

namespace jittor {

#ifndef JIT
MpiTestOp::MpiTestOp(string cmd) : cmd(cmd) {
    output = create_output(1, ns_float32);
}

void MpiTestOp::jit_prepare(JK& jk) {
    jk << _CS("[T:float32]");
}

#else // JIT

void MpiTestOp::jit_run() {
    output->ptr<T>()[0] = 123;

    int world_size = mpi_world_size;

    int world_rank = mpi_world_rank;

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_CHECK(MPI_Get_processor_name(processor_name, &name_len));

    printf("Hello world from processor %s, rank %d out of %d processors\\n",processor_name, world_rank, world_size);

}

#endif // JIT

} // jittor
