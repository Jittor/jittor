// Stub <mpi.h> for the MPI-free NCCL build (JT_NCCL_NO_MPI).
//
// jittor's include scanner (cache_compile) is not #ifdef-aware: it textually
// finds every #include -- including the `#include "mpi_wrapper.h"` that lives in
// the *compiled-out* #else branch of nccl_wrapper.h, and mpi_wrapper.h in turn
// `#include <mpi.h>`. So the scanner must be able to *locate* an mpi.h even when
// no MPI is installed. The compiler never actually includes this file (the
// #else branch is removed by -DJT_NCCL_NO_MPI), so it only needs to exist.
#pragma once
