// Stub <mpi.h> for the MPI-free NCCL build (JT_NCCL_NO_MPI).
//
// jittor's include scanner (cache_compile) is not #ifdef-aware: it textually
// finds every #include -- including the `#include "mpi_wrapper.h"` that lives in
// the *compiled-out* #else branch of nccl_wrapper.h, and mpi_wrapper.h in turn
// `#include <mpi.h>`. So the scanner must be able to *locate* an mpi.h even when
// no MPI is installed. The compiler never actually includes this file (the
// #else branch is removed by -DJT_NCCL_NO_MPI), so it only needs to exist.
//
// EXCEPTION: when jittor IS launched under mpirun (real collectives needed for
// multi-GPU tensor-parallel), JT_NCCL_NO_MPI is NOT defined, so mpi_wrapper.h's
// real-MPI code (MPI_SUCCESS, MPI_Bcast, ...) IS compiled — but this stub still
// sits ahead of the real <mpi.h> on the JIT op include path and would shadow it.
// Conditionally pull in the real header when it exists so the real-MPI build
// compiles; harmless under no-MPI (declarations only, not linked).
#pragma once
#define OMPI_SKIP_MPICXX
// Pull in the REAL <mpi.h> that jittor's build already put on the include path
// (mpi_compile_flags, derived from mpicc) -- this stub dir sits AHEAD of it and
// would shadow it. include_next resumes the header search AFTER this stub's dir,
// so it finds the real mpi.h portably (no hardcoded path). Guarded: under the
// no-MPI build there is no "next" mpi.h, so this expands to nothing and the
// MPI_Bcast bootstrap is compiled out by -DJT_NCCL_NO_MPI anyway.
#ifdef __has_include_next
#  if __has_include_next(<mpi.h>)
#    include_next <mpi.h>
#  endif
#endif
