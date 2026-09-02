// ***************************************************************
// Copyright (c) 2023 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

#include "mpi_wrapper.h"
#include "common.h"
#include "ops/array_op.h"
#include "misc/collective_dtype.h"

char jt_mpi_err_buffer[MPI_MAX_ERROR_STRING];

void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line) {
    int resultlen;
    MPI_Error_string(result, jt_mpi_err_buffer, &resultlen);
    LOGf << "MPI error at " >> file >> ":" >> line << "code="
        >> result >> '(' >> jt_mpi_err_buffer >> ')' << func;
}

void report_mpi_error(int result,
    char const *const func, const char *const file, int const line) {
    int resultlen;
    MPI_Error_string(result, jt_mpi_err_buffer, &resultlen);
    LOGe << "MPI teardown error at " >> file >> ":" >> line << "code="
        >> result >> '(' >> jt_mpi_err_buffer >> ')' << func;
}

namespace jittor {

MPI_Datatype MPI_HALF;
MPI_Op MPI_HALF_ADD;

// ---- fp16 <-> fp32 --------------------------------------------------------
//
// One scalar implementation is the definition of the MPI_HALF sum on every
// architecture. Before this, x86 and ARM had entirely separate code and the two
// did not agree, so the same all-reduce gave different numbers on different
// hosts:
//
//   * x86 used _mm256_cvtph_ps / _mm256_cvtps_ph -- IEEE round-to-nearest-even
//     with full subnormal support -- and never checked at run time that the CPU
//     actually has F16C and AVX. On a machine without them the build either
//     fails to compile or the binary dies with SIGILL.
//   * the ARM fallback *truncated* the mantissa instead of rounding (up to one
//     ulp different on almost every value), and flushed everything below 2^-14
//     to zero, so fp16 subnormals vanished on ARM and survived on x86.
//
// The x86 SIMD path is kept, but only as an accelerator that is bit-identical
// to the scalar code, entered only after a run-time CPUID check.
// JT_MPI_HALF_SIMD=0 forces the scalar path, which is how the tests run the
// exact code an ARM host would run while sitting on an x86 machine.

static inline float jt_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;                              // +-0
        } else {
            // Subnormal: normalize into an fp32 normal. fp32 has the range for
            // every fp16 subnormal, so nothing is lost here.
            uint32_t e = 127 - 15 + 1;
            while (!(man & 0x0400u)) { man <<= 1; e--; }
            man &= 0x03ffu;
            out = sign | (e << 23) | (man << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (man << 13);      // inf / nan (payload kept)
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// Round to nearest, ties to even -- the same rounding F16C does, and the same
// rounding numpy's float32->float16 cast does, so results are checkable.
static inline uint16_t jt_fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    uint32_t mag = x & 0x7fffffffu;

    if (mag > 0x7f800000u)                           // nan -> quiet nan
        return (uint16_t)(sign | 0x7e00u);
    if (mag >= 0x47800000u)                          // >= 65536 -> inf
        return (uint16_t)(sign | 0x7c00u);
    if (mag >= 0x38800000u) {                        // >= 2^-14: fp16 normal
        uint32_t out = (uint32_t)sign
                     | (((mag >> 23) - 112u) << 10)  // 127-15 = 112
                     | ((mag >> 13) & 0x03ffu);
        uint32_t rem = mag & 0x1fffu;                // bits being dropped
        // A carry out of the mantissa lands in the exponent, which is exactly
        // what we want -- including 65504 rounding up to inf.
        if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) out++;
        return (uint16_t)out;
    }
    if (mag >= 0x33000000u) {                        // >= 2^-25: fp16 subnormal
        uint32_t m = (mag & 0x007fffffu) | 0x00800000u;
        int shift = 126 - (int)(mag >> 23);          // 14 .. 24
        uint32_t out = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (out & 1u))) out++;
        return (uint16_t)(sign | out);
    }
    return sign;                                     // underflow to signed zero
}

static void half_add_scalar(const uint16_t* in, uint16_t* inout, int n) {
    for (int i = 0; i < n; i++)
        inout[i] = jt_fp32_to_fp16(
            jt_fp16_to_fp32(in[i]) + jt_fp16_to_fp32(inout[i]));
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>

// target(...) lets this compile even when the translation unit is not built
// with -mf16c, so the binary still runs on CPUs without it -- the call is
// guarded by the CPUID check below.
__attribute__((target("avx,f16c")))
static void half_add_f16c(const uint16_t* in, uint16_t* inout, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(in + i)));
        __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(inout + i)));
        _mm_storeu_si128((__m128i*)(inout + i),
            _mm256_cvtps_ph(_mm256_add_ps(a, b),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    // Tail through the scalar code rather than a second SIMD spelling, so
    // there is only one thing to keep in agreement.
    half_add_scalar(in + i, inout + i, n - i);
}

static bool detect_half_simd() {
    if (const char* env = getenv("JT_MPI_HALF_SIMD"))
        if (env[0] == '0') return false;
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const unsigned int OSXSAVE = 1u << 27, AVX = 1u << 28, F16C = 1u << 29;
    if ((ecx & (OSXSAVE | AVX | F16C)) != (OSXSAVE | AVX | F16C)) return false;
    // The OS must also have enabled XMM+YMM state saving, otherwise AVX faults.
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    return (xcr0_lo & 0x6u) == 0x6u;
}

static bool half_simd_enabled() {
    static const bool enabled = detect_half_simd();
    return enabled;
}
#define JT_HAS_HALF_SIMD 1
#endif // GNUC || clang
#endif // x86_64

void HalfAdd(void* invec, void* inoutvec, int* len, MPI_Datatype* type) {
    const uint16_t* in = (const uint16_t*)invec;
    uint16_t* inout = (uint16_t*)inoutvec;
    int n = *len;
#ifdef JT_HAS_HALF_SIMD
    if (half_simd_enabled()) {
        half_add_f16c(in, inout, n);
        return;
    }
#endif
    half_add_scalar(in, inout, n);
}

int mpi_world_size = 1;
int mpi_world_rank = 0;
int mpi_local_size = 1;
int mpi_local_rank = 0;
bool inside_mpi = false;
bool mpi_enabled = false;
bool use_device_mpi = false;

// The one MPI dtype table. Expanded from the canonical list in
// misc/collective_dtype.h so it cannot drift from NCCL's and HCCL's.
// MPI_HALF is created in mpi_initer below (a contiguous 1 x MPI_SHORT), so
// this must not be called before MPI init.
static MPI_Datatype mpi_dtype_unsupported(NanoString dtype) {
    LOGf << "MPI collectives do not support dtype" << dtype;
    return MPI_DATATYPE_NULL;
}

#define JT_MPI_DTYPE_float16  MPI_HALF
#define JT_MPI_DTYPE_bfloat16 mpi_dtype_unsupported(dtype)
#define JT_MPI_DTYPE_float32 MPI_FLOAT
#define JT_MPI_DTYPE_float64 MPI_DOUBLE
#define JT_MPI_DTYPE_int16   MPI_INT16_T
#define JT_MPI_DTYPE_int32   MPI_INT32_T
#define JT_MPI_DTYPE_int64   MPI_INT64_T
#define JT_MPI_DTYPE_uint8   MPI_UINT8_T

MPI_Datatype mpi_dtype(NanoString dtype) {
    #define JT_MPI_DTYPE_CASE(T) if (dtype == ns_##T) return JT_MPI_DTYPE_##T;
    JT_COLLECTIVE_DTYPES(JT_MPI_DTYPE_CASE)
    #undef JT_MPI_DTYPE_CASE
    return mpi_dtype_unsupported(dtype);
}

MPI_Op mpi_add_op(NanoString dtype) {
    // float16 is not an MPI predefined type, so MPI_SUM does not know how to
    // add it; MPI_HALF_ADD is our user-defined operator (see HalfAdd above).
    if (dtype == ns_float16) return MPI_HALF_ADD;
    // Validate the dtype through the same table, so an unsupported dtype fails
    // here too rather than reaching MPI with a sum operator and no datatype.
    mpi_dtype(dtype);
    return MPI_SUM;
}

int _mpi_world_size() {
    return mpi_enabled ? mpi_world_size : 1;
}

int _mpi_world_rank() {
    return mpi_enabled ? mpi_world_rank : 0;
}

int _mpi_local_rank() {
    return mpi_enabled ? mpi_local_rank : 0;
}

void _mpi_broadcast(ArrayArgs&& args, int root) {
    if (!mpi_enabled) return;
    int64 size = args.dtype.dsize();
    for (auto j : args.shape)
        size *= j;
    MPI_CHECK(MPI_Bcast((void *)args.ptr, size, MPI_BYTE, root, MPI_COMM_WORLD));
}

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

// Was `getenv("OMPI_COMM_WORLD_SIZE")` alone, i.e. Open MPI only: launched
// with MPICH, Intel MPI, MVAPICH or srun, jittor did not notice it was in an
// MPI job at all and every rank ran as an independent single-card process.
//
// This list is mirrored by compile_extern._MPI_LAUNCHER_VARS /
// _MPI_LAUNCHER_SIZE_VARS, because Python must answer the same question before
// this module is even loaded. tests/distributed/test_mpi_launcher_env.py
// asserts the two stay identical.
static bool detect_inside_mpi() {
    // Explicit declaration always wins, in either direction: JT_MPI=0 to stay
    // single-process under a launcher, JT_MPI=1 for a launcher we do not know.
    if (const char* forced = getenv("JT_MPI")) {
        if (forced[0])
            return !(forced[0] == '0' && forced[1] == '\0');
    }
    // Presence of these means "started by an MPI launcher".
    static const char* launcher_vars[] = {
        "OMPI_COMM_WORLD_SIZE",   // Open MPI (mpirun / orterun / prterun)
        "PMI_SIZE",               // MPICH, Intel MPI (mpiexec.hydra)
        "MV2_COMM_WORLD_SIZE",    // MVAPICH2
        "PMIX_RANK",              // PMIx-based launchers
    };
    for (auto* v : launcher_vars)
        if (getenv(v)) return true;
    // Slurm is different: srun is routinely used to start ordinary single-task
    // jobs that have nothing to do with MPI, so require an actual multi-task
    // allocation before deciding to call MPI_Init.
    static const char* slurm_size_vars[] = { "SLURM_NTASKS", "SLURM_NPROCS" };
    for (auto* v : slurm_size_vars) {
        const char* s = getenv(v);
        if (s && atoi(s) > 1) return true;
    }
    return false;
}

static bool mpi_initialized_here = false;

// See cublas_shutdown: report, never raise, and idempotent.
void mpi_shutdown() {
    if (!mpi_initialized_here) return;
    mpi_initialized_here = false;
    MPI_PEEK(MPI_Type_free(&MPI_HALF));
    MPI_PEEK(MPI_Op_free(&MPI_HALF_ADD));
    MPI_PEEK(MPI_Finalize());
}

struct mpi_initer {

mpi_initer() {
    inside_mpi = detect_inside_mpi();
    if (!inside_mpi) return;
    mpi_enabled = true;
    LOGvv << "MPI init...";
    // MPI may already be initialized by another component in the process
    // (e.g. mpi4py, used on Ascend to bring MPI up BEFORE the CANN libs load
    // and avoid an ABI/symbol clash). Double MPI_Init is a fatal error, so
    // only initialize if needed.
    int already_inited = 0;
    MPI_Initialized(&already_inited);
    if (!already_inited)
        MPI_CHECK(MPI_Init(NULL, NULL));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));

    //calculating localRank based on hostname which is used in selecting a GPU
    std::vector<uint64_t> host_hashes(mpi_world_size);
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hashes[mpi_world_rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        host_hashes.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    mpi_local_rank = 0;
    for (int p=0; p<mpi_world_size; p++) {
        if (p == mpi_world_rank) break;
        if (host_hashes[p] == host_hashes[mpi_world_rank]) mpi_local_rank++;
    }
    mpi_local_size = 0;
    for (int p=0; p<mpi_world_size; p++) {
        if (host_hashes[p] == host_hashes[mpi_world_rank]) mpi_local_size++;
    }
    LOGv << "MPI init finished: local" << mpi_local_rank
        << "size" << mpi_local_size
        << "global" << mpi_world_rank
        << "size" << mpi_world_size;
        
    // init mpi half type
    MPI_Type_contiguous(1, MPI_SHORT, &MPI_HALF);
    MPI_Type_commit(&MPI_HALF);
    MPI_Op_create(HalfAdd, /* commute= */1, &MPI_HALF_ADD);
    mpi_initialized_here = true;
}

~mpi_initer() {
    mpi_shutdown();
}

};

static mpi_initer mpi_init;


void var_broadcast(VarHolder* x, int root) {
    if (!inside_mpi) return;
    Var* v = x->var;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    ASSERT(root >= 0 && root < mpi_world_size)
        << "mpi var_broadcast: root" << root << "out of range for world_size"
        << mpi_world_size;
    int64 MPI_MAX_SIZE = 1ll<<30;
    for (int64 i=0; i<v->size; i+=MPI_MAX_SIZE) {
        int64 size = std::min(v->size-i, MPI_MAX_SIZE);
        // Was hardcoded 0 here, silently ignoring the root the caller asked
        // for: every rank ended up with rank 0's data and the real root's
        // data was overwritten, with no error.
        MPI_CHECK(MPI_Bcast(v->ptr<uint8>()+i, size, MPI_BYTE, root, MPI_COMM_WORLD));
    }
}

void var_reduce(VarHolder* x, int root) {
    if (!inside_mpi) return;
    Var* v = x->var;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    MPI_Datatype dtype = mpi_dtype(v->dtype());
    MPI_Op op = mpi_add_op(v->dtype());
    // mpi reduce performace magically reduce from 4194304
    int64 MPI_MAX_SIZE = (4194304) / v->dtype().dsize();
    for (int64 i=0; i<v->num; i+=MPI_MAX_SIZE) {
        int64 size = std::min(v->num-i, MPI_MAX_SIZE);
        auto mem_ptr = v->ptr<uint8>()+i*v->dtype().dsize();
        if (mpi_world_rank == root)
            MPI_Reduce(MPI_IN_PLACE, mem_ptr, size, dtype, op, root, MPI_COMM_WORLD);
        else
            MPI_Reduce(mem_ptr, nullptr, size, dtype, op, root, MPI_COMM_WORLD);
    }
}

void var_all_reduce(VarHolder* x) {
    if (!inside_mpi) return;
    Var* v = x->var;
    ASSERT(v->mem_ptr && !v->allocator->is_cuda());
    MPI_Datatype dtype = mpi_dtype(v->dtype());
    MPI_Op op = mpi_add_op(v->dtype());
    int64 MPI_MAX_SIZE = (1<<30) / v->dtype().dsize();
    for (int64 i=0; i<v->num; i+=MPI_MAX_SIZE) {
        int64 size = std::min(v->num-i, MPI_MAX_SIZE);
        auto mem_ptr = v->ptr<uint8>()+i*v->dtype().dsize();
        MPI_Allreduce(MPI_IN_PLACE, mem_ptr, size, dtype, op, MPI_COMM_WORLD);
    }
}

void mpi_barrier() {
    if (!inside_mpi) return;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

} // jittor
