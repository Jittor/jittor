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

namespace jittor {

MPI_Datatype MPI_HALF;
MPI_Op MPI_HALF_ADD;

#if !defined(__x86_64__) && !defined(_M_X64)
// ARM架构下的FP16-FP32转换辅助函数
static inline float fp16_to_fp32_value(uint16_t h) {
    unsigned sign = ((h >> 15) & 1);
    unsigned exponent = ((h >> 10) & 0x1f);
    unsigned mantissa = ((h & 0x3ff) << 13);
    
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // 非规格化数
            while (!(mantissa & 0x400000)) {
                mantissa <<= 1;
                exponent -= 1;
            }
            exponent += 1;
            mantissa &= ~0x400000;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    
    exponent += (127 - 15);
    mantissa <<= 10;
    
    unsigned int i = ((sign << 31) | (exponent << 23) | mantissa);
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

static inline uint16_t fp32_to_fp16_value(float f) {
    unsigned int i;
    std::memcpy(&i, &f, sizeof(float));
    
    unsigned sign = ((i >> 31) & 0x1);
    unsigned exponent = ((i >> 23) & 0xff);
    unsigned mantissa = (i & 0x7fffff);
    
    unsigned short h = 0;
    
    if (exponent == 0) {
        // 零或非规格化数
        h = (sign << 15);
    } else if (exponent == 0xff) {
        // 无穷大或NaN
        h = (sign << 15) | 0x7c00;
        if (mantissa) h |= 0x200;
    } else {
        // 规格化数
        int new_exp = exponent - 127 + 15;
        if (new_exp < 0) {
            // 下溢出到零
            h = (sign << 15);
        } else if (new_exp > 30) {
            // 上溢出到无穷大
            h = (sign << 15) | 0x7c00;
        } else {
            // 正常转换
            h = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
        }
    }
    
    return h;
}
#endif

void HalfAdd(void* invec, void* inoutvec, int* len, MPI_Datatype* type) {
#if defined(__x86_64__) || defined(_M_X64)
    short* in = (short*)invec;
    short* inout = (short*)inoutvec;

    int i = 0;
    int total = *len;
    for (; i+8 <= total; i += 8) {
        // 将半精度浮点数转换为单精度浮点数
        __m256 in1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
        __m256 in2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

        // 执行向量加法
        __m256 out = _mm256_add_ps(in1, in2);

        // 将单精度浮点数转换回半精度浮点数，并存储结果
        _mm_storeu_si128((__m128i*)(inout + i), _mm256_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // 处理剩余的半精度浮点数
    for (; i < total; i++) {
        // 将半精度浮点数转换为单精度浮点数
        __m128 in1 = _mm_cvtph_ps(_mm_set1_epi16(*(in + i)));
        __m128 in2 = _mm_cvtph_ps(_mm_set1_epi16(*(inout + i)));

        // 执行向量加法
        __m128 out = _mm_add_ps(in1, in2);

        // 将单精度浮点数转换回半精度浮点数，并存储结果
        *(inout + i) = _mm_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)[0];
    }
#else
    // ARM架构实现：使用基本的半精度浮点数运算
    uint16_t* in = (uint16_t*)invec;
    uint16_t* inout = (uint16_t*)inoutvec;
    int total = *len;
    
    // 简单的逐元素相加实现
    for (int i = 0; i < total; i++) {
        // 将FP16转换为FP32
        float in_val = fp16_to_fp32_value(in[i]);
        float inout_val = fp16_to_fp32_value(inout[i]);
        
        // 执行加法
        float result = in_val + inout_val;
        
        // 将结果转回FP16
        inout[i] = fp32_to_fp16_value(result);
    }
#endif
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

}

~mpi_initer() {
    if (!inside_mpi) return;
    MPI_Type_free(&MPI_HALF);
    MPI_Op_free(&MPI_HALF_ADD);
    MPI_CHECK(MPI_Finalize());
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
