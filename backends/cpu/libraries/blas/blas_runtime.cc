// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <climits>
#include <cstring>
#include <omp.h>

#include "blas_runtime.h"
#include "utils/log.h"

#if !defined(JT_BLAS_CGEMM) || !defined(JT_BLAS_ZGEMM) || !defined(JT_BLAS_INT)
#error "setup_blas passes JT_BLAS_CGEMM, JT_BLAS_ZGEMM and JT_BLAS_INT"
#endif

// The CBLAS prototypes, declared here rather than taken from a cblas.h: the
// library is found at run time (numpy's bundled OpenBLAS exports them as
// `cblas_cgemm64_`, with 64-bit integers), so there is no header to match it.
// The enum arguments are C enums, i.e. ints.
extern "C" {
void JT_BLAS_CGEMM(int order, int trans_a, int trans_b, JT_BLAS_INT m, JT_BLAS_INT n,
                   JT_BLAS_INT k, const void* alpha, const void* a, JT_BLAS_INT lda,
                   const void* b, JT_BLAS_INT ldb, const void* beta, void* c, JT_BLAS_INT ldc);
void JT_BLAS_ZGEMM(int order, int trans_a, int trans_b, JT_BLAS_INT m, JT_BLAS_INT n,
                   JT_BLAS_INT k, const void* alpha, const void* a, JT_BLAS_INT lda,
                   const void* b, JT_BLAS_INT ldb, const void* beta, void* c, JT_BLAS_INT ldc);
#ifdef JT_BLAS_SET_THREADS
void JT_BLAS_SET_THREADS(int threads);
int JT_BLAS_GET_THREADS();
#endif
}

namespace jittor {

namespace {
constexpr int kRowMajor = 101, kNoTrans = 111, kTrans = 112;

// Below this many complex multiply-adds a product runs on the calling thread:
// waking a team costs more than it saves.
constexpr int64 kSerialWork = 1 << 15;
// A panel narrower than this wastes the library's register blocking.
constexpr int64 kMinPanel = 32;
// Products with an inner dimension this short, or this little arithmetic per
// matrix, are memory- or call-bound: the library adds packing and, called
// from every thread at once, contends on its buffer allocator (numpy's
// OpenBLAS took 16 ms for 4096 4x4 products, torch 0.2 ms). They take the
// direct loop below instead.
constexpr int64 kDirectInner = 16;
constexpr int64 kDirectWork = 16 * 16 * 16;

// c[r0:r1, c0:c1] = op(a) @ op(b) as plain loops over interleaved (re, im)
// pairs; the innermost loop runs along a row of C, contiguous unless B is
// transposed.
// `__restrict__` and `omp simd` are what let the compiler vectorise the j
// loop at all: without them it must assume a row of C may alias a row of B.
template <class Real>
void direct(const Real* __restrict__ a, const Real* __restrict__ b, Real* __restrict__ c,
            int64 m, int64 lda, int64 ldb, int64 ldc,
            bool ta, bool tb, int64 r0, int64 r1, int64 c0, int64 c1) {
    for (int64 i = r0; i < r1; i++) {
        Real* __restrict__ crow = c + 2 * (i * ldc);
        #pragma omp simd
        for (int64 j = 2 * c0; j < 2 * c1; j++) crow[j] = 0;
        for (int64 p = 0; p < m; p++) {
            const Real* ap = a + 2 * (ta ? p * lda + i : i * lda + p);
            const Real ar = ap[0], ai = ap[1];
            if (!tb) {
                const Real* __restrict__ brow = b + 2 * (p * ldb);
                #pragma omp simd
                for (int64 j = c0; j < c1; j++) {
                    Real br = brow[2 * j], bi = brow[2 * j + 1];
                    crow[2 * j] += ar * br - ai * bi;
                    crow[2 * j + 1] += ar * bi + ai * br;
                }
            } else {
                #pragma omp simd
                for (int64 j = c0; j < c1; j++) {
                    const Real* bp = b + 2 * (j * ldb + p);
                    crow[2 * j] += ar * bp[0] - ai * bp[1];
                    crow[2 * j + 1] += ar * bp[1] + ai * bp[0];
                }
            }
        }
    }
}

struct Gemm {
    bool is_double, ta, tb, use_direct;
    int64 n, m, k;
    size_t elem;
    JT_BLAS_INT lda, ldb, ldc;

    // c[r0:r1, c0:c1] of one matrix. Row-major: rows of op(A) are rows of A,
    // or columns of A when transposed; columns of op(B) likewise for B.
    void run(const char* a, const char* b, char* c, int64 r0, int64 r1, int64 c0, int64 c1) const {
        if (r1 <= r0 || c1 <= c0) return;
        if (use_direct) {
            if (is_double)
                direct((const double*)a, (const double*)b, (double*)c, m, lda, ldb, ldc,
                       ta, tb, r0, r1, c0, c1);
            else
                direct((const float*)a, (const float*)b, (float*)c, m, lda, ldb, ldc,
                       ta, tb, r0, r1, c0, c1);
            return;
        }
        static const double one[2] = {1, 0}, zero[2] = {0, 0};
        static const float onef[2] = {1, 0}, zerof[2] = {0, 0};
        const char* ap = a + size_t(ta ? r0 : r0 * lda) * elem;
        const char* bp = b + size_t(tb ? c0 * ldb : c0) * elem;
        char* cp = c + size_t(r0 * ldc + c0) * elem;
        int opa = ta ? kTrans : kNoTrans, opb = tb ? kTrans : kNoTrans;
        if (is_double)
            JT_BLAS_ZGEMM(kRowMajor, opa, opb, r1 - r0, c1 - c0, m, one, ap, lda, bp, ldb,
                          zero, cp, ldc);
        else
            JT_BLAS_CGEMM(kRowMajor, opa, opb, r1 - r0, c1 - c0, m, onef, ap, lda, bp, ldb,
                          zerof, cp, ldc);
    }
};

// Run `batch` products on Jittor's own OpenMP team, the library kept to one
// thread for the duration.
//
// The library's own threading is a second pool beside Jittor's: numpy's
// OpenBLAS keeps pthreads that busy-wait after every call, while Jittor's
// libgomp team busy-waits between kernels, and on a machine with as many
// threads as cores each pool spins on the cores the other needs. Measured on
// 16 cores, a complex64 1024-cube forward+backward step took 130-200 ms that
// way against 28 ms with one pool (OMP_WAIT_POLICY=PASSIVE stood in for it);
// the GEMM itself went from 7.6 to 18 ms and the elementwise kernels around
// it by 30x. torch has one pool because MKL runs on its OpenMP.
//
// So the work is split here instead: across the batch when it is at least as
// large as the team, otherwise each product into panels of rows or columns
// of C -- whichever dimension is longer -- each a single-threaded GEMM on its
// own block. Every thread packs its own slice of the shared operand; for the
// sizes that split, that is O(n m) against O(n m k / threads) of arithmetic.
void run_team(const Gemm& g, int64 batch, const char* a, const char* b, char* c) {
    const int64 stride_a = g.n * g.m, stride_b = g.m * g.k, stride_c = g.n * g.k;
    int threads = omp_get_max_threads();
    int64 work = batch * g.n * g.m * g.k;
    if (threads <= 1 || work <= kSerialWork) {
        for (int64 i = 0; i < batch; i++)
            g.run(a + size_t(i * stride_a) * g.elem, b + size_t(i * stride_b) * g.elem,
                  c + size_t(i * stride_c) * g.elem, 0, g.n, 0, g.k);
        return;
    }
    // Panels per product, so that batch * panels covers the team.
    int64 panels = 1;
    bool split_rows = g.n >= g.k;
    int64 extent = split_rows ? g.n : g.k;
    if (batch < threads) {
        panels = (threads + batch - 1) / batch;
        panels = std::max<int64>(1, std::min(panels, g.use_direct ? extent : extent / kMinPanel));
    }
    int64 panel = (extent + panels - 1) / panels;
    int64 items = batch * panels;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64 item = 0; item < items; item++) {
        int64 i = item / panels, p = item % panels;
        int64 lo = p * panel, hi = std::min(extent, lo + panel);
        const char* ai = a + size_t(i * stride_a) * g.elem;
        const char* bi = b + size_t(i * stride_b) * g.elem;
        char* ci = c + size_t(i * stride_c) * g.elem;
        if (split_rows)
            g.run(ai, bi, ci, lo, hi, 0, g.k);
        else
            g.run(ai, bi, ci, 0, g.n, lo, hi);
    }
}
} // namespace

void blas_gemm_execute(NanoString dtype, int64 batch, int64 n, int64 m, int64 k,
                       bool ta, bool tb, const void* a, const void* b, void* c) {
    ASSERT(blas_gemm_dtype(dtype)) << "blas gemm got" << dtype;
    if (!batch || !n || !k) return;
    if (!m) {
        std::memset(c, 0, size_t(batch * n * k) * dtype.dsize());
        return;
    }
    // An LP64 CBLAS takes 32-bit dimensions; a product past them must not be
    // truncated into a wrong one.
    if (sizeof(JT_BLAS_INT) < sizeof(int64))
        USER_CHECK(n <= INT_MAX && m <= INT_MAX && k <= INT_MAX)
            << "the linked CBLAS takes 32-bit dimensions, but this product is"
            << n << "x" << m << "x" << k;
    Gemm g;
    g.is_double = dtype == ns_complex128;
    g.ta = ta;
    g.tb = tb;
    g.n = n;
    g.m = m;
    g.k = k;
    g.elem = dtype.dsize();
    // Row-major: the leading dimension is the stored row length.
    g.lda = ta ? n : m;
    g.ldb = tb ? m : k;
    g.ldc = k;
    g.use_direct = m <= kDirectInner || n * m * k <= kDirectWork;
    if (g.use_direct) {
        // The library is not called, so its thread count is left alone: the
        // first `openblas_set_num_threads` in a process starts OpenBLAS's
        // lazily created pool, whose new threads busy-wait for ~0.1-0.15 s
        // and slow every CPU kernel in that window. Tiny contractions -- most
        // of a 1-D circuit -- never pay that.
        run_team(g, batch, (const char*)a, (const char*)b, (char*)c);
        return;
    }
#ifdef JT_BLAS_SET_THREADS
    int library_threads = JT_BLAS_GET_THREADS();
    JT_BLAS_SET_THREADS(1);
    run_team(g, batch, (const char*)a, (const char*)b, (char*)c);
    JT_BLAS_SET_THREADS(library_threads);
#else
    // A library without OpenBLAS's thread controls (MKL, a system CBLAS)
    // threads each call itself.
    for (int64 i = 0; i < batch; i++)
        g.run((const char*)a + size_t(i * n * m) * g.elem, (const char*)b + size_t(i * m * k) * g.elem,
              (char*)c + size_t(i * n * k) * g.elem, 0, n, 0, k);
#endif
}

} // jittor
