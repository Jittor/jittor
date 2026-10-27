// Stage-1 probe: fused AscendC elementwise kernel vs N aclnn launches.
//   kernel A: z = a + b                    (1 aclnn op)
//   kernel B: z = (a + b) * a - b          (3 aclnn ops)
// Reports host-queue cost per call and steady-state per-iteration time.
#include "kernel_operator.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sub.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace AscendC;

constexpr int32_t TILE = 4096;      // elements per tile (fp32 -> 16KB per buffer)
constexpr int32_t BUFN = 2;         // double buffering

__aicore__ inline uint32_t ceil8(uint32_t x) { return (x + 7) / 8 * 8; }

// ---------------- kernel A: z = a + b ----------------
extern "C" __global__ __aicore__ void k_add(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n)
{
    uint32_t blocks = GetBlockNum();
    uint32_t bid = GetBlockIdx();
    uint32_t per = ((n + blocks - 1) / blocks + 63) / 64 * 64;
    uint32_t start = bid * per;
    if (start >= n) return;
    uint32_t len = n - start < per ? n - start : per;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFN> qa, qb;
    TQue<QuePosition::VECOUT, BUFN> qc;
    pipe.InitBuffer(qa, BUFN, TILE * sizeof(float));
    pipe.InitBuffer(qb, BUFN, TILE * sizeof(float));
    pipe.InitBuffer(qc, BUFN, TILE * sizeof(float));

    GlobalTensor<float> ga, gb, gc;
    ga.SetGlobalBuffer((__gm__ float*)a + start, len);
    gb.SetGlobalBuffer((__gm__ float*)b + start, len);
    gc.SetGlobalBuffer((__gm__ float*)c + start, len);

    for (uint32_t off = 0; off < len; off += TILE) {
        uint32_t cur = len - off < (uint32_t)TILE ? len - off : (uint32_t)TILE;
        uint32_t pad = ceil8(cur);
        LocalTensor<float> la = qa.AllocTensor<float>();
        LocalTensor<float> lb = qb.AllocTensor<float>();
        DataCopy(la, ga[off], pad);
        DataCopy(lb, gb[off], pad);
        qa.EnQue(la); qb.EnQue(lb);
        la = qa.DeQue<float>(); lb = qb.DeQue<float>();
        LocalTensor<float> lc = qc.AllocTensor<float>();
        Add(lc, la, lb, cur);
        qa.FreeTensor(la); qb.FreeTensor(lb);
        qc.EnQue(lc);
        lc = qc.DeQue<float>();
        DataCopy(gc[off], lc, pad);
        qc.FreeTensor(lc);
    }
}

// ---------------- kernel B: z = (a+b)*a - b ----------------
extern "C" __global__ __aicore__ void k_chain3(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n)
{
    uint32_t blocks = GetBlockNum();
    uint32_t bid = GetBlockIdx();
    uint32_t per = ((n + blocks - 1) / blocks + 63) / 64 * 64;
    uint32_t start = bid * per;
    if (start >= n) return;
    uint32_t len = n - start < per ? n - start : per;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFN> qa, qb;
    TQue<QuePosition::VECOUT, BUFN> qc;
    pipe.InitBuffer(qa, BUFN, TILE * sizeof(float));
    pipe.InitBuffer(qb, BUFN, TILE * sizeof(float));
    pipe.InitBuffer(qc, BUFN, TILE * sizeof(float));

    GlobalTensor<float> ga, gb, gc;
    ga.SetGlobalBuffer((__gm__ float*)a + start, len);
    gb.SetGlobalBuffer((__gm__ float*)b + start, len);
    gc.SetGlobalBuffer((__gm__ float*)c + start, len);

    for (uint32_t off = 0; off < len; off += TILE) {
        uint32_t cur = len - off < (uint32_t)TILE ? len - off : (uint32_t)TILE;
        uint32_t pad = ceil8(cur);
        LocalTensor<float> la = qa.AllocTensor<float>();
        LocalTensor<float> lb = qb.AllocTensor<float>();
        DataCopy(la, ga[off], pad);
        DataCopy(lb, gb[off], pad);
        qa.EnQue(la); qb.EnQue(lb);
        la = qa.DeQue<float>(); lb = qb.DeQue<float>();
        LocalTensor<float> lc = qc.AllocTensor<float>();
        Add(lc, la, lb, cur);          // t = a + b
        Mul(lc, lc, la, cur);          // t = t * a
        Sub(lc, lc, lb, cur);          // t = t - b
        qa.FreeTensor(la); qb.FreeTensor(lb);
        qc.EnQue(lc);
        lc = qc.DeQue<float>();
        DataCopy(gc[off], lc, pad);
        qc.FreeTensor(lc);
    }
}

// ---------------- empty kernel: pure launch floor ----------------
extern "C" __global__ __aicore__ void k_empty(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n)
{
    if (GetBlockIdx() == 0xffffffff) { GlobalTensor<float> g; g.SetGlobalBuffer((__gm__ float*)c, 1); }
}

// =================== host ===================
#define CHECK(expr, what) do { auto _s = (expr); if (_s != 0) { \
    std::printf("%s failed: %d\n", what, (int)_s); std::exit(1); } } while (0)

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

struct AclChain {
    aclTensor *ta, *tb, *tt, *tc;
    aclScalar *alpha;
    void *wsp; uint64_t wsp_cap;
    aclrtStream stream;
};

int main(int argc, char **argv) {
    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    int blockdim = getenv("BLOCKDIM") ? atoi(getenv("BLOCKDIM")) : 48;
    std::printf("blockDim=%d\n", blockdim);

    const int sides[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const int nsides = sizeof(sides) / sizeof(sides[0]);

    size_t maxbytes = (size_t)4096 * 4096 * sizeof(float);
    void *da, *db, *dc, *dt, *wsp;
    CHECK(aclrtMalloc(&da, maxbytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc a");
    CHECK(aclrtMalloc(&db, maxbytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc b");
    CHECK(aclrtMalloc(&dc, maxbytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc c");
    CHECK(aclrtMalloc(&dt, maxbytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc t");
    CHECK(aclrtMalloc(&wsp, 128u << 20, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");

    std::vector<float> ha, hb, hc, ref;
    float one = 1.0f;
    aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);

    std::printf("%8s %10s | %10s %10s | %10s %10s | %10s %10s | %s\n",
                "shape", "n", "aclnn1", "asc1", "aclnn3", "asc3", "speed1", "speed3", "maxerr3");

    for (int si = 0; si < nsides; si++) {
        int s = sides[si];
        uint32_t n = (uint32_t)s * s;
        size_t bytes = (size_t)n * sizeof(float);
        ha.resize(n); hb.resize(n); hc.resize(n); ref.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            ha[i] = std::sin(i * 0.001f) * 3.0f;
            hb[i] = std::cos(i * 0.0007f) * 2.0f;
            ref[i] = (ha[i] + hb[i]) * ha[i] - hb[i];
        }
        CHECK(aclrtMemcpy(da, bytes, ha.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "cpy a");
        CHECK(aclrtMemcpy(db, bytes, hb.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "cpy b");

        std::vector<int64_t> shape{s, s};
        std::vector<int64_t> stride{s, 1};
        auto make = [&](void *p) {
            return aclCreateTensor(shape.data(), 2, ACL_FLOAT, stride.data(), 0,
                                   ACL_FORMAT_ND, shape.data(), 2, p);
        };

        // ---- correctness of fused 3-op kernel ----
        CHECK(aclrtMemset(dc, bytes, 0, bytes), "memset c");
        k_chain3<<<blockdim, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, n);
        CHECK(aclrtSynchronizeStream(stream), "sync chk");
        CHECK(aclrtMemcpy(hc.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "cpy back");
        double maxerr = 0;
        for (uint32_t i = 0; i < n; i++) {
            double e = std::fabs((double)hc[i] - (double)ref[i]);
            double d = std::fabs((double)ref[i]);
            double rel = d > 1e-6 ? e / d : e;
            if (rel > maxerr) maxerr = rel;
        }

        const int warm = 20, iters = 200;

        // ---- aclnn 1 op ----
        auto aclnn1 = [&]() {
            aclTensor *ta = make(da), *tb = make(db), *tc = make(dc);
            uint64_t ws = 0; aclOpExecutor *ex = nullptr;
            aclnnAddGetWorkspaceSize(ta, tb, alpha, tc, &ws, &ex);
            aclnnAdd(ws ? wsp : nullptr, ws, ex, stream);
            aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
        };
        // ---- aclnn 3 op chain: t=a+b ; t=t*a ; c=t-b ----
        auto aclnn3 = [&]() {
            aclTensor *ta = make(da), *tb = make(db), *tt = make(dt), *tc = make(dc);
            uint64_t ws = 0; aclOpExecutor *ex = nullptr;
            aclnnAddGetWorkspaceSize(ta, tb, alpha, tt, &ws, &ex);
            aclnnAdd(ws ? wsp : nullptr, ws, ex, stream);
            aclTensor *tt2 = make(dt), *ta2 = make(da), *tt3 = make(dt);
            ws = 0; ex = nullptr;
            aclnnMulGetWorkspaceSize(tt2, ta2, tt3, &ws, &ex);
            aclnnMul(ws ? wsp : nullptr, ws, ex, stream);
            aclTensor *tt4 = make(dt), *tb2 = make(db), *tc2 = make(dc);
            ws = 0; ex = nullptr;
            aclnnSubGetWorkspaceSize(tt4, tb2, alpha, tc2, &ws, &ex);
            aclnnSub(ws ? wsp : nullptr, ws, ex, stream);
            aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tt); aclDestroyTensor(tc);
            aclDestroyTensor(tt2); aclDestroyTensor(ta2); aclDestroyTensor(tt3);
            aclDestroyTensor(tt4); aclDestroyTensor(tb2); aclDestroyTensor(tc2);
        };

        auto bench = [&](void (*none)(), auto &&fn) {
            for (int i = 0; i < warm; i++) fn();
            CHECK(aclrtSynchronizeStream(stream), "warm sync");
            double t0 = now_us();
            for (int i = 0; i < iters; i++) fn();
            CHECK(aclrtSynchronizeStream(stream), "sync");
            return (now_us() - t0) / iters;
        };

        double t_aclnn1 = bench(nullptr, aclnn1);
        double t_aclnn3 = bench(nullptr, aclnn3);
        double t_asc1 = bench(nullptr, [&]() {
            k_add<<<blockdim, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, n);
        });
        double t_asc3 = bench(nullptr, [&]() {
            k_chain3<<<blockdim, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, n);
        });

        std::printf("%4dx%-4d %10u | %9.2f %9.2f | %9.2f %9.2f | %9.2fx %9.2fx | %.2e\n",
                    s, s, n, t_aclnn1, t_asc1, t_aclnn3, t_asc3,
                    t_aclnn1 / t_asc1, t_aclnn3 / t_asc3, maxerr);
        fflush(stdout);
    }

    // ---- pure launch floor ----
    {
        const int iters = 2000;
        for (int i = 0; i < 50; i++) k_empty<<<blockdim, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, 0);
        CHECK(aclrtSynchronizeStream(stream), "sync");
        double t0 = now_us();
        for (int i = 0; i < iters; i++) k_empty<<<blockdim, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, 0);
        double t_queue = (now_us() - t0) / iters;
        CHECK(aclrtSynchronizeStream(stream), "sync");
        double t_full = (now_us() - t0) / iters;
        std::printf("\nempty AscendC kernel: host-queue %.3f us/launch, incl. device drain %.3f us/launch (blockDim=%d)\n",
                    t_queue, t_full, blockdim);
        for (int bd : {1, 8, 16, 24, 40, 48, 56}) {
            for (int i = 0; i < 50; i++) k_empty<<<bd, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, 0);
            CHECK(aclrtSynchronizeStream(stream), "sync");
            double s0 = now_us();
            for (int i = 0; i < iters; i++) k_empty<<<bd, nullptr, stream>>>((GM_ADDR)da, (GM_ADDR)db, (GM_ADDR)dc, 0);
            CHECK(aclrtSynchronizeStream(stream), "sync");
            std::printf("   blockDim=%2d empty kernel round-trip %.3f us\n", bd, (now_us() - s0) / iters);
        }
    }
    return 0;
}
