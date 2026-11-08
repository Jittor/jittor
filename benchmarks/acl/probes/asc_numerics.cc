// Bit-exactness of AscendC vector intrinsics vs the aclnn ops jittor uses today,
// plus tail handling for element counts that are not a multiple of 8.
#include "kernel_operator.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_minimum.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_neg.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

using namespace AscendC;
constexpr int32_t TILE = 4096;

#define GEN_BIN(NAME, EXPR)                                                        \
extern "C" __global__ __aicore__ void NAME(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t n) \
{                                                                                  \
    uint32_t blocks = GetBlockNum(), bid = GetBlockIdx();                          \
    uint32_t chunk = ((n + blocks - 1) / blocks + 7) / 8 * 8;                      \
    uint32_t start = bid * chunk;                                                  \
    if (start >= n) return;                                                        \
    uint32_t len = (n - start) < chunk ? (n - start) : chunk;                      \
    TPipe pipe;                                                                    \
    TQue<QuePosition::VECIN, 2> qa, qb;                                            \
    TQue<QuePosition::VECOUT, 2> qc;                                               \
    pipe.InitBuffer(qa, 2, TILE * sizeof(float));                                  \
    pipe.InitBuffer(qb, 2, TILE * sizeof(float));                                  \
    pipe.InitBuffer(qc, 2, TILE * sizeof(float));                                  \
    GlobalTensor<float> ga, gb, gc;                                                \
    ga.SetGlobalBuffer((__gm__ float*)a + start, len);                             \
    gb.SetGlobalBuffer((__gm__ float*)b + start, len);                             \
    gc.SetGlobalBuffer((__gm__ float*)c + start, len);                             \
    for (uint32_t off = 0; off < len; off += TILE) {                               \
        uint32_t cur = len - off < (uint32_t)TILE ? len - off : (uint32_t)TILE;    \
        LocalTensor<float> la = qa.AllocTensor<float>();                           \
        LocalTensor<float> lb = qb.AllocTensor<float>();                           \
        if (cur % 8 == 0) { DataCopy(la, ga[off], cur); DataCopy(lb, gb[off], cur); } \
        else {                                                                     \
            DataCopyExtParams ep{1, cur * (uint32_t)sizeof(float), 0, 0, 0};       \
            DataCopyPadExtParams<float> pp{false, 0, 0, 0};                        \
            DataCopyPad(la, ga[off], ep, pp);                                      \
            DataCopyPad(lb, gb[off], ep, pp);                                      \
        }                                                                          \
        qa.EnQue(la); qb.EnQue(lb);                                                \
        la = qa.DeQue<float>(); lb = qb.DeQue<float>();                            \
        LocalTensor<float> lc = qc.AllocTensor<float>();                           \
        EXPR;                                                                      \
        qa.FreeTensor(la); qb.FreeTensor(lb);                                      \
        qc.EnQue(lc); lc = qc.DeQue<float>();                                      \
        if (cur % 8 == 0) DataCopy(gc[off], lc, cur);                              \
        else { DataCopyExtParams ep{1, cur * (uint32_t)sizeof(float), 0, 0, 0};    \
               DataCopyPad(gc[off], lc, ep); }                                     \
        qc.FreeTensor(lc);                                                         \
    }                                                                              \
}

GEN_BIN(k_add,  Add(lc, la, lb, cur))
GEN_BIN(k_sub,  Sub(lc, la, lb, cur))
GEN_BIN(k_mul,  Mul(lc, la, lb, cur))
GEN_BIN(k_div,  Div(lc, la, lb, cur))
GEN_BIN(k_max,  Max(lc, la, lb, cur))
GEN_BIN(k_min,  Min(lc, la, lb, cur))
GEN_BIN(k_abs,  Abs(lc, la, cur))
GEN_BIN(k_neg,  Muls(lc, la, -1.0f, cur))
GEN_BIN(k_exp,  Exp(lc, la, cur))
GEN_BIN(k_log,  Ln(lc, la, cur))
GEN_BIN(k_sqrt, Sqrt(lc, la, cur))

#define CHECK(e, w) do { auto _s=(e); if(_s!=0){ std::printf("%s failed %d\n", w, (int)_s); std::exit(1);} } while(0)

int main() {
    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "setdev");
    aclrtStream st; CHECK(aclrtCreateStream(&st), "stream");

    const uint32_t n = 1000003;   // deliberately not a multiple of 8
    size_t bytes = (size_t)n * 4;
    std::vector<float> ha(n), hb(n), hasc(n), hacl(n);
    srand(1234);
    for (uint32_t i = 0; i < n; i++) {
        int m = i % 16;
        float v;
        if (m == 0) v = 0.0f; else if (m == 1) v = -0.0f;
        else if (m == 2) v = 1e-38f; else if (m == 3) v = 1e38f;
        else v = (float)((rand() / (double)RAND_MAX) * 8.0 - 4.0);
        ha[i] = v;
        float w = (float)((rand() / (double)RAND_MAX) * 8.0 - 4.0);
        if (i % 23 == 0) w = 0.0f;
        hb[i] = w;
    }
    void *da, *db, *dc, *wsp;
    CHECK(aclrtMalloc(&da, bytes + 4096, ACL_MEM_MALLOC_HUGE_FIRST), "m");
    CHECK(aclrtMalloc(&db, bytes + 4096, ACL_MEM_MALLOC_HUGE_FIRST), "m");
    CHECK(aclrtMalloc(&dc, bytes + 4096, ACL_MEM_MALLOC_HUGE_FIRST), "m");
    CHECK(aclrtMalloc(&wsp, 64u << 20, ACL_MEM_MALLOC_HUGE_FIRST), "m");
    CHECK(aclrtMemcpy(da, bytes, ha.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "c");
    CHECK(aclrtMemcpy(db, bytes, hb.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "c");

    int64_t shape[1] = {(int64_t)n}; int64_t stride[1] = {1};
    auto mk = [&](void *p) { return aclCreateTensor(shape, 1, ACL_FLOAT, stride, 0, ACL_FORMAT_ND, shape, 1, p); };
    float one = 1.0f; aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);

    struct Case { const char *name; void (*k)(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t); int unary; };
    auto run_asc = [&](int which) {
        switch (which) {
            case 0: k_add <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 1: k_sub <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 2: k_mul <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 3: k_div <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 4: k_max <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 5: k_min <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 6: k_abs <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 7: k_neg <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 8: k_exp <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 9: k_log <<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
            case 10: k_sqrt<<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n); break;
        }
    };
    auto run_acl = [&](int which) {
        aclTensor *ta = mk(da), *tb = mk(db), *tc = mk(dc);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        switch (which) {
            case 0: aclnnAddGetWorkspaceSize(ta,tb,alpha,tc,&ws,&ex); aclnnAdd(ws?wsp:nullptr,ws,ex,st); break;
            case 1: aclnnSubGetWorkspaceSize(ta,tb,alpha,tc,&ws,&ex); aclnnSub(ws?wsp:nullptr,ws,ex,st); break;
            case 2: aclnnMulGetWorkspaceSize(ta,tb,tc,&ws,&ex); aclnnMul(ws?wsp:nullptr,ws,ex,st); break;
            case 3: aclnnDivGetWorkspaceSize(ta,tb,tc,&ws,&ex); aclnnDiv(ws?wsp:nullptr,ws,ex,st); break;
            case 4: aclnnMaximumGetWorkspaceSize(ta,tb,tc,&ws,&ex); aclnnMaximum(ws?wsp:nullptr,ws,ex,st); break;
            case 5: aclnnMinimumGetWorkspaceSize(ta,tb,tc,&ws,&ex); aclnnMinimum(ws?wsp:nullptr,ws,ex,st); break;
            case 6: aclnnAbsGetWorkspaceSize(ta,tc,&ws,&ex); aclnnAbs(ws?wsp:nullptr,ws,ex,st); break;
            case 7: aclnnNegGetWorkspaceSize(ta,tc,&ws,&ex); aclnnNeg(ws?wsp:nullptr,ws,ex,st); break;
            default: return false;
        }
        aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
        return true;
    };
    const char *names[] = {"add","sub","mul","div","maximum","minimum","abs","neg","exp","log","sqrt"};
    std::printf("%-9s %12s %12s %14s   (n=%u, tail=%u)\n", "op", "bitdiff", "maxulp", "maxrel", n, n % 8);
    for (int w = 0; w < 11; w++) {
        CHECK(aclrtMemset(dc, bytes, 0xCC, bytes), "memset");
        run_asc(w); CHECK(aclrtSynchronizeStream(st), "sync");
        CHECK(aclrtMemcpy(hasc.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "cp");
        CHECK(aclrtMemset(dc, bytes, 0xCC, bytes), "memset");
        if (!run_acl(w)) { std::printf("%-9s %12s (no aclnn reference in this probe)\n", names[w], "-"); continue; }
        CHECK(aclrtSynchronizeStream(st), "sync");
        CHECK(aclrtMemcpy(hacl.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "cp");
        long bitdiff = 0, maxulp = 0; double maxrel = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t x, y; std::memcpy(&x, &hasc[i], 4); std::memcpy(&y, &hacl[i], 4);
            if (x != y) {
                bitdiff++;
                if (std::isnan(hasc[i]) && std::isnan(hacl[i])) continue;
                long u = (long)x - (long)y; if (u < 0) u = -u;
                if (u > maxulp) maxulp = u;
                double d = std::fabs((double)hasc[i] - (double)hacl[i]);
                double r = std::fabs((double)hacl[i]) > 1e-30 ? d / std::fabs((double)hacl[i]) : d;
                if (r > maxrel) maxrel = r;
            }
        }
        std::printf("%-9s %12ld %12ld %14.3e\n", names[w], bitdiff, maxulp, maxrel);
    }

    // tail correctness against the host for the fused 3-op chain shape
    std::printf("\ntail check: last 16 elements of add, n=%u\n", n);
    CHECK(aclrtMemset(dc, bytes + 4096, 0xCC, bytes + 4096), "memset");
    k_add<<<48,nullptr,st>>>((GM_ADDR)da,(GM_ADDR)db,(GM_ADDR)dc,n);
    CHECK(aclrtSynchronizeStream(st), "sync");
    std::vector<float> tailbuf(1024);
    CHECK(aclrtMemcpy(tailbuf.data(), 1024*4, (char*)dc + bytes, 1024*4, ACL_MEMCPY_DEVICE_TO_HOST), "cp");
    int overwritten = 0;
    for (int i = 0; i < 1024; i++) { uint32_t u; std::memcpy(&u, &tailbuf[i], 4); if (u != 0xCCCCCCCCu) overwritten++; }
    std::printf("  bytes written past the %u-element output: %d of 1024 words\n", n, overwritten);
    return 0;
}
