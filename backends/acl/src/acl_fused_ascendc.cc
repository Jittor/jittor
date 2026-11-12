// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// One generated AscendC kernel per eligible floating-point elementwise group.
//
// The generic ACL path (exec_acl_sequence) issues one aclnn launch per node of
// a fused group: a five-node activation costs five launches, each carrying a
// tensor-descriptor build, a workspace query and a kernel launch. This file is
// the ACL analogue of what the CUDA backend does with nvcc: it turns the whole
// group into a single device kernel, compiles it with ccec, and launches it
// once.
//
// It is deliberately narrow. A group is eligible only when every fact needed
// for a bit-exact result is checked, never assumed:
//   * vars use supported float32/float16/bfloat16 combinations, device storage
//     and checked dense or broadcast layouts with compatible element counts;
//   * every node is a unary/binary/copy whose AscendC intrinsic was measured
//     bit-identical to the aclnn op the generic path would have run
//     (add/sub/mul/max/min/abs/neg are exact; div is within 1 ulp);
//   * a node output is written to global memory exactly when the executor
//     allocated it, and lives only in unified buffer otherwise.
// Anything else returns false and the caller runs its normal per-node path.
// Nothing here ever selects a different backend.
#include <acl/acl.h>
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common.h"
#include "core/op.h"
#include "core/var.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/composite/array_op.h"
#include "ops/binary_op.h"
#include "mem/allocator.h"
#include "utils/cache_compile.h"
#include "acl_jittor.h"
#include "acl_fused_ascendc.h"

namespace jittor
{
    DECLARE_FLAG(string, cache_path);

    // Off by default: a generated kernel that is wrong is a silent wrong-result
    // bug across the whole framework, so the per-node path stays the default
    // until this one has been through a full acceptance run.
    DEFINE_FLAG(int, use_acl_ascendc_fusion, 0,
        "Run eligible floating-point elementwise fused groups as one generated AscendC kernel");
    DEFINE_FLAG(int, acl_ascendc_min_nodes, 1,
        "Smallest fused group the AscendC path takes over (1 also replaces single-node groups)");

    namespace
    {
        constexpr int kMaxNodes = 16;
        constexpr int kMaxArgs = 12;      // pointer parameters of one kernel
        constexpr int kUbBudget = 160 * 1024;
        constexpr int kMaxTile = 8192;
        constexpr int kMinTile = 256;
        constexpr int kElemsPerBlock = 2048;

        typedef void (*LaunchFn)(void **, uint32_t, uint32_t, void *);

        int64 fused_count = 0;
        int64 fused_node_count = 0;

        // Why a group was not taken, counted per reason. Only used for
        // coverage reports; costs one pointer compare on the reject path.
        std::mutex reject_mutex;
        std::unordered_map<string, int64> reject_counts;
        bool reject_trace = getenv("acl_ascendc_report") != nullptr;

        bool reject(const char *why)
        {
            if (reject_trace)
            {
                std::lock_guard<std::mutex> lock(reject_mutex);
                int64 &n = reject_counts[why];
                n++;
                // First hits of each reason on stderr: the counter alone is
                // unreachable from python, and "why was this chain not fused"
                // is the first question a coverage run asks.
                if (n <= 3)
                    std::fprintf(stderr, "ACL_ASCENDC_REJECT %s\n", why);
            }
            return false;
        }

        // ---- node model -------------------------------------------------
        enum NodeKind { kCopy = 0, kUnary = 1, kBinary = 2, kConst = 3 };

        struct Node
        {
            NodeKind kind;
            NanoString ns;      // the operation this node performs
            int op_idx;         // index into FusedOp::ops
            int nin;
            int src[2];         // >=0 value index, <0 -(group input index + 1)
            int out_mat;        // materialized output index, or -1
            NanoString dtype;   // dtype of this node's output in the graph
            double cval = 0.0;  // value of a kConst node (one-element array)
            bool round;         // the graph rounds here: re-round through the
                                // 2-byte scratch after the float computation
        };

        // Where a var lives inside the group, so the plan can find it again in
        // a later execution without re-running the analysis.
        struct VarSlot
        {
            int node;           // index into Plan::nodes
            int port;           // input port, or -1 for the node output
        };

        struct Plan
        {
            bool ok = false;
            vector<Node> nodes;
            vector<VarSlot> inputs;
            // 0: dense operand read at the element the tile is on.
            // P>0: broadcast operand whose value at flat index i is base[i % P]
            //      -- the only strided views this path accepts.
            vector<int> input_period;
            vector<VarSlot> outputs;
            vector<NanoString> input_dtype;
            vector<NanoString> output_dtype;
            int align = 8;
            bool widen = false;            // compute in float32 for half/bfloat16
            NanoString wide_dtype = ns_float32;  // 2-byte type used for rounding
            bool scratch = false;          // at least one node rounds
            LaunchFn launch = nullptr;
        };

        Var *locate(FusedOp *fop, const vector<Node> &nodes, const VarSlot &s)
        {
            Op *op = fop->ops[nodes[s.node].op_idx];
            return s.port < 0 ? op->output(0) : op->input(s.port);
        }

        // ---- what the AscendC vector API can reproduce exactly -----------
        // Every entry was measured against the aclnn op the generic path runs
        // (benchmarks/acl/probes/asc_numerics.cc): add/sub/mul/maximum/minimum/abs/neg
        // are bit-identical over 1e6 mixed values including signed zeros and
        // denormals; div differs by at most 1 ulp.
        const char *binary_intrinsic(NanoString ns)
        {
            if (ns == ns_add) return "Add";
            if (ns == ns_subtract) return "Sub";
            if (ns == ns_multiply) return "Mul";
            if (ns == ns_divide) return "Div";
            if (ns == ns_maximum) return "Max";
            if (ns == ns_minimum) return "Min";
            return nullptr;
        }

        bool unary_supported(NanoString ns)
        {
            return ns == ns_abs || ns == ns_negative || ns == ns_cast;
        }

        // The dtype the generated kernel works in.  float32 was the only
        // admitted type; the H3 DiT runs bfloat16 and its chains were left to
        // the per-node path -- measured 17-20 ms against 0.31 ms of torch_npu
        // for a three-node 3600x4096 bf16 chain.
        bool supported_dtype(NanoString ns, int &esize, string &ctype)
        {
            if (ns == ns_float32) { esize = 4; ctype = "float"; return true; }
            if (ns == ns_float16) { esize = 2; ctype = "half"; return true; }
            if (ns == ns_bfloat16) { esize = 2; ctype = "bfloat16_t"; return true; }
            return false;
        }

        // Elements in one 32-byte vector lane.
        int vec_lane(NanoString ns) { return ns == ns_float32 ? 8 : 16; }

        bool is_2byte(NanoString ns) { return ns == ns_float16 || ns == ns_bfloat16; }
        int dtype_size(NanoString ns) { return ns == ns_float32 ? 4 : 2; }
        const char *ctype_of(NanoString ns)
        {
            if (ns == ns_float32) return "float";
            if (ns == ns_float16) return "half";
            return "bfloat16_t";
        }

        bool var_is_plain(Var *v, NanoString dtype)
        {
            return v && v->dtype() == dtype && v->is_contiguous()
                && v->storage_offset_bytes == 0;
        }

        // A pure broadcast view: some leading axes repeat (stride 0) and the
        // trailing axes are densely packed. Returns the period P, so the value
        // at flat index i is base[i % P]; 0 when the view is anything else.
        // A transposed or otherwise permuted view is not accepted here.
        int broadcast_period(Var *v, int vec = 8)
        {
            const int nd = (int)v->shape.size();
            if (nd == 0 || (int)v->storage_strides.size() != nd) return 0;
            int64 e = 1;
            int d = nd - 1;
            for (; d >= 0; d--)
            {
                if (v->shape[d] == 1) { continue; }
                if (v->storage_strides[d] != e) break;
                e *= v->shape[d];
            }
            for (int k = d; k >= 0; k--)
                if (v->shape[k] != 1 && v->storage_strides[k] != 0) return 0;
            if (e <= 0 || e > (int64)0x40000000) return 0;
            // The buffer must really hold those P elements and nothing of the
            // logical shape beyond them.
            if (v->storage_span_bytes() != e * v->dtype().dsize()) return 0;
            if (e == v->num) return 0;               // dense, not a broadcast
            if (e != 1 && (e % vec) != 0) return 0;  // 32-byte copies only
            return (int)e;
        }

        bool var_is_plain_float32(Var *v)
        {
            return v && v->dtype() == ns_float32 && v->is_contiguous()
                && v->storage_offset_bytes == 0;
        }

        bool var_on_device(Var *v)
        {
            return v->mem_ptr && v->allocator && v->allocator->is_cuda()
                && (reinterpret_cast<uintptr_t>(v->mem_ptr) % 32) == 0;
        }

        // The value of a one-element array op, in the dtype it was stored.
        double array_scalar_value(ArrayOp *aop)
        {
            if (aop->has_python_float) return aop->python_float;
            Var *out = aop->output;
            if (out->dtype() == ns_float32)
                return (double)*(const float *)aop->ptr<float>();
            if (out->dtype() == ns_float16 || out->dtype() == ns_bfloat16)
            {
                uint16_t bits = *(const uint16_t *)aop->ptr<uint16_t>();
                uint32_t wide;
                if (out->dtype() == ns_bfloat16)
                    wide = (uint32_t)bits << 16;
                else
                {
                    uint32_t s = (bits >> 15) & 1u, e = (bits >> 10) & 0x1fu, m = bits & 0x3ffu;
                    if (e == 0)
                    {
                        if (m == 0) wide = s << 31;
                        else
                        {
                            e = 1;
                            while ((m & 0x400u) == 0) { m <<= 1; e--; }
                            m &= 0x3ffu;
                            wide = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
                        }
                    }
                    else if (e == 31) wide = (s << 31) | 0x7f800000u | (m << 13);
                    else wide = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
                }
                float f;
                std::memcpy(&f, &wide, sizeof(f));
                return (double)f;
            }
            return 0.0;
        }

        NodeKind classify(Op *op)
        {
            const char *n = op->name();
            if (std::strcmp(n, "binary") == 0) return kBinary;
            if (std::strcmp(n, "unary") == 0) return kUnary;
            if (std::strcmp(n, "array") == 0) return kConst;
            return kCopy;   // only reached for "contiguous", checked by caller
        }

        // ---- source generation -------------------------------------------
        int64 gcd64(int64 a, int64 b) { while (b) { int64 t = a % b; a = b; b = t; } return a; }

        string generate_source(const Plan &plan, int &tile_out)
        {
            const vector<Node> &nodes = plan.nodes;
            const vector<int> &period = plan.input_period;
            const vector<NanoString> &idtype = plan.input_dtype;
            const vector<NanoString> &odtype = plan.output_dtype;
            const bool widen = plan.widen;
            const int nn = (int)nodes.size();
            const int n_in = (int)period.size();
            const int n_out = (int)odtype.size();

            // Temporary-slot allocation: a value that is not written to global
            // memory lives in unified buffer until its last consumer.
            vector<int> last_use(nn, -1);
            for (int i = 0; i < nn; i++)
                for (int j = 0; j < nodes[i].nin; j++)
                    if (nodes[i].src[j] >= 0) last_use[nodes[i].src[j]] = i;

            vector<int> slot_of(nn, -1);
            int n_temp = 0;
            {
                vector<int> pool;
                for (int i = 0; i < nn; i++)
                {
                    if (nodes[i].out_mat < 0)
                    {
                        if (pool.empty()) { slot_of[i] = n_temp++; }
                        else { slot_of[i] = pool.back(); pool.pop_back(); }
                    }
                    for (int j = 0; j < nodes[i].nin; j++)
                    {
                        int v = nodes[i].src[j];
                        if (v >= 0 && last_use[v] == i && slot_of[v] >= 0)
                            pool.push_back(slot_of[v]);
                    }
                }
            }

            // Unified buffer accounting in bytes per element.
            int64 bytes = 0;
            int64 fixed_bytes = 0;
            for (int i = 0; i < n_in; i++)
            {
                int d = dtype_size(idtype[i]);
                if (!period[i]) bytes += 2 * d + ((widen && d < 4) ? 4 : 0);
                else {
                    // Every broadcast owns a tile-sized bp (and optionally
                    // bx). A scalar also needs one 32-byte load buffer bs.
                    bytes += d + ((widen && d < 4) ? 4 : 0);
                    if (period[i] == 1) fixed_bytes += 32;
                }
            }
            for (int i = 0; i < n_out; i++)
            {
                int d = dtype_size(odtype[i]);
                bytes += 2 * d + ((d < 4) ? 4 : 0);
            }
            bytes += (int64)n_temp * 4;
            if (plan.scratch) bytes += 2;   // the 2-byte round-trip buffer
            if (bytes <= 0) return string();
            int tile = (int)((kUbBudget - fixed_bytes) / bytes);
            tile = tile / 64 * 64;
            if (tile > kMaxTile) tile = kMaxTile;
            tile = tile / plan.align * plan.align;
            if (tile < plan.align || tile < kMinTile) return string();
            tile_out = tile;

            auto value_name = [&](int v) -> string
            {
                if (nodes[v].out_mat >= 0) return "y" + S(nodes[v].out_mat);
                return "t" + S(slot_of[v]);
            };
            auto operand = [&](const Node &nd, int j) -> string
            {
                if (nd.src[j] >= 0) return value_name(nd.src[j]);
                int g = -nd.src[j] - 1;
                if (!period[g]) return "x" + S(g);
                return (widen ? "pf" : "p") + S(g);
            };

            const char *WT = ctype_of(plan.wide_dtype);

            std::ostringstream s;
            s << "// generated by jittor backends/acl/src/acl_fused_ascendc.cc\n"
              << "#include \"kernel_operator.h\"\n#include <acl/acl.h>\nusing namespace AscendC;\n\n"
              << "constexpr int32_t JT_TILE = " << tile << ";\n"
              << "constexpr uint32_t JT_ALIGN = " << plan.align << ";\n\n"
              << "extern \"C\" __global__ __aicore__ void jt_asc_kernel(\n    ";
            for (int i = 0; i < n_in; i++) s << "GM_ADDR gi" << i << ", ";
            for (int i = 0; i < n_out; i++) s << "GM_ADDR go" << i << ", ";
            s << "uint32_t n)\n{\n"
              << "    uint32_t blocks = GetBlockNum();\n"
              << "    uint32_t bid = GetBlockIdx();\n"
              << "    uint32_t chunk = ((n + blocks - 1) / blocks + JT_ALIGN - 1) / JT_ALIGN * JT_ALIGN;\n"
              << "    uint32_t start = bid * chunk;\n"
              << "    if (start >= n) return;\n"
              << "    uint32_t len = (n - start) < chunk ? (n - start) : chunk;\n\n"
              << "    TPipe pipe;\n";
            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "    TQue<QuePosition::VECIN, 2> qi" << i << ";\n";
            for (int i = 0; i < n_out; i++)
                s << "    TQue<QuePosition::VECOUT, 2> qo" << i << ";\n";
            for (int i = 0; i < n_in; i++)
                if (period[i]) s << "    TBuf<QuePosition::VECCALC> bp" << i << ";\n";
            for (int i = 0; i < n_in; i++)
                if (period[i] == 1) s << "    TBuf<QuePosition::VECCALC> bs" << i << ";\n";
            for (int i = 0; i < n_temp; i++)
                s << "    TBuf<QuePosition::VECCALC> bt" << i << ";\n";
            for (int i = 0; i < n_in; i++)
                if (widen && dtype_size(idtype[i]) < 4) s << "    TBuf<QuePosition::VECCALC> bx" << i << ";\n";
            for (int i = 0; i < n_out; i++)
                if (dtype_size(odtype[i]) < 4) s << "    TBuf<QuePosition::VECCALC> by" << i << ";\n";
            if (plan.scratch) s << "    TBuf<QuePosition::VECCALC> bsc;\n";
            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "    pipe.InitBuffer(qi" << i << ", 2, JT_TILE * " << dtype_size(idtype[i]) << ");\n";
            for (int i = 0; i < n_out; i++)
                s << "    pipe.InitBuffer(qo" << i << ", 2, JT_TILE * " << dtype_size(odtype[i]) << ");\n";
            for (int i = 0; i < n_in; i++)
                if (period[i]) s << "    pipe.InitBuffer(bp" << i << ", JT_TILE * " << dtype_size(idtype[i]) << ");\n";
            for (int i = 0; i < n_in; i++)
                if (period[i] == 1) s << "    pipe.InitBuffer(bs" << i << ", 32);\n";
            for (int i = 0; i < n_temp; i++)
                s << "    pipe.InitBuffer(bt" << i << ", JT_TILE * 4);\n";
            for (int i = 0; i < n_in; i++)
                if (widen && dtype_size(idtype[i]) < 4) s << "    pipe.InitBuffer(bx" << i << ", JT_TILE * 4);\n";
            for (int i = 0; i < n_out; i++)
                if (dtype_size(odtype[i]) < 4) s << "    pipe.InitBuffer(by" << i << ", JT_TILE * 4);\n";
            if (plan.scratch) s << "    pipe.InitBuffer(bsc, JT_TILE * 2);\n";
            s << "\n";
            for (int i = 0; i < n_in; i++)
            {
                s << "    GlobalTensor<" << ctype_of(idtype[i]) << "> mi" << i << ";\n";
                if (period[i])
                    s << "    mi" << i << ".SetGlobalBuffer((__gm__ " << ctype_of(idtype[i])
                      << "*)gi" << i << ", " << period[i] << ");\n";
                else
                    s << "    mi" << i << ".SetGlobalBuffer((__gm__ " << ctype_of(idtype[i])
                      << "*)gi" << i << " + start, len);\n";
            }
            for (int i = 0; i < n_out; i++)
                s << "    GlobalTensor<" << ctype_of(odtype[i]) << "> mo" << i << ";\n"
                  << "    mo" << i << ".SetGlobalBuffer((__gm__ " << ctype_of(odtype[i])
                  << "*)go" << i << " + start, len);\n";

            // Materialise every broadcast operand once, tiled to JT_TILE.
            bool any_bcast = false;
            for (int i = 0; i < n_in; i++) if (period[i]) any_bcast = true;
            if (any_bcast) s << "\n";
            for (int i = 0; i < n_in; i++)
            {
                if (!period[i]) continue;
                const char *CT = ctype_of(idtype[i]);
                if (period[i] == 1)
                {
                    s << "    LocalTensor<" << CT << "> p" << i << " = bp" << i << ".Get<" << CT << ">();\n"
                      << "    {\n"
                      << "        LocalTensor<" << CT << "> sv = bs" << i << ".Get<" << CT << ">();\n"
                      << "        DataCopyExtParams sep{1, (uint32_t)" << dtype_size(idtype[i]) << ", 0, 0, 0};\n"
                      << "        DataCopyPadExtParams<" << CT << "> spp{false, 0, 0, 0};\n"
                      << "        DataCopyPad(sv, mi" << i << ", sep, spp);\n"
                      << "        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                      << "        SetFlag<HardEvent::MTE2_S>(ev);\n"
                      << "        WaitFlag<HardEvent::MTE2_S>(ev);\n"
                      << "        Duplicate(p" << i << ", sv.GetValue(0), JT_TILE);\n"
                      << "    }\n";
                }
                else
                {
                    s << "    LocalTensor<" << CT << "> p" << i << " = bp" << i << ".Get<" << CT << ">();\n"
                      << "    for (int32_t r = 0; r < JT_TILE / " << period[i] << "; r++)\n"
                      << "        DataCopy(p" << i << "[r * " << period[i] << "], mi" << i
                      << ", " << period[i] << ");\n"
                      // A plain TBuf has no queue events, and --cce-auto-sync
                      // does not order this MTE2 fill before the vector reads
                      // of the same buffer: without the barrier the first
                      // tile used uninitialized UB (NaN results on 910B3).
                      << "    {\n"
                      << "        event_t evb" << i << " = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));\n"
                      << "        SetFlag<HardEvent::MTE2_V>(evb" << i << ");\n"
                      << "        WaitFlag<HardEvent::MTE2_V>(evb" << i << ");\n"
                      << "    }\n";
                }
                if (widen && dtype_size(idtype[i]) < 4)
                    s << "    LocalTensor<float> pf" << i << " = bx" << i << ".Get<float>();\n"
                      << "    Cast(pf" << i << ", p" << i << ", RoundMode::CAST_NONE, JT_TILE);\n";
                else if (widen)
                    s << "    LocalTensor<float> pf" << i << " = p" << i << ";\n";
            }

            s << "\n    for (uint32_t off = 0; off < len; off += JT_TILE) {\n"
              << "        uint32_t cur = (len - off) < (uint32_t)JT_TILE ? (len - off) : (uint32_t)JT_TILE;\n";
            for (int i = 0; i < n_in; i++)
            {
                if (period[i]) continue;
                const char *CT = ctype_of(idtype[i]);
                int d = dtype_size(idtype[i]);
                int lane = vec_lane(idtype[i]);
                s << "        LocalTensor<" << CT << "> xr" << i << " = qi" << i << ".AllocTensor<" << CT << ">();\n"
                  << "        if (cur % " << lane << " == 0) DataCopy(xr" << i << ", mi" << i << "[off], cur);\n"
                  << "        else {\n"
                  << "            DataCopyExtParams ep{1, cur * " << d << ", 0, 0, 0};\n"
                  << "            DataCopyPadExtParams<" << CT << "> pp{false, 0, 0, 0};\n"
                  << "            DataCopyPad(xr" << i << ", mi" << i << "[off], ep, pp);\n"
                  << "        }\n"
                  << "        qi" << i << ".EnQue(xr" << i << ");\n";
            }
            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "        xr" << i << " = qi" << i << ".DeQue<" << ctype_of(idtype[i]) << ">();\n";
            for (int i = 0; i < n_in; i++)
            {
                if (period[i]) continue;
                if (widen && dtype_size(idtype[i]) < 4)
                    s << "        LocalTensor<float> x" << i << " = bx" << i << ".Get<float>();\n"
                      << "        Cast(x" << i << ", xr" << i << ", RoundMode::CAST_NONE, cur);\n";
                else
                    s << "        LocalTensor<float> x" << i << " = xr" << i << ";\n";
            }
            for (int i = 0; i < n_out; i++)
            {
                const char *CT = ctype_of(odtype[i]);
                s << "        LocalTensor<" << CT << "> yr" << i << " = qo" << i << ".AllocTensor<" << CT << ">();\n";
                if (dtype_size(odtype[i]) < 4)
                    s << "        LocalTensor<float> y" << i << " = by" << i << ".Get<float>();\n";
                else
                    s << "        LocalTensor<float> y" << i << " = yr" << i << ";\n";
            }
            for (int i = 0; i < n_temp; i++)
                s << "        LocalTensor<float> t" << i << " = bt" << i << ".Get<float>();\n";
            if (plan.scratch)
                s << "        LocalTensor<" << WT << "> scr = bsc.Get<" << WT << ">();\n";

            auto emit_round = [&](const string &dst)
            {
                s << "        Cast(scr, " << dst << ", RoundMode::CAST_RINT, cur);\n"
                  << "        Cast(" << dst << ", scr, RoundMode::CAST_NONE, cur);\n";
            };

            for (int i = 0; i < nn; i++)
            {
                const Node &nd = nodes[i];
                string dst = value_name(i);
                s << "        ";
                if (nd.kind == kConst)
                {
                    char cbuf[48];
                    std::snprintf(cbuf, sizeof(cbuf), "%.17g", nd.cval);
                    s << "Duplicate(" << dst << ", (float)(" << cbuf << "), cur);\n";
                    continue;
                }
                if (nd.kind == kBinary)
                {
                    s << binary_intrinsic(nd.ns) << "(" << dst << ", "
                      << operand(nd, 0) << ", " << operand(nd, 1) << ", cur);\n";
                }
                else if (nd.kind == kUnary && nd.ns == ns_abs)
                {
                    s << "Abs(" << dst << ", " << operand(nd, 0) << ", cur);\n";
                }
                else if (nd.kind == kUnary && nd.ns == ns_negative)
                {
                    s << "Muls(" << dst << ", " << operand(nd, 0) << ", -1.0f, cur);\n";
                }
                else
                {
                    // Copy / same-space cast. Multiplying by one is exact for
                    // float32 and for every half/bf16 value the vector unit
                    // passes through.
                    s << "Muls(" << dst << ", " << operand(nd, 0) << ", 1.0f, cur);\n";
                }
                if (nd.round && plan.scratch) emit_round(dst);
            }

            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "        qi" << i << ".FreeTensor(xr" << i << ");\n";
            for (int i = 0; i < n_out; i++)
            {
                const char *CT = ctype_of(odtype[i]);
                int lane = vec_lane(odtype[i]);
                int d = dtype_size(odtype[i]);
                if (d < 4)
                    s << "        Cast(yr" << i << ", y" << i << ", RoundMode::CAST_RINT, cur);\n";
                s << "        qo" << i << ".EnQue(yr" << i << ");\n"
                  << "        yr" << i << " = qo" << i << ".DeQue<" << CT << ">();\n"
                  << "        if (cur % " << lane << " == 0) DataCopy(mo" << i << "[off], yr" << i << ", cur);\n"
                  << "        else {\n"
                  << "            DataCopyExtParams ep{1, cur * " << d << ", 0, 0, 0};\n"
                  << "            DataCopyPad(mo" << i << "[off], yr" << i << ", ep);\n"
                  << "        }\n"
                  << "        qo" << i << ".FreeTensor(yr" << i << ");\n";
            }
            s << "    }\n}\n\n";

            s << "extern \"C\" void jt_asc_launch(void** p, uint32_t n, uint32_t bd, void* stream)\n{\n"
              << "    jt_asc_kernel<<<bd, nullptr, (aclrtStream)stream>>>(\n        ";
            for (int i = 0; i < n_in + n_out; i++) s << "(GM_ADDR)p[" << i << "], ";
            s << "n);\n}\n";
            return s.str();
        }

        // ---- ccec toolchain ----------------------------------------------
        struct Toolchain
        {
            bool ready = false;
            bool probed = false;
            string ccec;
            string flags;
            string link;
        };

        bool dir_exists(const string &p)
        {
            return access(p.c_str(), X_OK) == 0;
        }

        string ascend_home()
        {
            const char *e = getenv("ASCEND_TOOLKIT_HOME");
            if (e && *e && dir_exists(e)) return e;
            e = getenv("ASCEND_HOME_PATH");
            if (e && *e && dir_exists(e)) return e;
            return string();
        }

        // clang picks the newest /usr/lib/gcc/<triple>/<n> directory, which on
        // this image holds no C++ headers at all; name the real ones instead.
        // The directory is not always the integer major version: openEuler's
        // GCC 10.3 installs its headers under "/usr/include/c++/10.3.1", and
        // the old integer scan found nothing there -- which silently disabled
        // AscendC fusion for every group on the machine.
        string host_cxx_includes()
        {
            static const char *triples[] = {
                "x86_64-linux-gnu", "aarch64-linux-gnu", nullptr};
            const string root = "/usr/include/c++";
            string chosen;
            double chosen_v = -1;
            if (DIR *d = opendir(root.c_str()))
            {
                while (struct dirent *e = readdir(d))
                {
                    string name = e->d_name;
                    if (name == "." || name == "..") continue;
                    if (access((root + "/" + name + "/type_traits").c_str(), R_OK) != 0) continue;
                    int major = 0, minor = 0;
                    if (std::sscanf(name.c_str(), "%d.%d", &major, &minor) != 2)
                        minor = 0;
                    double v = major + minor / 1000.0;
                    if (v > chosen_v) { chosen_v = v; chosen = name; }
                }
                closedir(d);
            }
            if (chosen.empty()) return string();
            string best = root + "/" + chosen;
            string out = " -I" + best + " -I" + best + "/backward";
            for (int i = 0; triples[i]; i++)
            {
                string t = "/usr/include/" + string(triples[i]) + "/c++/" + chosen;
                if (access((t + "/bits/c++config.h").c_str(), R_OK) == 0)
                    out += " -I" + t;
            }
            return out;
        }

        // dav-c310-vec for Ascend950, dav-c220-vec for 910B, ... read from the
        // installed platform config rather than guessed from the chip name.
        string ccec_arch(const string &home, const string &devkit)
        {
            const char *soc = aclrtGetSocName();
            if (!soc || !*soc) return string();
            string ini = devkit + "/data/platform_config/" + soc + ".ini";
            string text;
            try { text = jit_compiler::read_all(ini); }
            catch (...) { return string(); }
            const string key = "CCEC_AIV_version=";
            auto pos = text.find(key);
            if (pos == string::npos)
            {
                const string key2 = "CCEC_VECTOR_version=";
                pos = text.find(key2);
                if (pos == string::npos) return string();
                pos += key2.size();
            }
            else pos += key.size();
            auto end = text.find_first_of("\r\n", pos);
            return text.substr(pos, end == string::npos ? end : end - pos);
        }

        // How many vector cores to spread a kernel over. Measured on
        // Ascend950PR (benchmarks/acl/probes/asc_fuse_bench.cc): using all 56 rather
        // than 48 is worth ~10% at 4M elements and neutral below that; using
        // half of them costs ~80%.
        int vector_cores()
        {
            static int cached = 0;
            if (cached) return cached;
            cached = 48;
            string home = ascend_home();
            const char *soc = aclrtGetSocName();
            if (!home.empty() && soc && *soc)
            {
                string devkit = home + "/x86_64-linux";
                if (!dir_exists(devkit)) devkit = home + "/aarch64-linux";
                string text;
                try { text = jit_compiler::read_all(devkit + "/data/platform_config/" + soc + ".ini"); }
                catch (...) { text.clear(); }
                auto pos = text.find("vector_core_cnt=");
                if (pos != string::npos)
                {
                    int v = atoi(text.c_str() + pos + 16);
                    if (v >= 1 && v <= 256) cached = v;
                }
            }
            return cached;
        }

        Toolchain &toolchain()
        {
            static Toolchain tc;
            if (tc.probed) return tc;
            tc.probed = true;
            string home = ascend_home();
            if (home.empty())
            {
                LOGw << "AscendC fusion disabled: no ASCEND_TOOLKIT_HOME/ASCEND_HOME_PATH";
                return tc;
            }
            string devkit = home + "/x86_64-linux";
            if (!dir_exists(devkit)) devkit = home + "/aarch64-linux";
            if (!dir_exists(devkit))
            {
                LOGw << "AscendC fusion disabled: no <toolkit>/<arch>-linux devkit under" << home;
                return tc;
            }
            string ccec = home + "/bin/ccec";
            if (access(ccec.c_str(), X_OK) != 0)
            {
                LOGw << "AscendC fusion disabled: no ccec at" << ccec;
                return tc;
            }
            string arch = ccec_arch(home, devkit);
            if (arch.empty())
            {
                LOGw << "AscendC fusion disabled: no CCEC vector arch for" << aclrtGetSocName();
                return tc;
            }
            string cxx = host_cxx_includes();
            if (cxx.empty())
            {
                LOGw << "AscendC fusion disabled: no libstdc++ headers for ccec";
                return tc;
            }
            string inc;
            const char *dirs[] = {
                "/asc/impl/adv_api", "/asc/impl/basic_api", "/asc/impl/c_api",
                "/asc/impl/basic_api/reg_compute", "/asc/impl/simt_api", "/asc/impl/utils",
                "/asc", "/asc/include", "/asc/include/adv_api", "/asc/include/basic_api",
                "/asc/include/aicpu_api", "/asc/include/c_api",
                "/asc/include/basic_api/reg_compute", "/asc/include/simt_api",
                "/asc/include/utils", "/tikcpp/tikcfw", "/tikcpp/tikcfw/interface",
                "/tikcpp/tikcfw/impl", nullptr};
            for (int i = 0; dirs[i]; i++) inc += " -I" + devkit + dirs[i];
            inc += " -I" + home + "/include";

            string arch_define;
            if (arch.compare(0, 8, "dav-c310") == 0) arch_define = " -D__DAV_C310__";

            tc.ccec = ccec;
            tc.flags = " -O2 -std=c++17 --cce-aicore-lang" + arch_define
                + " --cce-aicore-arch=" + arch
                + " --cce-auto-sync --cce-mask-opt -w -DTILING_KEY_VAR=0 -fPIC -shared"
                + cxx + inc;
            tc.link = " -L" + home + "/lib64 -lascendcl -lruntime";
            tc.ready = true;
            LOGv << "AscendC fusion toolchain:" << ccec << "arch" << arch;
            return tc;
        }

        // ---- compiled kernel cache ---------------------------------------
        struct KernelLib
        {
            void *handle = nullptr;
            LaunchFn launch = nullptr;
        };

        std::mutex lib_mutex;
        std::unordered_map<string, KernelLib> lib_cache;

        uint64 fnv1a(const string &s)
        {
            uint64 h = 1469598103934665603ull;
            for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
            return h;
        }

        string hex64(uint64 v)
        {
            char buf[20];
            std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
            return buf;
        }

        LaunchFn build_or_get(const string &src)
        {
            std::lock_guard<std::mutex> lock(lib_mutex);
            auto found = lib_cache.find(src);
            if (found != lib_cache.end()) return found->second.launch;

            Toolchain &tc = toolchain();
            if (!tc.ready) { lib_cache[src] = KernelLib(); return nullptr; }

            string tag = hex64(fnv1a(src));
            string base = cache_path + "/jit/acl_asc_" + tag;
            string cc = base + ".cc";
            string so = base + ".so";

            // Reuse a kernel an earlier process built, but only after proving
            // the cached source is the source we asked for: the file name is a
            // hash, and a hash is not an identity.
            bool reusable = false;
            if (jit_compiler::file_exist(so) && jit_compiler::file_exist(cc))
            {
                try { reusable = jit_compiler::read_all(cc) == src; }
                catch (...) { reusable = false; }
            }
            if (!reusable)
            {
                string cc_tmp = cc + ".tmp" + S((int)getpid());
                string so_tmp = so + ".tmp" + S((int)getpid());
                jit_compiler::write(cc_tmp, src);
                string log = base + ".log";
                string cmd = "\"" + tc.ccec + "\"" + tc.flags + " \"" + cc_tmp
                    + "\" -o \"" + so_tmp + "\"" + tc.link
                    + " > \"" + log + "\" 2>&1";
                LOGvvv << "AscendC fusion compile:" << cmd;
                int status = std::system(cmd.c_str());
                if (status != 0 || !jit_compiler::file_exist(so_tmp))
                {
                    LOGw << "AscendC fusion: ccec failed (" << status
                         << "), keeping the per-node path for this group; see"
                         << log << "and" << cc_tmp;
                    std::remove(so_tmp.c_str());
                    lib_cache[src] = KernelLib();
                    return nullptr;
                }
                std::rename(cc_tmp.c_str(), cc.c_str());
                std::rename(so_tmp.c_str(), so.c_str());
            }

            void *handle = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle)
            {
                LOGw << "AscendC fusion: dlopen failed:" << dlerror();
                lib_cache[src] = KernelLib();
                return nullptr;
            }
            LaunchFn fn = (LaunchFn)dlsym(handle, "jt_asc_launch");
            if (!fn)
            {
                LOGw << "AscendC fusion: jt_asc_launch missing in" << so;
                dlclose(handle);
                lib_cache[src] = KernelLib();
                return nullptr;
            }
            KernelLib lib{handle, fn};
            lib_cache[src] = lib;
            return fn;
        }

        // ---- analysis -----------------------------------------------------
        bool analyze(FusedOp *fop, Plan &plan)
        {
            const auto &ops = fop->ops;
            const int nn = (int)ops.size();
            if (nn == 0 || nn > kMaxNodes) return reject("group too large");
            if (nn < acl_ascendc_min_nodes) return reject("group below acl_ascendc_min_nodes");

            // Dtypes present.  A group with any half/bfloat16 var computes in
            // float32 (this arch has no bfloat16 vector arithmetic through
            // these intrinsics) and re-rounds where the graph does, so the
            // result is the one the per-node path produces.
            bool any_f16 = false, any_bf16 = false;
            auto note_dtype = [&](NanoString d) -> bool
            {
                int es; string ct;
                if (!supported_dtype(d, es, ct)) return false;
                if (d == ns_float16) any_f16 = true;
                if (d == ns_bfloat16) any_bf16 = true;
                return true;
            };
            for (Op *op : ops)
            {
                if (std::strcmp(op->name(), "array") == 0)
                {
                    // A Python number an operator met: one element, float32,
                    // host value available.  It becomes a Duplicate in the
                    // kernel instead of a group input.
                    ArrayOp *aop = static_cast<ArrayOp *>(op);
                    Var *cst = op->output(0);
                    if (cst->num != 1) return reject("non-scalar constant");
                    int cbytes = 4;
                    string ctype_name;
                    if (!supported_dtype(cst->dtype(), cbytes, ctype_name))
                        return reject((string("constant dtype ") + cst->dtype().to_cstring()).c_str());
                    if (!aop->host_value_known()) return reject("constant not on host");
                    continue;
                }
                for (Var *v : op->inputs())
                    if (!note_dtype(v->dtype()))
                        return reject((string("input dtype ") + v->dtype().to_cstring()).c_str());
                for (Var *v : op->outputs())
                    if (!note_dtype(v->dtype()))
                        return reject((string("output dtype ") + v->dtype().to_cstring()).c_str());
                if (op->outputs().size() != 1) return reject("multi-output node");
                // Casts between the two 2-byte types would have to convert in
                // float space and re-round; both directions are legal but no
                // model emits them, and refusing keeps the scratch single-typed.
                if (std::strcmp(op->name(), "unary") == 0 && op->ns == ns_cast)
                {
                    if (is_2byte(op->input(0)->dtype()) && is_2byte(op->output(0)->dtype())
                            && op->input(0)->dtype() != op->output(0)->dtype())
                        return reject("cast between half and bfloat16");
                }
            }
            if (any_f16 && any_bf16) return reject("mixed half and bfloat16 in one group");
            plan.widen = any_f16 || any_bf16;
            plan.wide_dtype = any_bf16 ? ns_bfloat16 : (any_f16 ? ns_float16 : ns_float32);

            // Only elementwise nodes with an exact intrinsic.
            for (Op *op : ops)
            {
                const char *name = op->name();
                if (std::strcmp(name, "binary") == 0)
                {
                    if (!binary_intrinsic(op->ns)) return reject((string("binary ") + op->ns.to_cstring()).c_str());
                }
                else if (std::strcmp(name, "unary") == 0)
                {
                    if (!unary_supported(op->ns)) return reject((string("unary ") + op->ns.to_cstring()).c_str());
                }
                else if (std::strcmp(name, "contiguous") != 0
                        && std::strcmp(name, "array") != 0
                        && std::strcmp(name, "broadcast_to") != 0)
                {
                    return reject((string("node ") + name).c_str());
                }
                if (std::strcmp(name, "array") == 0) continue;
                for (Var *v : op->inputs())
                {
                    if (var_is_plain(v, v->dtype())) continue;
                    if (broadcast_period(v, vec_lane(v->dtype())) > 0) continue;
                    std::ostringstream d;
                    d << "input is a strided view shape=" << v->shape
                      << " strides=" << v->storage_strides
                      << " off=" << (long long)v->storage_offset_bytes;
                    return reject(d.str().c_str());
                }
                for (Var *v : op->outputs())
                    if (!var_is_plain(v, v->dtype()))
                        return reject("output is a strided view");
            }

            // Topological order over the group.
            std::unordered_map<Var *, int> producer;   // var -> node index in `ops`
            for (int i = 0; i < nn; i++) producer[ops[i]->output(0)] = i;

            int64 numel = -1;
            for (Op *op : ops)
            {
                if (std::strcmp(op->name(), "array") == 0) continue;
                // Outputs define the iteration domain. A computed scalar
                // broadcast has one external input element and N outputs;
                // the scalar is loaded once through the period-one buffer.
                for (Var *v : op->outputs())
                {
                    if (numel < 0) numel = v->num;
                    if (v->num != numel) return reject("mixed element counts (broadcast)");
                }
                for (Var *v : op->inputs())
                {
                    if (producer.count(v)) continue;   // produced in-group
                    if (v->num != numel && !(v->num == 1 &&
                            std::strcmp(op->name(), "broadcast_to") == 0))
                        return reject("mixed element counts (broadcast)");
                }
            }
            if (numel <= 0 || numel > (int64)0x7fffffff) return reject("element count out of range");
            vector<int> indeg(nn, 0);
            vector<vector<int>> succ(nn);
            for (int i = 0; i < nn; i++)
                for (Var *v : ops[i]->inputs())
                {
                    auto f = producer.find(v);
                    if (f == producer.end()) continue;
                    succ[f->second].push_back(i);
                    indeg[i]++;
                }
            vector<int> order;
            order.reserve(nn);
            std::queue<int> q;
            for (int i = 0; i < nn; i++) if (indeg[i] == 0) q.push(i);
            while (!q.empty())
            {
                int i = q.front(); q.pop();
                order.push_back(i);
                for (int j : succ[i]) if (--indeg[j] == 0) q.push(j);
            }
            if ((int)order.size() != nn) return reject("cyclic group");   // not ours to run

            vector<int> pos(nn, -1);
            for (int k = 0; k < nn; k++) pos[order[k]] = k;

            plan.nodes.assign(nn, Node());
            plan.inputs.clear();
            plan.input_period.clear();
            plan.input_dtype.clear();
            plan.outputs.clear();
            plan.output_dtype.clear();
            int64 align = plan.widen ? 16 : 8;
            std::unordered_map<Var *, int> input_index;
            int n_out = 0;
            for (int k = 0; k < nn; k++)
            {
                Op *op = ops[order[k]];
                Node &nd = plan.nodes[k];
                nd.kind = classify(op);
                nd.ns = op->ns;
                nd.op_idx = order[k];
                nd.nin = (int)op->inputs().size();
                Var *out = op->output(0);
                if (nd.kind == kConst)
                {
                    ArrayOp *aop = static_cast<ArrayOp *>(op);
                    if (nd.nin != 0) return reject("constant arity");
                    if (out->mem_ptr) return reject("materialized constant");
                    nd.dtype = out->dtype();
                    nd.round = false;
                    nd.cval = array_scalar_value(aop);
                    nd.out_mat = -1;
                    continue;
                }
                if (nd.kind == kBinary && nd.nin != 2) return reject("binary arity");
                if (nd.kind != kBinary && nd.nin != 1) return reject("unary arity");
                if (std::strcmp(op->name(), "broadcast_to") == 0)
                {
                    auto f = producer.find(op->input(0));
                    if (f != producer.end()) {
                        if (plan.nodes[pos[f->second]].kind != kConst)
                            return reject("broadcast of a non-constant");
                    } else if (op->input(0)->num != 1) {
                        return reject("computed broadcast of a non-scalar input");
                    }
                }
                nd.dtype = out->dtype();
                // Where the graph's value is half/bf16 but we compute in float,
                // round the same way the per-node op does before the next node
                // (or the store) reads it.  A copy of an already-rounded value
                // is exact and needs no second rounding.
                nd.round = plan.widen && is_2byte(nd.dtype) && nd.kind != kCopy;
                if (std::strcmp(op->name(), "unary") == 0 && op->ns == ns_cast
                        && op->input(0)->dtype() == out->dtype())
                    nd.round = false;   // same-dtype cast is a copy
                if (nd.round) plan.scratch = true;

                for (int j = 0; j < nd.nin; j++)
                {
                    Var *v = op->input(j);
                    auto f = producer.find(v);
                    if (f != producer.end())
                    {
                        nd.src[j] = pos[f->second];
                        continue;
                    }
                    if (!var_on_device(v)) return reject("group input not device-resident/aligned");
                    auto g = input_index.find(v);
                    if (g == input_index.end())
                    {
                        int idx = (int)plan.inputs.size();
                        int per = v->num == 1 && numel > 1 ? 1 :
                            (var_is_plain(v, v->dtype()) ? 0 : broadcast_period(v, vec_lane(v->dtype())));
                        input_index[v] = idx;
                        plan.inputs.push_back(VarSlot{k, j});
                        plan.input_period.push_back(per);
                        plan.input_dtype.push_back(v->dtype());
                        int lane = vec_lane(v->dtype());
                        align = align / gcd64(align, lane) * lane;
                        if (per > 1)
                        {
                            align = align / gcd64(align, per) * per;
                            if (align > kMaxTile) return reject("broadcast period too large");
                        }
                        nd.src[j] = -idx - 1;
                    }
                    else nd.src[j] = -g->second - 1;
                }
                if (out->mem_ptr)
                {
                    if (!var_on_device(out)) return reject("output not device-resident/aligned");
                    nd.out_mat = n_out++;
                    plan.outputs.push_back(VarSlot{k, -1});
                    plan.output_dtype.push_back(out->dtype());
                }
                else
                {
                    nd.out_mat = -1;
                    if (out->allocator) return reject("half-allocated intermediate");
                }
            }
            if (plan.outputs.empty()) return reject("no materialized output");
            if ((int)plan.inputs.size() + n_out > kMaxArgs) return reject("too many kernel arguments");

            // Every intermediate must be consumed, otherwise the node is dead.
            vector<char> consumed(nn, 0);
            for (int k = 0; k < nn; k++)
                for (int j = 0; j < plan.nodes[k].nin; j++)
                    if (plan.nodes[k].src[j] >= 0) consumed[plan.nodes[k].src[j]] = 1;
            for (int k = 0; k < nn; k++)
                if (plan.nodes[k].out_mat < 0 && !consumed[k]) return reject("dead intermediate");

            plan.align = (int)align;
            int tile = 0;
            string src = generate_source(plan, tile);
            if (src.empty()) return reject("unified buffer budget exceeded");
            LaunchFn fn = build_or_get(src);
            if (!fn) return reject("ccec compile/load failed");
            plan.launch = fn;
            plan.ok = true;
            return true;
        }

        // Re-checks, against the vars of this execution, every fact the plan
        // was built on.
        bool validate(FusedOp *fop, const Plan &plan, int64 &numel_out)
        {
            const auto &ops = fop->ops;
            if (ops.size() != plan.nodes.size()) return false;
            const int nn = (int)plan.nodes.size();
            int64 numel = -1;
            for (int k = 0; k < nn; k++)
            {
                const Node &nd = plan.nodes[k];
                if (nd.op_idx >= (int)ops.size()) return false;
                Op *op = ops[nd.op_idx];
                if (classify(op) != nd.kind || op->ns.data != nd.ns.data) return false;
                if ((int)op->inputs().size() != nd.nin) return false;
                if (op->outputs().size() != 1) return false;
                Var *out = op->output(0);
                if (nd.kind == kConst)
                {
                    ArrayOp *aop = static_cast<ArrayOp *>(op);
                    if (out->num != 1 || out->dtype() != ns_float32) return false;
                    if (!aop->host_value_known()) return false;
                    if (array_scalar_value(aop) != nd.cval) return false;
                    continue;
                }
                if (out->dtype() != nd.dtype) return false;
                if (out->num != numel && numel >= 0) return false;
                if (numel < 0) numel = out->num;
                for (int j = 0; j < nd.nin; j++)
                {
                    Var *v = op->input(j);
                    if (nd.src[j] >= 0)
                    {
                        if (!var_is_plain(v, v->dtype())) return false;
                        if (nd.src[j] >= nn) return false;
                        if (v != ops[plan.nodes[nd.src[j]].op_idx]->output(0)) return false;
                    }
                    else
                    {
                        if (v->num != numel && !(v->num == 1 &&
                                std::strcmp(op->name(), "broadcast_to") == 0)) return false;
                        int g = -nd.src[j] - 1;
                        if (g >= (int)plan.inputs.size()) return false;
                        if (v != locate(fop, plan.nodes, plan.inputs[g])) return false;
                        if (!var_on_device(v)) return false;
                        if (v->dtype() != plan.input_dtype[g]) return false;
                        int per = v->num == 1 && numel > 1 ? 1 :
                            (var_is_plain(v, v->dtype()) ? 0 : broadcast_period(v, vec_lane(v->dtype())));
                        if (per != plan.input_period[g]) return false;
                    }
                }
                if (!var_is_plain(out, out->dtype())) return false;
                if (nd.out_mat >= 0)
                {
                    if (nd.out_mat >= (int)plan.outputs.size()) return false;
                    if (out != locate(fop, plan.nodes, plan.outputs[nd.out_mat])) return false;
                    if (!var_on_device(out)) return false;
                }
                else if (out->mem_ptr || out->allocator) return false;
            }
            if (numel <= 0 || numel > (int64)0x7fffffff) return false;

            // Elementwise in place is safe only when an input and an output are
            // the *same* buffer.
            for (int oi = 0; oi < (int)plan.outputs.size(); oi++)
            {
                Var *ov = locate(fop, plan.nodes, plan.outputs[oi]);
                char *ob = (char *)ov->mem_ptr;
                const int64 out_bytes = numel * dtype_size(ov->dtype());
                for (int ii = 0; ii < (int)plan.inputs.size(); ii++)
                {
                    Var *iv = locate(fop, plan.nodes, plan.inputs[ii]);
                    char *ib = (char *)iv->mem_ptr;
                    int64 in_bytes = plan.input_period[ii]
                        ? (int64)plan.input_period[ii] * dtype_size(iv->dtype()) : out_bytes;
                    if (ib == ob && in_bytes == out_bytes) continue;   // exact alias
                    if (ib < ob + out_bytes && ob < ib + in_bytes) return false;
                }
                for (int oj = oi + 1; oj < (int)plan.outputs.size(); oj++)
                {
                    Var *o2v = locate(fop, plan.nodes, plan.outputs[oj]);
                    char *o2 = (char *)o2v->mem_ptr;
                    int64 o2_bytes = numel * dtype_size(o2v->dtype());
                    if (o2 < ob + out_bytes && ob < o2 + o2_bytes) return false;
                }
            }
            numel_out = numel;
            return true;
        }

        std::mutex plan_mutex;
        // A null entry is a group this path has already declined. Plans are
        // shared and never mutated after analysis, so an execution copies one
        // pointer rather than the node and slot vectors.
        std::unordered_map<const void *, std::shared_ptr<const Plan>> plan_cache;

    } // namespace

    int64 acl_ascendc_fused_count() { return fused_count; }
    int64 acl_ascendc_fused_node_count() { return fused_node_count; }

    string acl_ascendc_reject_report()
    {
        std::lock_guard<std::mutex> lock(reject_mutex);
        std::ostringstream s;
        for (const auto &kv : reject_counts) s << kv.second << "\t" << kv.first << "\n";
        return s.str();
    }

    bool exec_fused_ascendc(FusedOp *fop)
    {
        if (!use_acl_ascendc_fusion)
        {
            static bool said = false;
            if (!said && getenv("acl_ascendc_report"))
            {
                said = true;
                std::fprintf(stderr, "ACL_ASCENDC_OFF use_acl_ascendc_fusion=%d\n",
                             (int)use_acl_ascendc_fusion);
            }
            return false;
        }

        // The plan is cached under the compiled fused context, which the jit
        // cache keys by jit key, but the cache is only a hint: `validate`
        // re-derives from this execution's vars every fact the generated
        // kernel encodes, so a context address the allocator reused for a
        // different group cannot make us launch the wrong kernel.
        const void *key = fop->context;
        std::shared_ptr<const Plan> plan;
        if (key)
        {
            std::lock_guard<std::mutex> lock(plan_mutex);
            auto found = plan_cache.find(key);
            if (found != plan_cache.end())
            {
                if (!found->second) return false;
                plan = found->second;
            }
        }

        int64 numel = 0;
        if (!plan || !validate(fop, *plan, numel))
        {
            auto fresh = std::make_shared<Plan>();
            if (!analyze(fop, *fresh) || !validate(fop, *fresh, numel))
            {
                if (key)
                {
                    std::lock_guard<std::mutex> lock(plan_mutex);
                    plan_cache[key] = nullptr;
                }
                return false;
            }
            plan = fresh;
            if (key)
            {
                std::lock_guard<std::mutex> lock(plan_mutex);
                plan_cache[key] = fresh;
            }
        }

        void *args[kMaxArgs];
        int na = 0;
        for (const auto &slot : plan->inputs)
            args[na++] = locate(fop, plan->nodes, slot)->mem_ptr;
        for (const auto &slot : plan->outputs)
            args[na++] = locate(fop, plan->nodes, slot)->mem_ptr;

        uint32_t n = (uint32_t)numel;
        uint32_t bd = (uint32_t)((numel + kElemsPerBlock - 1) / kElemsPerBlock);
        const uint32_t cores = (uint32_t)vector_cores();
        if (bd < 1) bd = 1;
        if (bd > cores) bd = cores;
        plan->launch(args, n, bd, aclstream);
        fused_count++;
        fused_node_count += (int64)plan->nodes.size();
        if (getenv("acl_ascendc_report"))
        {
            static int64 said = 0;
            if (said++ < 20)
                std::fprintf(stderr, "ACL_ASCENDC_TAKE wide=%s nodes=%d numel=%u\n",
                             plan->wide_dtype.to_cstring(), (int)plan->nodes.size(), n);
        }
        return true;
    }

} // jittor

// Lets a benchmark A/B the two paths inside one process, where the machine's
// background load is the same for both halves.
extern "C" void jt_acl_ascendc_set_enabled(int on)
{
    jittor::use_acl_ascendc_fusion = on;
}

extern "C" int jt_acl_ascendc_enabled()
{
    return jittor::use_acl_ascendc_fusion;
}

extern "C" int64_t jt_acl_ascendc_fused_count()
{
    return (int64_t)jittor::acl_ascendc_fused_count();
}

extern "C" int64_t jt_acl_ascendc_fused_node_count()
{
    return (int64_t)jittor::acl_ascendc_fused_node_count();
}

extern "C" const char *jt_acl_ascendc_reject_report()
{
    static std::string buffer;
    buffer = jittor::acl_ascendc_reject_report();
    return buffer.c_str();
}
