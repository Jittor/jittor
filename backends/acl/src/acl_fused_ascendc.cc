// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// One generated AscendC kernel per fused float32 elementwise group.
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
//   * every var is float32, contiguous, on the device, 32-byte aligned and has
//     the same element count;
//   * every node is a unary/binary/copy whose AscendC intrinsic was measured
//     bit-identical to the aclnn op the generic path would have run
//     (add/sub/mul/max/min/abs/neg are exact; div is within 1 ulp);
//   * a node output is written to global memory exactly when the executor
//     allocated it, and lives only in unified buffer otherwise.
// Anything else returns false and the caller runs its normal per-node path.
// Nothing here ever selects a different backend.
#include <acl/acl.h>
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
        "Run eligible float32 elementwise fused groups as one generated AscendC kernel");
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
                reject_counts[why]++;
            }
            return false;
        }

        // ---- node model -------------------------------------------------
        enum NodeKind { kCopy = 0, kUnary = 1, kBinary = 2 };

        struct Node
        {
            NodeKind kind;
            NanoString ns;      // the operation this node performs
            int op_idx;         // index into FusedOp::ops
            int nin;
            int src[2];         // >=0 value index, <0 -(group input index + 1)
            int out_mat;        // materialized output index, or -1
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
            int align = 8;
            LaunchFn launch = nullptr;
        };

        Var *locate(FusedOp *fop, const vector<Node> &nodes, const VarSlot &s)
        {
            Op *op = fop->ops[nodes[s.node].op_idx];
            return s.port < 0 ? op->output(0) : op->input(s.port);
        }

        // ---- what the AscendC vector API can reproduce exactly -----------
        // Every entry was measured against the aclnn op the generic path runs
        // (bench/aclprobe/asc_numerics.cc): add/sub/mul/maximum/minimum/abs/neg
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

        // A pure broadcast view: some leading axes repeat (stride 0) and the
        // trailing axes are densely packed. Returns the period P, so the value
        // at flat index i is base[i % P]; 0 when the view is anything else.
        // A transposed or otherwise permuted view is not accepted here.
        int broadcast_period(Var *v)
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
            if (e != 1 && (e % 8) != 0) return 0;    // 32-byte copies only
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

        NodeKind classify(Op *op)
        {
            const char *n = op->name();
            if (std::strcmp(n, "binary") == 0) return kBinary;
            if (std::strcmp(n, "unary") == 0) return kUnary;
            return kCopy;   // only reached for "contiguous", checked by caller
        }

        // ---- source generation -------------------------------------------
        int64 gcd64(int64 a, int64 b) { while (b) { int64 t = a % b; a = b; b = t; } return a; }

        string generate_source(const vector<Node> &nodes, const vector<int> &period,
                               int n_out, int align, int &tile_out)
        {
            // Temporary-slot allocation: a value that is not written to global
            // memory lives in unified buffer until its last consumer.
            const int nn = (int)nodes.size();
            const int n_in = (int)period.size();
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

            int n_dense = 0, n_bcast = 0;
            for (int i = 0; i < n_in; i++) (period[i] ? n_bcast : n_dense)++;

            // Dense operands are double-buffered queues; a broadcast operand is
            // one buffer filled once before the loop.
            int slots = n_dense * 2 + n_out * 2 + n_temp + n_bcast;
            if (slots <= 0) return string();
            int tile = kUbBudget / (slots * 4);
            tile = tile & ~63;
            if (tile > kMaxTile) tile = kMaxTile;
            tile = tile / align * align;
            if (tile < align || tile < kMinTile) return string();
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
                return (period[g] ? "p" : "x") + S(g);
            };

            std::ostringstream s;
            s << "// generated by jittor backends/acl/src/acl_fused_ascendc.cc\n"
              << "#include \"kernel_operator.h\"\n#include <acl/acl.h>\nusing namespace AscendC;\n\n"
              << "constexpr int32_t JT_TILE = " << tile << ";\n"
              << "constexpr uint32_t JT_ALIGN = " << align << ";\n\n"
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
                if (!period[i]) s << "    pipe.InitBuffer(qi" << i << ", 2, JT_TILE * sizeof(float));\n";
            for (int i = 0; i < n_out; i++)
                s << "    pipe.InitBuffer(qo" << i << ", 2, JT_TILE * sizeof(float));\n";
            for (int i = 0; i < n_in; i++)
                if (period[i]) s << "    pipe.InitBuffer(bp" << i << ", JT_TILE * sizeof(float));\n";
            for (int i = 0; i < n_in; i++)
                if (period[i] == 1) s << "    pipe.InitBuffer(bs" << i << ", 32);\n";
            for (int i = 0; i < n_temp; i++)
                s << "    pipe.InitBuffer(bt" << i << ", JT_TILE * sizeof(float));\n";
            s << "\n";
            for (int i = 0; i < n_in; i++)
            {
                s << "    GlobalTensor<float> mi" << i << ";\n";
                if (period[i])
                    // A broadcast operand is read from its own base, never from
                    // the element offset this block is working on.
                    s << "    mi" << i << ".SetGlobalBuffer((__gm__ float*)gi" << i
                      << ", " << period[i] << ");\n";
                else
                    s << "    mi" << i << ".SetGlobalBuffer((__gm__ float*)gi" << i << " + start, len);\n";
            }
            for (int i = 0; i < n_out; i++)
                s << "    GlobalTensor<float> mo" << i << ";\n"
                  << "    mo" << i << ".SetGlobalBuffer((__gm__ float*)go" << i << " + start, len);\n";

            // Materialise every broadcast operand once, tiled to JT_TILE. Both
            // `start` and every tile offset are multiples of JT_ALIGN, which is
            // a multiple of each period, so a tile always begins at phase 0 of
            // the pattern and the prefix of this buffer is the right answer for
            // a short final tile too.
            bool any_bcast = false;
            for (int i = 0; i < n_in; i++) if (period[i]) any_bcast = true;
            if (any_bcast) s << "\n";
            for (int i = 0; i < n_in; i++)
            {
                if (!period[i]) continue;
                s << "    LocalTensor<float> p" << i << " = bp" << i << ".Get<float>();\n";
                if (period[i] == 1)
                {
                    // Reading unified buffer from the scalar unit right after
                    // an MTE2 copy needs an explicit MTE2->S event. --cce-auto-sync
                    // inserts one, but this is the single place where a missing
                    // barrier would produce a stale value instead of a compile
                    // error, so the barrier is written out.
                    s << "    {\n"
                      << "        LocalTensor<float> sv = bs" << i << ".Get<float>();\n"
                      << "        DataCopyExtParams sep{1, (uint32_t)sizeof(float), 0, 0, 0};\n"
                      << "        DataCopyPadExtParams<float> spp{false, 0, 0, 0};\n"
                      << "        DataCopyPad(sv, mi" << i << ", sep, spp);\n"
                      << "        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                      << "        SetFlag<HardEvent::MTE2_S>(ev);\n"
                      << "        WaitFlag<HardEvent::MTE2_S>(ev);\n"
                      << "        Duplicate(p" << i << ", sv.GetValue(0), JT_TILE);\n"
                      << "    }\n";
                }
                else
                {
                    s << "    for (int32_t r = 0; r < JT_TILE / " << period[i] << "; r++)\n"
                      << "        DataCopy(p" << i << "[r * " << period[i] << "], mi" << i
                      << ", " << period[i] << ");\n";
                }
            }

            s << "\n    for (uint32_t off = 0; off < len; off += JT_TILE) {\n"
              << "        uint32_t cur = (len - off) < (uint32_t)JT_TILE ? (len - off) : (uint32_t)JT_TILE;\n"
              << "        DataCopyExtParams ep{1, cur * (uint32_t)sizeof(float), 0, 0, 0};\n"
              << "        DataCopyPadExtParams<float> pp{false, 0, 0, 0};\n";
            for (int i = 0; i < n_in; i++)
            {
                if (period[i]) continue;
                s << "        LocalTensor<float> x" << i << " = qi" << i << ".AllocTensor<float>();\n"
                  << "        if (cur % 8 == 0) DataCopy(x" << i << ", mi" << i << "[off], cur);\n"
                  << "        else DataCopyPad(x" << i << ", mi" << i << "[off], ep, pp);\n"
                  << "        qi" << i << ".EnQue(x" << i << ");\n";
            }
            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "        x" << i << " = qi" << i << ".DeQue<float>();\n";
            for (int i = 0; i < n_out; i++)
                s << "        LocalTensor<float> y" << i << " = qo" << i << ".AllocTensor<float>();\n";
            for (int i = 0; i < n_temp; i++)
                s << "        LocalTensor<float> t" << i << " = bt" << i << ".Get<float>();\n";

            for (int i = 0; i < nn; i++)
            {
                const Node &nd = nodes[i];
                string dst = value_name(i);
                s << "        ";
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
                    // float32 -> float32 copy (cast / contiguous). Multiplying
                    // by one is exact for every float32 including -0 and NaN.
                    s << "Muls(" << dst << ", " << operand(nd, 0) << ", 1.0f, cur);\n";
                }
            }

            for (int i = 0; i < n_in; i++)
                if (!period[i]) s << "        qi" << i << ".FreeTensor(x" << i << ");\n";
            for (int i = 0; i < n_out; i++)
                s << "        qo" << i << ".EnQue(y" << i << ");\n"
                  << "        y" << i << " = qo" << i << ".DeQue<float>();\n"
                  << "        if (cur % 8 == 0) DataCopy(mo" << i << "[off], y" << i << ", cur);\n"
                  << "        else DataCopyPad(mo" << i << "[off], y" << i << ", ep);\n"
                  << "        qo" << i << ".FreeTensor(y" << i << ");\n";
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
        string host_cxx_includes()
        {
            static const char *triples[] = {
                "x86_64-linux-gnu", "aarch64-linux-gnu", nullptr};
            string best;
            int best_v = -1;
            for (int v = 20; v >= 5; v--)
            {
                string base = "/usr/include/c++/" + S(v);
                if (access((base + "/type_traits").c_str(), R_OK) != 0) continue;
                if (v > best_v) { best_v = v; best = base; }
                break;
            }
            if (best.empty()) return string();
            string out = " -I" + best + " -I" + best + "/backward";
            for (int i = 0; triples[i]; i++)
            {
                string t = "/usr/include/" + string(triples[i]) + "/c++/" + S(best_v);
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
        // Ascend950PR (bench/aclprobe/asc_fuse_bench.cc): using all 56 rather
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
                else if (std::strcmp(name, "contiguous") != 0)
                {
                    return reject((string("node ") + name).c_str());
                }
                for (Var *v : op->inputs())
                {
                    if (v->dtype() != ns_float32)
                        return reject((string("input dtype ") + v->dtype().to_cstring()).c_str());
                    if (var_is_plain_float32(v)) continue;
                    if (broadcast_period(v) > 0) continue;
                    std::ostringstream d;
                    d << "input is a strided view shape=" << v->shape
                      << " strides=" << v->storage_strides
                      << " off=" << (long long)v->storage_offset_bytes;
                    return reject(d.str().c_str());
                }
                for (Var *v : op->outputs())
                    if (!var_is_plain_float32(v))
                        return reject(v->dtype() != ns_float32
                            ? (string("output dtype ") + v->dtype().to_cstring()).c_str()
                            : "output is a strided view");
                if (op->outputs().size() != 1) return reject("multi-output node");
            }

            const int64 numel = ops[0]->output(0)->num;
            if (numel <= 0 || numel > (int64)0x7fffffff) return reject("element count out of range");
            for (Op *op : ops)
            {
                for (Var *v : op->inputs()) if (v->num != numel) return reject("mixed element counts (broadcast)");
                for (Var *v : op->outputs()) if (v->num != numel) return reject("mixed element counts (broadcast)");
            }

            // Topological order over the group.
            std::unordered_map<Var *, int> producer;   // var -> node index in `ops`
            for (int i = 0; i < nn; i++) producer[ops[i]->output(0)] = i;
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

            // Classify vars. A node output goes to global memory exactly when
            // the executor already allocated it; everything else stays in UB.
            plan.nodes.assign(nn, Node());
            plan.inputs.clear();
            plan.input_period.clear();
            plan.outputs.clear();
            int64 align = 8;
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
                if (nd.kind == kBinary && nd.nin != 2) return reject("binary arity");
                if (nd.kind != kBinary && nd.nin != 1) return reject("unary arity");
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
                        int per = var_is_plain_float32(v) ? 0 : broadcast_period(v);
                        input_index[v] = idx;
                        plan.inputs.push_back(VarSlot{k, j});
                        plan.input_period.push_back(per);
                        if (per > 1)
                        {
                            align = align / gcd64(align, per) * per;
                            if (align > kMaxTile) return reject("broadcast period too large");
                        }
                        nd.src[j] = -idx - 1;
                    }
                    else nd.src[j] = -g->second - 1;
                }
                Var *out = op->output(0);
                if (out->mem_ptr)
                {
                    if (!var_on_device(out)) return reject("output not device-resident/aligned");
                    nd.out_mat = n_out++;
                    plan.outputs.push_back(VarSlot{k, -1});
                }
                else
                {
                    nd.out_mat = -1;
                    if (out->allocator) return reject("half-allocated intermediate");
                }
            }
            if (plan.outputs.empty()) return reject("no materialized output");
            if ((int)plan.inputs.size() + n_out > kMaxArgs) return reject("too many kernel arguments");

            // Every intermediate must be consumed, otherwise the node is dead
            // and the generated kernel would compute a value nothing reads.
            vector<char> consumed(nn, 0);
            for (int k = 0; k < nn; k++)
                for (int j = 0; j < plan.nodes[k].nin; j++)
                    if (plan.nodes[k].src[j] >= 0) consumed[plan.nodes[k].src[j]] = 1;
            for (int k = 0; k < nn; k++)
                if (plan.nodes[k].out_mat < 0 && !consumed[k]) return reject("dead intermediate");

            plan.align = (int)align;
            int tile = 0;
            string src = generate_source(plan.nodes, plan.input_period, n_out, plan.align, tile);
            if (src.empty()) return reject("unified buffer budget exceeded");
            LaunchFn fn = build_or_get(src);
            if (!fn) return reject("ccec compile/load failed");
            plan.launch = fn;
            plan.ok = true;
            return true;
        }

        // Re-checks, against the vars of this execution, every fact the plan
        // was built on. A plan is usable exactly when the group in front of us
        // has the structure the generated kernel encodes, so reusing a cache
        // entry can never silently apply the wrong kernel.
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
                for (int j = 0; j < nd.nin; j++)
                {
                    Var *v = op->input(j);
                    if (v->dtype() != ns_float32) return false;
                    if (numel < 0) numel = v->num;
                    if (v->num != numel) return false;
                    if (nd.src[j] >= 0)
                    {
                        if (!var_is_plain_float32(v)) return false;
                        if (nd.src[j] >= nn) return false;
                        if (v != ops[plan.nodes[nd.src[j]].op_idx]->output(0)) return false;
                    }
                    else
                    {
                        int g = -nd.src[j] - 1;
                        if (g >= (int)plan.inputs.size()) return false;
                        if (v != locate(fop, plan.nodes, plan.inputs[g])) return false;
                        if (!var_on_device(v)) return false;
                        // The generated kernel hard-codes each operand's period
                        // and the block alignment derived from it, so both have
                        // to be exactly what they were when it was generated.
                        int per = var_is_plain_float32(v) ? 0 : broadcast_period(v);
                        if (per != plan.input_period[g]) return false;
                    }
                }
                Var *out = op->output(0);
                if (!var_is_plain_float32(out)) return false;
                if (out->num != numel) return false;
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
            // the *same* buffer: every element of a tile is read before the
            // matching element is written. A partial overlap (two views into
            // one allocation at different offsets) is not, and neither is an
            // output landing on a broadcast operand, which every block reads
            // once at kernel entry while other blocks are already writing.
            const int64 out_bytes = numel * 4;
            for (int oi = 0; oi < (int)plan.outputs.size(); oi++)
            {
                char *ob = (char *)locate(fop, plan.nodes, plan.outputs[oi])->mem_ptr;
                for (int ii = 0; ii < (int)plan.inputs.size(); ii++)
                {
                    char *ib = (char *)locate(fop, plan.nodes, plan.inputs[ii])->mem_ptr;
                    int64 in_bytes = plan.input_period[ii]
                        ? (int64)plan.input_period[ii] * 4 : out_bytes;
                    if (ib == ob && in_bytes == out_bytes) continue;   // exact alias
                    if (ib < ob + out_bytes && ob < ib + in_bytes) return false;
                }
                for (int oj = oi + 1; oj < (int)plan.outputs.size(); oj++)
                {
                    char *o2 = (char *)locate(fop, plan.nodes, plan.outputs[oj])->mem_ptr;
                    if (o2 < ob + out_bytes && ob < o2 + out_bytes) return false;
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
        if (!use_acl_ascendc_fusion) return false;

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
