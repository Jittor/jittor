// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/fake_main_pass.h"

namespace jittor {

void FakeMainPass::run() {
    // if this op is relayed, we don't run fake main pass
    for (auto& o : op->ops)
        if (!pm->oc->op_exist(o))
            return;

    // TODO: fake_main only supported when compile_shapes
    if (!op->get_loop_option("jtune"))
        return;
    all->push_back("#include <iomanip>");
    all->push_back("#include <chrono>");
    all->push_back("using namespace std;");
    all->push_back("using namespace jittor;");
    if (op->flags.get(NodeFlags::_cpu)) {
        all->push_back("void* _fake_alloc(size_t size) {\n"
            "return aligned_alloc(alignment, size);\n"
        "}", nullptr, true);
    } else {
        all->push_back("void* _fake_alloc(size_t size) {\n"
            "char* ptr;\n"
            "checkCudaErrors(cudaMallocManaged((void**)&ptr, sizeof(char)*size));\n"
            "return (void*) ptr;\n"
        "}", nullptr, true);
    }
    all->push_back("int64_t _getenv(const char* name, int64_t _default) {\n"
        "auto* v = getenv(name);\n"
        "return v?stoll(v):_default;\n"
    "}", nullptr, true);
    all->push_back("void output_float(const string& scale, int base, const string& suffix, double k) {\n"
        "uint w=10, p=3;\n"
        "cout << ' ' << std::setw(w-2-suffix.size());\n"
        "cout << std::setprecision(p);\n"
        "uint i=0;\n"
        "for (; i+1<scale.size(); i++) {\n"
        "    if (k<base) break;\n"
        "    k /= base;\n"
        "}\n"
        "cout << k << scale[i];\n"
        "cout << suffix;\n"
    "}", nullptr, true);
    all->push_back("extern \"C\" void fake_main() {\n"
        "cout << \"Enter fake_main entry.\" << endl;\n"
        "#define fake_new(T) ((T*)(new char[sizeof(T)]()))\n"
        "auto* op = fake_new(FusedOp);\n"
        "auto& ops = op->ops;\n"
        "Var* var;"
    "}", nullptr, true);
    auto& main = all->children.back();
    // fake ops
    for (uint i=0; i<op->ops.size(); i++) {
        auto* opi = op->ops[i];
        string name = opi->name();
        string name2 = Op::op_name_to_file_name(name);
        string name3 = Op::file_name_to_class_name(name2);
        main->push_back(
            "ops.push_back(fake_new("+name3+"Op));"
        );
        if (name3=="Array") {
            main->push_back(
                "{\n"
                    "auto ptr = new double[1];\n"
                    "((ArrayOp*)(ops["+S(i)+"]))->allocation.ptr = (void*)ptr;\n"
                    "ptr[0] = 0;\n"
                "}\n"
            );
            main->push_back("ArrayOp* op"+S(i)+"=((ArrayOp*)(ops["+S(i)+"]));");
        }
    }
    // fake vars
    map<size_t, string> var_map;
    for (auto& c : ir->children) {
        if (c->type != "define") continue;
        auto& name = c->attrs["lvalue"];
        auto& rvalue = c->attrs["rvalue"];
        uint op_id, var_id;
        Op* op;
        Var* var;
        try {
            pm->oc->get_op_var_by_name(name, op_id, var_id, op, var);
        } catch (...) {
            continue;
        }
        // build fake var
        auto vec_to_str = [](const NanoVector& v) -> string {
            std::stringstream ss;
            ss << '{';
            for (uint i=0; i<v.size(); i++)
                ss << v[i] << (i+1==v.size() ? ' ':',');
            ss << '}';
            return ss.str();
        };
        // if two var are the same, we share their memory
        if (var_map.count((size_t)var)) {
            main->push_back(rvalue+" = "+var_map[(size_t)var]+";", nullptr, true);
            continue;
        }
        var_map[(size_t)var] = rvalue;
        main->push_back("{\n"
            +rvalue+"= var = fake_new(Var);\n"
            "var->flags.flags = "+S(var->flags.flags)+";\n"
            "var->shape = "+vec_to_str(var->shape)+";\n"
            "var->size = "+S(var->size)+";\n"
            "var->num = "+S(var->num)+";\n"
            "var->mem_ptr = _fake_alloc(var->size);\n"
        "}", nullptr, true);
    }
    uint64_t in, out, compute;
    op->statistics(in, out, compute);
    string need_sync = op->flags.get(NodeFlags::_cuda) ? "checkCudaErrors(cudaDeviceSynchronize());\n" : "";
    main->push_back("{\n"
        "auto warmup = _getenv(\"warmup\", 2);\n"
        "auto rerun = _getenv(\"rerun\", 10);\n"
        "int loop = "+S(op->get_loop_option("insert_profile_loop")?10:0)+";\n"
        "warmup = warmup ? std::max(warmup>>loop, (int64_t)1) : 0;\n"
        "rerun = std::max((rerun+1)>>loop, (int64_t)1);\n"
        "int64_t in = "+S(in)+";\n"
        "int64_t out = "+S(out)+";\n"
        "int64_t compute = "+S(compute)+";\n"
        "int64_t count=0, time_max=0, time_min=1ll<<62, time_total=0;\n"
        "int64_t in_total=0, out_total=0, compute_total=0;\n"
        "for (int64_t i=0; i<warmup; i++) op->jit_run();\n"
        +need_sync+
        "for (int64_t i=0; i<rerun; i++) {\n"
            "auto start = std::chrono::high_resolution_clock::now();\n"
            "op->jit_run();\n"
            +need_sync+
            "auto finish = std::chrono::high_resolution_clock::now();\n"
            "auto total_ns = (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();\n"
            "// 24ns function call overhead\n"
            "total_ns = std::max((int64_t)1, total_ns-24);\n"
            "count += 1<<loop;\n"
            "time_max = std::max(time_max, total_ns>>loop);\n"
            "time_min = std::min(time_min, total_ns>>loop);\n"
            "time_total += total_ns;\n"
            "in_total += in<<loop;\n"
            "out_total += out<<loop;\n"
            "compute_total += compute<<loop;\n"
        "}\n"
        "cout << \"     Count TotalTime   AvgTime   MinTime   MaxTime     Input    Output   Compute\" << endl;\n"
        "cout << setw(10) << count;\n"
        "output_float(\"num \", 1000, \"s\", time_total*1.0);\n"
        "output_float(\"num \", 1000, \"s\", time_total*1.0/count);\n"
        "output_float(\"num \", 1000, \"s\", time_min*1.0);\n"
        "output_float(\"num \", 1000, \"s\", time_max*1.0);\n"
        "output_float(\" KMG\", 1024, \"B/s\", in_total*1e9/time_total);\n"
        "output_float(\" KMG\", 1024, \"B/s\", out_total*1e9/time_total);\n"
        "output_float(\" KMG\", 1000, \"it/s\", compute_total*1e9/time_total);\n"
        "cout << endl;\n"
    "}", nullptr, true);
}

} // jittor