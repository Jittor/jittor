// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/pass/shared_reduce_pass.h"
#include <set>
#include <fstream>
#include <algorithm>
#include "ops/reduce_op.h"

namespace jittor {

int begin_integer(const string &str, int pos) {
    string digits = "";
    for (; pos < str.size() && str[pos] >= '0' && str[pos] <= '9'; pos ++)
        digits += str[pos];
    return stoi(digits);
}

// Returns the name of kernel function call that appeared in loop 
// and its corresponding kerenel index
std::pair<string, int> look_up_kernel_function(unique_ptr<KernelIR> &loop, vector<unique_ptr<KernelIR>> &functions) {
    string kernel_name = "";
    for (int i = 0; i < loop->children.size(); ++i) {
        auto& code = loop->children[i]->get_attr("code");
        if (code.substr(0, 5) == "func_" && code.find("<<<") != string::npos && code.find(">>>") != string::npos) {
            kernel_name = code.substr(0, code.find("<<<"));
            break;
        }
    }

    for (int i = 0; i < functions.size(); ++i)
        if (functions[i]->get_attr("lvalue") == kernel_name) {
            bool has_atomic = false;
            functions[i]->dfs([&](unique_ptr<KernelIR>& p) {
                if (p->get_attr("code").find("atomic") != string::npos)
                    has_atomic = true;
            });
            if (has_atomic)
                return std::make_pair(kernel_name, i);
        }
    return std::make_pair("", -1);
}

// Return the index of reduce op in a kernel function by finding "opIDX_" patterns
// Return -1 if not found
// TODO: consider more than one reduce op in a kernel
int look_up_reduce_op(unique_ptr<KernelIR> &kernel_ir, FusedOp *op) {
    std::set<int> ids;
    kernel_ir->dfs([&](unique_ptr<KernelIR>& p) {
        auto& code = p->attrs["code"];

        int start_pos = 0;
        while (true) {
            int pos = code.find("op", start_pos);
            if (pos == string::npos)
                break;
            int id = begin_integer(code, pos + 2);
            ASSERT(id >= 0 && id < op->ops.size());
            if (op->ops[id]->name() == string("reduce"))
                ids.insert(id);
            start_pos = pos + 1;
        }
    });

    if (ids.empty())
        return -1;
    else
        return *ids.begin();
}


/*
    Modify atomic operators to shared-memory-reduce in GPU kernel.

    before:
        atomicAdd(&(target),tmp);

    After:
        tmp = shared_reduce<float32, shared_reduce_add>(tmp);
        if (threadIdx.x == 0) {
            atomicAdd(&(target),tmp);
        }
*/
void modify_reduce_kernel(unique_ptr<KernelIR>& kernel) {
    vector<pair<unique_ptr<KernelIR>*, string>> atomic_irs;

    while (true) {
        string op = "";
        unique_ptr<KernelIR>* ir_p = nullptr;

        kernel->dfs([&](unique_ptr<KernelIR>& p) {
            string& code = p->attrs["code"];            
            if (p->father && p->father->type == "if")
                return;
            if (code.find("atomic") == 0) {
                op = code.substr(6, code.find("(") - 6);
                ASSERT(op == "Add" || op == "And" || op == "Or" || op == "Xor") << op;
                op[0] = op[0] - 'A' + 'a';
                ir_p = &p;
            }
            if (code.find("cuda_atomic_") == 0) {
                op = code.substr(12, code.find("(") - 12);
                ASSERT(op == "min" || op == "max" || op == "mul") << op;
                ir_p = &p;
            }
        });

        if (ir_p == nullptr)
            break;

        auto& atomic_ir = *ir_p;
        string code = atomic_ir->attrs["code"];
        string target = code.substr(code.find("&(")+2, code.rfind("),") - (code.find("&(")+2));
        string value = code.substr(code.rfind("),")+2, code.rfind(");") - (code.rfind("),")+2));

        if (value.find("(") != string::npos) {
            // extract "tmp0" from "cuda_atomic_max(&(op0_yp[op0_yid]),float32(tmp0))"
            value = value.substr(value.find("(") + 1, value.find(")") - (value.find("(") + 1));
        }

        auto father = atomic_ir->father;
        string dtype = atomic_ir->father->find_define(value)->get_attr("dtype");

        int pos = 0;
        for (; pos < atomic_ir->flist->size() && atomic_ir->flist->at(pos) != atomic_ir; pos++);
        ASSERT(pos < atomic_ir->flist->size());

        atomic_ir->attrs["code"] = value + " = shared_reduce<" + dtype + ", shared_reduce_" + op + ">(" + value + ");";
        father->push_back("if (threadIdx.x == 0) " + code, atomic_ir->flist, true);
    }
}

typedef map<int, vector<int>> T_Range;      // tnx = rangey * rangez *
std::tuple<int, vector<int>, T_Range> find_TNs_and_reorder(unique_ptr<KernelIR> &ir, ReduceOp *reduce_op) {
    vector<int> TNs;
    T_Range ranges;

    for (auto &define : ir->children) {
        if (define->type != "define")
            continue;
        if (define->get_attr("lvalue").substr(0, 2) != "tn")
            continue;
        
        // Now we assume the defination is:
        //  int tnX = get_thread_range_log(thread_num_left, rangeY * rangeZ * ...);
        int tn_x = stoi(define->get_attr("lvalue").substr(2));

        string rvalue = define->get_attr("rvalue");
        rvalue = rvalue.substr(rvalue.find(','));

        vector<int> dims;
        while (true) {
            int pos = rvalue.find("range");
            if (pos == string::npos)
                break;
            int range_y = begin_integer(rvalue, pos + 5);
            dims.push_back(range_y);
            rvalue = rvalue.substr(pos + 5);
        }
        TNs.push_back(tn_x);
        ranges[tn_x] = dims;
    }
    
    auto reduce_mask = reduce_op->reduce_mask;
    
    vector<int> TNs0;       // tnx (dims) to reduce
    vector<int> TNs1;       // tnx (dims) to be reduced
    for (int tn = 0; tn < TNs.size(); ++tn) {
        auto is_reduced = [&](int x) -> bool {
            return (1 << x) & reduce_mask;
        };

        int tn_x = TNs[tn];

        for (int k = 1; k < ranges[tn_x].size(); ++k) 
            ASSERT(is_reduced(ranges[tn_x][k]) == is_reduced(ranges[tn_x][0]));

        if (is_reduced(ranges[tn_x][0]))
            TNs0.push_back(tn_x);
        else
            TNs1.push_back(tn_x);
    }

    vector<int> TNs_new;
    for (int tn = 0; tn < TNs0.size(); ++tn)
        TNs_new.push_back(TNs0[tn]);
    for (int tn = 0; tn < TNs1.size(); ++tn)
        TNs_new.push_back(TNs1[tn]);
    return std::make_tuple(TNs0.size() - 1, TNs_new, ranges);
}


/*
    Modify the thread allocation before kernel launch.
    Take batchnorm as an example:
    0. the order of dimensions from the last pass are B, C, W, H
    1. first allocate the dimensions to reduce and then the dimensions to be reduced:
        B, C, W, H => C, B, W, H
    2. guarantee that the production of dimensions to be reduced no less than 1024 (number of threads).
        1 << tn = max(B*W*H, 1024)

    example JIT codes:

    before
        {
            int thread_num = 65536;
            int thread_num_left = thread_num;
            int tn1 = get_thread_range_log(thread_num_left, range1);
            int tn2 = get_thread_range_log(thread_num_left, range3 * range2);
            int tn0 = get_thread_range_log(thread_num_left, range0);
            tn1=tn1+tn2;
            tn0=tn0+tn1;
            tn0=std::max(tn0, 5);
            thread_num=1<<tn0;
            int p1 = std::max(thread_num/1024, 1);
            int p2 = std::min(thread_num, 1024);
            func_7cd1490029ee50f8_1<<<p1,p2>>>(range0,range1,range2,range3,op0_xp,op0_yp,tn2,tn1,tn0);
        }
    after
        {
            int thread_num = 65536;
            int thread_num_left = thread_num;
            int tn2 = get_thread_range_log(thread_num_left,  range3 * range2);
            int tn0 = get_thread_range_log(thread_num_left,  range0);
            int tn1 = get_thread_range_log(thread_num_left,  range1);
            tn0 = tn2 + tn0;
            tn0 = std::max(tn0, 5);
            tn1 = tn0 + tn1;
            thread_num=1<<tn1;
            int p1 = std::max(thread_num / std::min(1 << tn0, 1024), 1);
            int p2 = std::min(1 << tn0, 1024);
            func_7cd1490029ee50f8_1<<<p1,p2>>>(range0,range1,range2,range3,op0_xp,op1_yp,tn2,tn1,tn0);
        }
*/
void modify_reduce_launch(unique_ptr<KernelIR> &ir, unique_ptr<KernelIR> &kernel, ReduceOp *reduce_op) {
    auto ret = find_TNs_and_reorder(ir, reduce_op);
    int sep = std::get<0>(ret);       // tn (dims) in reorder[0...sep] is to be reduced
    auto reorder = std::get<1>(ret);   // reduce to tn (dims) in reorder[sep+1...]
    auto ranges = std::get<2>(ret);

    // Re-allocate threads
    int cid = 0;
    for (auto& child: ir->children) {
        if (child->type == "define" && child->get_attr("lvalue").substr(0, 2) == "tn")
            break;
        ++cid;
    }

    for (int tn : reorder) {
        string range_multiply = "";
        for (int i: ranges[tn])
            range_multiply += " * range" + std::to_string(i);
        ir->children[cid]->attrs["lvalue"] = "tn" + std::to_string(tn);
        ir->children[cid]->attrs["rvalue"] = "get_thread_range_log(thread_num_left, " + range_multiply.substr(2) + ")";
        ++cid;
    }

    if (sep == 0) {
        string t0 = "tn" + std::to_string(reorder[0]);
        ir->children[cid++]->attrs["code"] = t0 + " = std::max(" + t0 + ", 5);";
    } 

    for (int t = 0; t < reorder.size() - 1; ++t) {
        string t0 = "tn" + std::to_string(reorder[t]);
        string t1 = "tn" + std::to_string(reorder[t+1]);
        ir->children[cid++]->attrs["code"] = t1 + " = " + t0 + " + " + t1 + ";";

        if (t + 1 == sep) {
            ir->children[cid++]->attrs["code"] = t1 + " = std::max(" + t1 + ", 5);";
        }
    }
    ir->children[cid]->attrs["code"] = "thread_num=1<<tn" + std::to_string(reorder.back()) + ";";
    
    // Modify the tnum in kernel
    cid = 0;
    for (auto& sentence: kernel->children) {
        if (sentence->type == "define" && sentence->get_attr("lvalue").substr(0, 4) == "tnum") {
            break;
        }
        ++cid;
    }

    for (int tn = 0; tn < reorder.size(); ++tn) {
        auto& tnum_ir = kernel->children[cid++];
        string last_tn = "tn" + (tn == 0 ? std::to_string(reorder.size()): std::to_string(reorder[tn-1]));
        string now_tn = "tn" + std::to_string(reorder[tn]);
        string tnum = "tnum" + std::to_string(reorder[tn]);
        tnum_ir->attrs["lvalue"] = tnum;
        tnum_ir->attrs["rvalue"] = "1<<(" + now_tn + "-" + last_tn + ")";

        auto& tid_ir = kernel->children[cid++];
        tid_ir->attrs["lvalue"] = "tid" + std::to_string(reorder[tn]);
        tid_ir->attrs["rvalue"] = "(thread_id>>" + last_tn + ") & (" + tnum + "-1)";
    }

    ir->find_define("p1")->attrs["rvalue"] = string("std::max(thread_num / std::min(1 << tn") + std::to_string(reorder[sep]) + ", 1024), 1)";
    ir->find_define("p2")->attrs["rvalue"] = string("std::min(1 << tn") + std::to_string(reorder[sep]) + ", 1024)";
}

extern int para_opt_level;

void SharedReducePass::run() {
    auto choice = op->get_loop_option("parallel");
    auto use_shared_reduce = op->get_loop_option("use_shared_reduce", 1);
    if (use_shared_reduce == 0) return;
    if (para_opt_level < 4) return;

    bool is_cuda = op->flags.get(NodeFlags::_cuda);
    if (is_cuda) choice = 1;
    if (!choice) return;

    for (int loop_id = 0; loop_id < ir->children.size(); ++loop_id) {
        auto& loop = ir->children[loop_id];

        // Looks up the main src and the kernel function of a reduce op
        // 1.1. ir type is loop
        if (loop->type != "loop") 
            continue;

        // 1.2. a kernel is called
        auto kernel_info = look_up_kernel_function(loop, ir->before);
        if (kernel_info.second == -1)
            continue;

        string kernel_name = kernel_info.first;
        auto& kernel_ir = ir->before[kernel_info.second];

        // 1.3. reduce op is fused
        int reduce_op_id = look_up_reduce_op(kernel_ir, op);
        if (reduce_op_id < 0)
            continue;

        modify_reduce_launch(loop, kernel_ir, dynamic_cast<ReduceOp*>(op->ops[reduce_op_id]));
        modify_reduce_kernel(kernel_ir);
    }
}

} // jittor
