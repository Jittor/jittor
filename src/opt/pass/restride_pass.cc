// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "op_compiler.h"
#include "opt/pass_manager.h"
#include "opt/pass/restride_pass.h"

namespace jittor {

// find t{number} in s
int findn(const string& s, const string&t) {
    for (uint i=0; i+t.size()<=s.size(); i++) {
        bool found = true;
        for (uint j=0; j<t.size(); j++) {
            if (s[i+j] != t[j]) {
                found = false;
                break;
            }
        }
        if (found) {
            if (i+t.size()<s.size() && isdigit(s[i+t.size()]))
                continue;
            return i;
        }
    }
    return -1;
}

void RestridePass::run() {
    auto choice = op->get_loop_option("restride");
    auto pf = op->get_loop_option("restride_profile");
    if (!choice) return;
    vector<KernelIR*> q({ir});
    unordered_map<string, string> replaces;
    unordered_map<string, vector<KernelIR*>> rloops;
    unordered_map<string, string> origin_defs;
    vector<KernelIR*> defs;
    for (uint i=0; i<q.size(); i++) {
        KernelIR* ir = q[i];
        if (ir->type == "define") {
            vector<KernelIR*> loops, splits;
            KernelIR* fa = ir->father;
            // find all loop index affect this define
            while (fa && fa->type=="loop" && fa->has_attr("loop_id")) {
                auto idname = "id"+fa->attrs["loop_id"];
                if (findn(ir->attrs["rvalue"], idname) != -1)
                    loops.push_back(fa);
                fa = fa->father;
            }
            if (loops.size()) {
                string newid;
                // create new id which is continuous
                for (uint i=0; i<loops.size(); i++) {
                    newid += "+" + loops[i]->get_attr("lvalue");
                    bool found = false;
                    for (auto& split : splits) {
                        if (split->get_attr("split_id") != loops[i]->get_attr("loop_id"))
                            newid += "*"+split->get_attr("rvalue");
                        else {
                            split = loops[i];
                            found = true;
                        }
                    }
                    if (!found) splits.push_back(loops[i]);
                }
                auto& lvalue = ir->get_attr("lvalue");
                auto& rvalue = ir->get_attr("rvalue");
                if (replaces.count(lvalue) && (replaces[lvalue] != newid || origin_defs[lvalue] != rvalue)) {
                    // conflict stride, pass
                    replaces[lvalue] = "";
                } else {
                    replaces[lvalue] = newid;
                    rloops[lvalue] = loops;
                    origin_defs[lvalue] = rvalue;
                }
                defs.push_back(ir);
            }
        }
        for (auto& c : ir->children)
            q.push_back(c.get());
    }
    string total_size = "0";
    string prev_name;
    vector<string> newdefs;
    for (auto& kv : replaces) {
        if (pf) break;
        if (kv.second.size() == 0) continue;
        string name = kv.first.substr(0, kv.first.size()-2);
        uint op_id, opvar_id;
        Op* op;
        Var* var;
        pm->oc->get_op_var_by_name(name, op_id, opvar_id, op, var);
        std::stringstream ss;
        ss << var->dtype() << "* __restrict__ " << name << "_new = (" << var->dtype() << "*)";
        if (prev_name.size())
            ss << "(((char*)" << prev_name << "_new)+" + prev_name + "->size);";
        else
            ss << "&buffer[0];";
        prev_name = name;
        total_size += "+" + name + "->size";
        newdefs.push_back(ss.str());
        
        KernelIR* cir = ir;
        std::stringstream s2;
        auto& loops = rloops[kv.first];
        bool is_output = opvar_id >= op->inputs().size();
        for (int i=(int)loops.size()-1; i>=0; i--) {
            if (!is_output && i==(int)loops.size()-1) {
                cir->push_front(loops[i]->clone(false));
                cir = cir->children.front().get();
                continue;
            }
            cir->push_back(loops[i]->clone(false));
            cir = cir->children.back().get();
        }
        auto org_id = origin_defs[kv.first];
        if (is_output) {
            // this var is output
            s2 << name << "p[" << org_id << "] = " << name << "_new[" << kv.second << "];";
            cir->push_back(s2.str());
        } else {
            // this var is input
            s2 << name << "_new[" << kv.second << "] = "<< name << "p[" << org_id << "];";
            cir->push_front(s2.str());
        }
    }
    if (total_size != "0") {
        ir->push_back("auto total_size = "+total_size+";", nullptr, true);
        ir->push_back("char* __restrict__ buffer = (char*)aligned_alloc(alignment, total_size);", nullptr, true);
        for (auto& def : newdefs)
            ir->push_back(def);
        ir->move_loop_back();
        ir->push_back("::free(buffer);");
    }
    // replace prev id with new id
    for (auto ir : defs) {
        auto& lvalue = ir->attrs["lvalue"];
        auto& rvalue = ir->attrs["rvalue"];
        if (replaces.count(lvalue) && replaces[lvalue] != "") {
            string name = lvalue.substr(0, lvalue.size()-2);
            rvalue = replaces[lvalue];
            ASSERT(ir->father);
            if (!pf)
                ir->father->replace({{name+"p", name+"_new"}});
        }
    }
}

} // jittor