// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/check_cache_pass.h"
#include <regex>
#include <iostream>
#include "profiler/memory_checker.h"
using namespace std;

namespace jittor {

void CheckCachePass::run() {
    auto choice = op->get_loop_option("check_cache");

    if (!choice) return;
    
    /*
        input: 
            single simple assignment like: 
                a[x] = b[y] + c[z]; 
            or read only like:
                f(a[x]);
        output: (read_addr_list, write_addr_list)
     */
    auto get_read_write_address = [&](string code) -> pair<vector<string>, vector<string>> {
        vector<string> assignment_list({"[^><=!]=[^=]", "\\+=", "-=", "\\*=", "/=", "%=", ">>=", "<<=", "&=", "\\|=", "\\^="});
        vector<string> read_addr, write_addr;
        string pa;
        for (int i = 0; i < (int)assignment_list.size(); ++i) {
            if (i > 0) {
                pa += "|";
            }
            pa += assignment_list[i];
        }
        regex pattern(pa);
        int assignment_cnt = 0, assignment_pos = -1;
        string temp_code = code;
        smatch m;
        while (regex_search(temp_code, m, pattern)) {
            assignment_pos = m.position(0);
            ++assignment_cnt;
            temp_code = m.suffix().str();
        }
        ASSERT(assignment_cnt <= 1); // for simple assignment only
        vector<int> address_pos;
        for (int i = 0; i < (int)code.length(); ++i) {
            if (code[i] == '[') {
                address_pos.push_back(i);
            }
            if (code[i] == ']') {
                int sp = address_pos.back() - 1;
                // don't check cache of shape[...]
                if (sp>=4 && code.substr(sp-4, 5) == "shape") {
                    address_pos.pop_back();
                    continue;
                }
                while (sp >= 0 && ((code[sp] >= 'A' && code[sp] <= 'Z') || (code[sp] >= 'a' && code[sp] <= 'z') || 
                (code[sp] >= '0' && code[sp] <= '9') || code[sp] == '_' || code[sp] == '.' || (sp > 0 && code[sp] == '>' && code[sp - 1] == '-'))) {
                    if (sp > 0 && code[sp] == '>' && code[sp - 1] == '-')
                        sp -= 2;
                    else
                        --sp;
                }
                ++sp;
                string s = "(size_t)&(" + code.substr(sp, i - sp + 1) + ")";
                if (i <= assignment_pos)
                    write_addr.push_back(s);
                else
                    read_addr.push_back(s);
                address_pos.pop_back();
            }
        }
        return make_pair(read_addr, write_addr);
    };
    size_t  page_size = op->get_loop_option("page_size"), vtop = op->get_loop_option("vtop"),
            tlb_size = op->get_loop_option("tlb_size"), tlb_ways = op->get_loop_option("tlb_ways"), tlb_line_size = op->get_loop_option("tlb_line_size"),
            L1_size = op->get_loop_option("L1_size"), L1_ways = op->get_loop_option("L1_ways"), L1_line_size = op->get_loop_option("L1_line_size"),
            L2_size = op->get_loop_option("L2_size"), L2_ways = op->get_loop_option("L2_ways"), L2_line_size = op->get_loop_option("L2_line_size"),
            L3_size = op->get_loop_option("L3_size"), L3_ways = op->get_loop_option("L3_ways"), L3_line_size = op->get_loop_option("L3_line_size");

    ir->push_back("#include \"profiler/memory_checker.h\"", &ir->before);
    ir->push_back("using namespace jittor;", &ir->before);
    // declaration
    ir->push_back("extern \"C\" std::unique_ptr<MemoryChecker> memory_checker;", &ir->before);
    // definition
    ir->push_back("std::unique_ptr<MemoryChecker> memory_checker;", &ir->before);
    vector<string> commands;
    stringstream command;
    string replace_strategy = MemoryChecker::get_replace_strategy(op->get_loop_option("replace_strategy"));
    
    command << "Cache* tlb = new " << replace_strategy << "(CacheConfig(" << tlb_size << "," << tlb_ways << "," << tlb_line_size << "));";
    commands.push_back(command.str());
    command.str("");
    command << "Cache* L1 = new " << replace_strategy << "(CacheConfig(" << L1_size << "," << L1_ways << "," << L1_line_size << "));";
    commands.push_back(command.str());
    command.str("");
    command << "Cache* L2 = new " << replace_strategy << "(CacheConfig(" << L2_size << "," << L2_ways << "," << L2_line_size << "));";
    commands.push_back(command.str());
    command.str("");
    command << "Cache* L3 = new " << replace_strategy << "(CacheConfig(" << L3_size << "," << L3_ways << "," << L3_line_size << "));";
    commands.push_back(command.str());
    command.str("");
    commands.push_back("vector<Cache*> caches({L1, L2, L3});");
    commands.push_back("memory_checker.reset(new MemoryChecker(tlb, caches, "+S(page_size)+","+S(vtop)+"));");
    
    while (commands.size()) {
        ir->push_front(commands.back(), &ir->children, true);
        commands.pop_back();
    }
    vector<KernelIR*> q({ir});
    vector<string> attrs_to_check{"code", "rvalue"};
    for (uint i=0; i<q.size(); i++) {
        KernelIR* ir = q[i];
        ir->for_each([&](unique_ptr<KernelIR>& c) {
            q.push_back(c.get());
        });

        vector<string> codes_to_check;
        for (auto& attr : attrs_to_check) {
            if (!ir->has_attr(attr)) continue;
            auto& code = ir->attrs[attr];
            codes_to_check.push_back(code);
        }
        for (int j = 0; j < (int)ir->inner.size(); ++j) {
            codes_to_check.push_back(ir->inner[j]->to_string());
        }
        for (int j = 0; j < (int)codes_to_check.size(); ++j) {
            string code = codes_to_check[j];
            pair<vector<string>, vector<string>> rw_list = get_read_write_address(code);
            for (int k = 0; k < (int)rw_list.first.size(); ++k) {
                string addr = rw_list.first[k];
                ir->push_back("memory_checker->check_hit(" + addr + ");", &ir->before);
            }
            for (int k = 0; k < (int)rw_list.second.size(); ++k) {
                string addr = rw_list.second[k];
                ir->push_back("memory_checker->check_hit(" + addr + ");", &ir->before);
            }
        }
    }
    //ir->push_back("memory_checker->print_miss();", &ir->children);
}

} // jittor