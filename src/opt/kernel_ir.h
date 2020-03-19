// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "misc/str_utils.h"

namespace jittor {

struct KernelIR {
    // if type == define
    //     src is dtype lvalue = rvalue;
    // if type == loop
    //     src is for (inner[0]; inner[1]; inner[2])
    // if type == if
    //     src is if (inner[0])
    // if type == func
    //     src is dtype lvalue(*inner)
    // else
    //     src is code
    string type;
    KernelIR* father=nullptr;
    vector<unique_ptr<KernelIR>>* flist;
    // available attrs:
    // * code: used in macro, comment type
    // * dtype: used in define, func type
    // * lvalue: used in define, func type
    // * rvalue: used in define type
    // * loop_id: generate by rename_loop_index
    // * split_id: generate by split_loop
    // * used: generate by check_unused
    // * rvalue2: generate by split_loop, used in loop type, represent stride
    unordered_map<string,string> attrs;
    // before...
    // src {
    //     rest of inner...
    //     children...
    // }
    // after...
    vector<unique_ptr<KernelIR>> before, inner, children, after;
    unordered_map<string,list<KernelIR*>> scope;
    
    KernelIR() {}
    KernelIR(const string& src, bool raw=false);
    string& get_attr(const string& s);
    bool has_attr(const string& s);
    bool check_attr(const string& k, const string& v);

    // src: source for kernel ir
    // irp: kernel ir
    // raw: raw code or not
    // ls: can be [before, inner, children, after], default is children
    void push_back(const string& src, vector<unique_ptr<KernelIR>>* ls=nullptr, bool raw=false);
    void push_front(const string& src, vector<unique_ptr<KernelIR>>* ls=nullptr, bool raw=false);
    void push_back(unique_ptr<KernelIR>&& irp, vector<unique_ptr<KernelIR>>* ls=nullptr);
    void push_front(unique_ptr<KernelIR>&& irp, vector<unique_ptr<KernelIR>>* ls=nullptr);

    // insert kernel ir (src) to pos
    void insert(uint pos, const string& src, bool raw=false);
    // insert kernel irs to pos
    void insert(uint pos, vector<unique_ptr<KernelIR>>& irs);
    // recursive clone kernel ir(type, attrs, before, inner, after)
    // with_children: recursively clone children or not
    unique_ptr<KernelIR> clone(bool with_children=true);
    // swap two loop
    // with_children: swap children or not
    void swap(KernelIR& other, bool with_children=false);

    // add into parent scope
    void add_scope();
    // delete in parent scope
    void del_scope();
    // clear and reconstruct scope
    void rebuild_scope();
    // self destroy
    void erase();
    // clear self(attr, inner), preserve children, before, after
    void clear();
    // move out self from father
    unique_ptr<KernelIR> move_out();
    // move out all children
    vector<unique_ptr<KernelIR>> move_out_children();

    // try pase define statement from string s
    // if failed, fail back to a normal code
    void try_parse_define(const string& s);

    // parse syntax "for (dtype lvalue = 0; lvalue<rvalue; lvalue++)"
    // raw: do not parse attrs(e.g. dtype, lvalue, rvalue) if true
    void parse_for_loop(const string& s, bool raw=false);
    // recursively find loops by loop id
    // lid: loop id string
    vector<KernelIR*> find_loops(string lid);
    // find definition from ancestors
    KernelIR* find_define(const string& name);
    // for each sub ir, include before, inner, children, after
    template <typename Func>
    void for_each(Func&& func) {
        vector<unique_ptr<KernelIR>>* ls[] = {&before, &inner, &children, &after};
        for (auto& l : ls) {
            for (auto& c : (*l))
                func(c);
        }
    }
    template <typename Func>
    void dfs(Func&& func) {
        vector<unique_ptr<KernelIR>>* ls[] = {&before, &inner, &children, &after};
        for (auto& l : ls) {
            for (auto& c : (*l)) {
                func(c);
                c->dfs(func);
            }
        }
    }
    // for each sub ir backward, include before, inner, children, after
    template <typename Func> void for_each_rev(Func&& func);
    // update sub irs' father to itself
    void update_father();
    
    // recursively to_string
    // level: indent level
    // debug: output type, attrs, scope in comments, check father
    string to_string(int level=0, bool debug=false);

    // move all loop back(exclude raw loop)
    void move_loop_back();

    // replace vars
    // replace_vars: pairs of string, e.g. [(a,b), (x,y)] replace a->b, x->y
    // equal: if true, replace vars need to match completely, if false, replace vars can match the prefix
    // remove_define: if a definition statement matched, remove or not
    //     would't remove if inside a loop or is in inner list
    void replace(const vector<pair<string,string>>& replace_vars, bool equal=false, bool remove_define=true);

    // recursively rename loop var by loop id, loop_id is parsed from rvalue
    // for (dtype i=lvalue; i<range{loop_id}; i++)
    //    -->
    // for (dtype id{loop_id}; id{loop_id}<range{loop_id}; id{loop_id}++)
    void rename_loop_index();

    // recursively merge loops' children with the same attr[loop_id]
    // for (...) s1;
    // for (...) s2;
    //     -->
    // for (...) { s1; s2; }
    void merge_loop();

    // recursively expand block if no attr[loop_id] and attr[rvalue]
    // { s1; s2; }
    //      -->
    //  s1; s2;
    void expand_empty_block();

    // recursively resolve conlict definitions
    // T a = 1; a++; T a = 2; a++;
    //     -->
    // T a = 1; a++; T _a = 2; _a++;
    void solve_conflict_define();

    // TODO: move to pass
    // remove intermediate variables in names
    // xxxp[...] -> xxxd
    // xxxd = xxx -> auto xxxd = xxx
    // xxxp -> 0
    void remove_intermediate(const unordered_set<string>& names);

    // remove definitions which attr[used]==1, return remove or not
    bool remove_unused();

    // remove all unused definitions, until no unused definition occurs.
    void remove_all_unused();

    // recursively generate attr[used]
    void check_unused();
    // recursively find used
    void find_used(KernelIR* def, vector<KernelIR*>& used);
    vector<KernelIR*> find_used();

    // split loop(loop_id=i) into two loop
    // for (T id{i}; id{i}<range{i}; id{i}++)
    //     stmt( id{i} )
    //     -->
    // for (T id{i}; id{i}<range{i}; id{i}+=stride{i})
    //     T range{j} = min(stride{i}, range{i}-id{i});
    //     for (T id{j}; id{j}<range{j}; id{j}++)
    //         stmt( (id{i}+id{j}) )
    void split_loop(int i, int j);

    // get const number of definition by name
    // name: name of definition
    // num: const number of definition, 
    //     return -1 if no such definition,
    //     return -2 if definition is not a constant
    // return: has const definition or not
    bool get_number(const string& name, int& num);

    // resplit loop from:
    // for (int i0=0; i0<range0; i0+=stride)
    //     int range1 = min(range0-i, stride)
    //    ...
    // to
    // int i0=0;
    // for (i0=0; i0+stride<=range0; i0+=stride)
    //     int range1 = stride
    //     ...
    // if (i0 < range0)
    //     int range1 = range0 - i0
    //     ...
    //
    void resplit();
    
    // sanity check each children->father == self
    void check_father();
};

std::ostream& operator<<(std::ostream& os, KernelIR& ir);

// match aaa::bbb
bool isvar(char x);

// match x[y]
bool isvarp(char x);

// remove arg_i-th arguments of func_call
// src: func_call source
// arg_i: arguments id
void remove_func_call_arg(string& src, int arg_i);

} // jittor
