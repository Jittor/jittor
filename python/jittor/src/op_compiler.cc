// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <regex>
#include <algorithm>
#include "op.h"
#include "fused_op.h"
#include "op_compiler.h"
#include "jit_compiler.h"
#include "utils/cache_compile.h"
#include "opt/tuner_manager.h"
#include "misc/str_utils.h"
#include "ops/op_register.h"
#include "ops/array_op.h"
#include "lock.h"
#include "opt/expr.h"
#include "pyjt/py_caller.h"

namespace jittor {

DECLARE_FLAG(string, jittor_path);

using namespace jit_compiler;

static bool isvar(char x) { return isalnum(x) || x == '_' || x == ':'; }

void OpCompiler::get_op_var_by_name(const string& name, uint& op_id, uint& opvar_id, Op*& op, Var*& var) {
    // name: op{id}_{varname}
    ASSERT(name.size()>3 && name[0]=='o' && name[1]=='p');
    uint j=2;
    while (j<name.size() && isdigit(name[j])) j++;
    ASSERT(j>2);
    op_id = std::stoi(name.substr(2, j-2));
    ASSERT(op_members.size() > op_id);
    bool found = false;
    for (opvar_id=0 ;opvar_id < op_members[op_id].size(); opvar_id++) {
        if (op_members[op_id][opvar_id] == name) {
            found = true;
            break;
        }
    }
    op = this->op->ops[op_id];
    ASSERT(found && opvar_id < op->inputs().size() + op->outputs().size());
    if (opvar_id >= op->inputs().size()) {
        auto iter = op->outputs().begin();
        for (uint t=op->inputs().size(); t<opvar_id; t++)
            iter++;
        var = *iter;
    } else {
        auto iter = op->inputs().begin();
        for (uint t=0; t<opvar_id; t++)
            iter++;
        var = *iter;
    }
}

string OpCompiler::get_name_by_op_var(Op* op, Var* var) {
    uint var_id=0;
    bool found = 0;
    for (Var* i : op->inputs()) {
        if (i==var) { 
            found = 1;
            break;
        }
        var_id++;
    }
    if (!found)
        for (Var* o : op->outputs()) {
            if (o==var) { 
                found = 1;
                break;
            }
            var_id++;
        }
    ASSERT(found);
    ASSERT(this->op);
    ASSERT(this->op->context);
    auto opid = this->op->context->node_id.at(op);
    ASSERT(opid<(int)op_members.size());
    auto& v = op_members[opid];
    ASSERT(var_id < v.size());
    return v[var_id];
}

string OpCompiler::get_name_by_op_input(Op* op, uint i) {
    return op_members.at(this->op->get_node_id(op)).at(i);
}

string OpCompiler::get_name_by_op_output(Op* op, uint i) {
    return op_members.at(this->op->get_node_id(op)).at(i+op->inputs().size());
}

bool OpCompiler::op_exist(Op* op) {
    return op_members.at(this->op->get_node_id(op)).size();
}

int OpCompiler::total_member_count() {
    int member_count=0;
    int i = 0;
    for (auto& v : op_members) {
        // array need a extra local var
        if (op->ops[i]->name()==string("array"))
            member_count += 1;
        member_count += v.size();
        i += 1;
    }
    return member_count;
}

int64_t OpCompiler::eval(const string& expr, const unordered_map<string,string>& vars) {
    if (expr.find("@") != string::npos) {
        string new_expr;
        for (size_t i=0; i<expr.size(); i++) {
            if (expr[i] != '@') new_expr += expr[i];
            else {
                size_t j=i+1;
                ASSERT(j < expr.size());
                // syntax @{...}
                //        ij    k
                if (expr[j] == '{') {
                    size_t k=j+1;
                    int presum = 1;
                    while (k<expr.size() && presum) {
                        if (expr[k] == '}')
                            presum--;
                        else if (expr[k] == '{')
                            presum++;
                        k++;
                    }
                    CHECK(presum==0) << "Jit error: braces are not matched.";
                    new_expr += S(eval(expr.substr(j+1, k-j-2), vars));
                    i = k-1;
                    continue;
                } else {
                    if (expr[j] == '@') {
                        // syntax @@
                        i = j;
                        continue;
                    }
                    // syntax: @x
                    CHECK(isvar(expr[j])) << expr[j] << "is not var";
                    size_t k=j+1;
                    while (k<expr.size() && isvar(expr[k])) k++;
                    if (k<expr.size() && expr[k]=='(') {
                        // syntax @xx(...)
                        //        ij k    l
                        size_t l=k+1;
                        int presum = 1;
                        while (l<expr.size() && presum) {
                            if (expr[l] == ')')
                                presum--;
                            else if (expr[l] == '(')
                                presum++;
                            l++;
                        }
                        new_expr += precompile(vars, expr.substr(i, l-i));
                        i = l-1;
                        continue;
                    }
                    string var = expr.substr(j, k-j);
                    auto iter = vars.find(var);
                    ASSERT(iter!=vars.end()) << "Jit var " << var << " not found." << vars;
                    new_expr += iter->second;
                    i = k-1;
                }
            }
        }
        return eval(new_expr, vars);
    }
    auto e = expr::make(expr);
    e->dfs([&](expr::Expr* s) {
        if (s->is_sym()) {
            auto iter = vars.find(s->str);
            ASSERT(iter!=vars.end()) << "Jit var " << s->str << " not found.";
            auto e = expr::make(iter->second);
            s->swap(e.get());
        }
    });
    e = e->eval();
    ASSERT(e->is(expr::_int));
    return e->as_int();
}

void load_macros(const string& src, unordered_map<string,string>& macros) {
    LOGvvvv << "load_macros" << src;
    for (size_t i=0; i<src.size(); i++) {
        if (src[i] == '#') {
            // #define xxx(...)  xxx
            // i      jk  r    l p  q
            auto j=i+1;
            while (j<src.size() && src[j] != ' ') j++;
            if (j-i!=7 || src.substr(i,j-i) != "#define") {
                i = j;
                continue;
            }
            ASSERT(j<src.size());
            auto k=j+1;
            while (k<src.size() && src[k] == ' ') k++;
            ASSERT(k<src.size());
            auto l=k+1;
            while (l<src.size() && (src[l] != '\n' && src[l-1] != ')')) l++;
            auto p=l;
            while (p<src.size() && (src[p] == ' ')) p++;
            auto q=p;
            while (q<src.size() && (src[q] != '\n')) q++;
            // TODO: multiline macro
            auto r=k;
            while (r<l && src[r] != '(') r++;
            auto body = q>p ? src.substr(p,q-p) : "";
            auto args = "<"+ (r+1<l?src.substr(r+1,l-r-2):"") + ">";
            // header <args>body
            body = args + body;
            auto header = src.substr(k,r-k);
            LOGvvvv << "header:" << header << "body:" << body;
            macros[header] = body;
            i = q;
        }
    }
}

void expand_macro(const string& macro, const vector<string>& args, string& new_src) {
    LOGvvvv << "expand_macro" << macro << "args:" << args;
    if (macro.size() == 0 || macro[0] != '<') {
        new_src += macro;
        return;
    }
    auto i = macro.find(">");
    ASSERT(i != string::npos);
    // <a1, a2, ...>body
    //  j k        i
    unordered_map<string, int> args_map;
    for (uint j=1, l=0; j<i; l++) {
        uint k=j;
        while (k<i && macro[k] != ',') k++;
        args_map[macro.substr(j,k-j)] = l;
        j = k+1;
        while (j<i && macro[j] == ' ') j++;
    }
    ASSERTop(args.size(),==,args_map.size()) << "Number of macro args not match.";
    for (i=i+1; i<macro.size(); i++) {
        if (isvar(macro[i])) {
            uint j = i+1;
            while (j<macro.size() && isvar(macro[j])) j++;
            string var = macro.substr(i, j-i);
            auto iter = args_map.find(var);
            if (iter == args_map.end()) {
                new_src += var;
            } else {
                new_src += args[iter->second];
            }
            i = j-1;
            continue;
        }
        new_src += macro[i];
    }
}

string precompile(unordered_map<string,string> defs, string src, unordered_map<string, string>& macros) {
    string new_src;
    new_src.reserve(src.size());
    // dirty fix windows \r\n change line
    for (auto& c : src)
        if (c == '\r') c = '\n';
    for (size_t i=0; i<src.size(); i++) {
        try{
        if (src[i] == '/' && (i+1<src.size() && src[i+1] == '/')) {
            size_t j=i+1;
            while (j<src.size() && src[j] != '\n') j++;
            if (j<src.size()) j++;
            // remove comment
            // for (size_t k=i; k<j; k++) new_src += src[k];
            if (src[j-1]=='\n')
                new_src += '\n';
            i = j-1;
            continue;
        } else
        if (src[i] == '/' && (i+1<src.size() && src[i+1] == '*')) {
            size_t j=i+1;
            while (j<src.size() && !(src[j] == '/' && src[j-1] == '*')) j++;
            if (j<src.size()) j++;
            // remove comment
            // for (size_t k=i; k<j; k++) new_src += src[k];
            i = j-1;
            continue;
        } else
        if (src[i] == '#') {
            // #include "a.h"
            // i       jk    l
            // #define xxx
            // i      jk  l
            auto j=i+1;
            while (j<src.size() && src[j] != ' ') j++;
            ASSERT(j<src.size());
            auto k=j+1;
            while (k<src.size() && src[k] == ' ') k++;
            ASSERT(k<src.size());
            auto l=k+1;
            while (l<src.size() && (src[l] != '\n')) l++;
            if (src[k] == '"' && src[l-1] == '"' && j-i==8 && src.substr(i,j-i) == "#include") {
                auto inc = src.substr(k+1, l-k-2);
                if (inc.size()>=6 && inc.substr(inc.size()-6) == "defs.h") {
                    LOGvvvv << "Found defs include" << inc;
                    auto src_path = join(jittor_path, "src");
                    src_path = join(src_path, inc);
                    auto inc_src = read_all(src_path);
                    // load_macros from include src
                    precompile(defs, inc_src, macros);
                    // we do not include defs.h
                    i = l;
                    continue;
                }
            } else
            if (j-i==7 && src.substr(i,j-i) == "#define") {
                load_macros(src.substr(i,l-i), macros);
            } else
            // #ifdef JITxxx
            // #else
            // #endif
            if (((j-i==6 && src.substr(i,j-i) == "#ifdef") ||
                (j-i==7 && src.substr(i,j-i) == "#ifndef")) && startswith(src, "JIT", k)) {
                bool is_ifndef = j-i==7;
                string key = src.substr(k, l-k);
                // find pair #endif and #else
                int presum = 1;
                size_t prev = l+1, ii = prev;
                string block, else_block;
                while (ii < src.size()) {
                    if (startswith(src, "#if", ii)) {
                        presum++;
                        ii += 3;
                        continue;
                    }
                    if (startswith(src, "#else", ii)) {
                        auto next_ii = ii+5;
                        // remove ' ' or '\n' after #else
                        if (next_ii<src.size() && (src[next_ii]==' ' || src[next_ii]=='\n'))
                            next_ii++;
                        if (presum==1) {
                            block = src.substr(prev, ii-prev);
                            prev = next_ii;
                        }
                        ii = next_ii;
                        continue;
                    }
                    if (startswith(src, "#endif", ii)) {
                        presum--;
                        auto next_ii = ii+6;
                        // remove ' ' or '\n' after #endif
                        if (next_ii<src.size() && (src[next_ii]==' ' || src[next_ii]=='\n'))
                            next_ii++;
                        if (presum==0) {
                            if (prev == l+1)
                                block = src.substr(prev, ii-prev);
                            else
                                else_block = src.substr(prev, ii-prev);
                            ii = next_ii;
                            break;
                        }
                        ii = next_ii;
                        continue;
                    }
                    ii++;
                }
                ASSERT(presum==0);
                if (is_ifndef) block.swap(else_block);
                if (defs.count(key) || macros.count(key)) {
                    new_src += precompile(defs, block, macros);
                } else {
                    new_src += precompile(defs, else_block, macros);
                }
                i = ii-1;
                continue;
            }
            for (auto k=i; k<l; k++) new_src += src[k];
            i= l-1;
            continue;
        } else
        if (src[i] == '@' && i+1<src.size()) {
            size_t j=i+1;
            // syntax @{...}
            //        ij    k
            if (src[j] == '{') {
                size_t k=j+1;
                int presum = 1;
                while (k<src.size() && presum) {
                    if (src[k] == '}')
                        presum--;
                    else if (src[k] == '{')
                        presum++;
                    k++;
                }
                CHECK(presum==0) << "Jit error: braces are not matched.";
                new_src += S(OpCompiler::eval(src.substr(j+1, k-j-2), defs));
                i = k-1;
                continue;
            } else if (src[j] == '(') {
            // syntax @(...)
            //        ij    k
                size_t k=j+1;
                int presum = 1;
                while (k<src.size() && presum) {
                    if (src[k] == ')')
                        presum--;
                    else if (src[k] == '(')
                        presum++;
                    k++;
                }
                CHECK(presum==0) << "Jit error: braces are not matched.";
                new_src += precompile(defs, src.substr(j+1, k-j-2), macros);
                i = k-1;
                continue;
            } else if (isvar(src[j])) {
                size_t k=j+1;
                while (k<src.size() && isvar(src[k])) k++;
                string expr = src.substr(j, k-j);
                // syntax for @python.module.function(args)
                if (expr == "python") {
                    while (k<src.size() && (isvar(src[k]) || src[k]=='.' )) k++;
                    string full_expr = src.substr(j, k-j);
                }
                int presum = 1;
                vector<int> comma;
                vector<string> args;
                size_t l = k+1;
                if (expr == "for" || expr == "if" || expr == "expand_macro" ||
                    expr == "is_def" || expr == "python" ||
                    (k<src.size() && src[k]=='(')) {
                    ASSERT(src[k] == '(');
                    comma.push_back(k);
                    while (l<src.size() && presum) {
                        if (src[l] == ')')
                            presum--;
                        else if (src[l] == '(')
                            presum++;
                        else if (presum == 1 && src[l] == ',')
                            comma.push_back(l);
                        l++;
                    }
                    CHECK(presum==0) << "Jit error: braces are not matched.";
                    comma.push_back(l-1);
                    for (uint i=0; i+1<comma.size(); i++)
                        args.push_back(src.substr(comma[i]+1, comma[i+1]-comma[i]-1));
                }
                if (expr == "python") {
                    string full_expr = src.substr(j, k-j);
                    LOGvvv << "python call" << full_expr << args;
                    int presum = 0;
                    auto ll = l;
                    while (l<src.size()) {
                        if (src[l] == '{')
                            presum++;
                        else if (src[l] == '}')
                            presum--;
                        if (presum==0 && (src[l] == '}' || src[l] == ';'))
                            break;
                        l++;
                    }
                    CHECK(l<src.size()) << "Jit error: braces are not matched.";
                    auto full_src = src.substr(ll, l+1-ll);
                    i = l;
                    full_src = py_caller(
                        full_expr.substr(7),
                        args, {{"src",full_src}}
                    );
                    new_src += precompile(defs, full_src, macros);
                    continue;
                }
                // syntax @for(i, l, r, ...)
                //        ij  k             l
                if (expr == "for") {
                    CHECKop(args.size(),>=,4u) << "Jit error: for missing arguments.";
                    string vi = args[0];
                    string vl = args[1];
                    string vr = args[2];
                    string vs = args[3];
                    auto vil = OpCompiler::eval(vl, defs);
                    auto vir = OpCompiler::eval(vr, defs);
                    int step = 1;
                    if (args.size() >= 5) {
                        step = OpCompiler::eval(vs, defs);
                        vs = args[4];
                        for (int i=5; i<args.size(); i++) {
                            vs += "," + args[i];
                        }
                    }
                    auto new_defs = defs;
                    LOGvvv << "Expand for" << expr >> "[" >> vil >> "," >> vir >> "," >> step >> "]";
                    int total_step = 0;
                    for (auto vii=vil; vii!=vir; vii+=step) {
                        total_step ++;
                        ASSERT(total_step < 1000) << "Too much step.";
                        new_defs[vi] = S(vii);
                        new_src += precompile(new_defs, vs, macros);
                    }
                    i = l-1;
                    continue;
                } else
                if (expr == "if") {
                    // syntax: @if(cond, true[, false])
                    //         ij k                    l
                    ASSERT(args.size()>=2u && args.size()<=3u)
                        << "Jit error: if wrong arguments.";
                    string vcond = args[0];
                    string vtrue = args[1];
                    string vfalse = args.size() == 3u ? args[2] : "";
                    int cond = OpCompiler::eval(vcond, defs);
                    new_src += precompile(defs, cond?vtrue:vfalse, macros);
                    i = l-1;
                    continue;
                } else
                if (expr == "is_def") {
                    ASSERT(args.size()==1)
                        << "Jit error: is_def wrong arguments.";
                    string vdef = args[0];
                    vdef = precompile(defs, vdef, macros);
                    if (defs.count(vdef) || macros.count(vdef))
                        new_src += "1";
                    else
                        new_src += "0";
                    i = l-1;
                    continue;
                } else
                if (expr == "expand_macro") {
                    // syntax: @expand_macro(macro, args)
                    //         ij           k            l
                    for (auto& arg : args) {
                        uint p=0;
                        while (p<arg.size() && arg[p] == ' ') p++;
                        arg = precompile(defs, arg.substr(p), macros);
                    }
                    string vmacro = args[0];
                    args.erase(args.begin());
                    auto iter = macros.find(vmacro);
                    string ns;
                    if (iter == macros.end()) {
                        if (defs.count(vmacro))
                            ns = defs[vmacro];
                        else
                            LOGf << "Macro" << vmacro << "not found.";
                    } else {
                        expand_macro(iter->second, args, ns);
                    }
                    new_src += precompile(defs, ns, macros);
                    i = l-1;
                    continue;
                } else
                if (expr == "define") {
                    // syntax: @define(macro, value)
                    //         ij     k             l
                    ASSERT(args.size()>=1u)
                        << "Jit error: define wrong arguments.";
                    new_src += "#define ";
                    auto key = precompile(defs, args[0], macros);
                    string value, src;
                    new_src += key;
                    if (args.size()>=2) {
                        new_src += " ";
                        string all_args = args[1];
                        for (int i=2; i<args.size(); i++) {
                            all_args += ',';
                            all_args += args[i];
                        }
                        src = precompile(defs, all_args, macros);
                        for (auto c : src) {
                            if (c == '\n')
                                value += " \\";
                            value += c;
                        }
                        new_src += value;
                    }
                    ASSERT(macros.count(key)==0) << "Macro" << key << "redefined.";
                    defs[key] = src;
                    macros[key] = value;
                    i = l-1;
                    continue;
                } else
                if (expr == "strcmp") {
                    // syntax: @strcmp(s1,s2)
                    //         ij     k      l
                    CHECK(args.size()==2u)
                        << "Jit error: strcmp wrong arguments.";
                    auto s1 = precompile(defs, args[0], macros);
                    auto s2 = precompile(defs, args[1], macros);
                    if (s1<s2) new_src += "-1"; else
                    if (s1==s2) new_src += "0"; else
                        new_src += "1";
                    i = l-1;
                    continue;
                } else
                if (expr == "alias") {
                    // syntax: @alias(s1,s2)
                    //         ij    k     l

                    // alias(a,b)
                    // a->b
                    // a_type->b_type
                    // a_dim -> b_dim
                    // for i in a_dim:
                    //   a_shapei -> b_shapei
                    //   a_stridei -> b_stridei
                    CHECK(args.size()==2u)
                        << "Jit error: alias wrong arguments.";
                    auto key = strip(precompile(defs, args[0], macros));
                    auto value = strip(precompile(defs, args[1], macros));
                    CHECK(defs.count(value+"_dim")) << '"' >> value >> '"' << "not exsit";
                    int dim = std::stoi(defs.at(value+"_dim"));
                    vector<string> keys = {"", "p", "dim", "type"};
                    for (int i=0; i<dim; i++) {
                        keys.push_back("stride"+S(i));
                        keys.push_back("shape"+S(i));
                    }
                    new_src += '\n';
                    for (auto& s : keys) {
                        string from = value+"_"+s;
                        string to = key+"_"+s;
                        if (!s.size()) {
                            from = value;
                            to = key;
                        }
                        if (defs.count(from))
                            from = defs.at(from);
                        else if (macros.count(from))
                            from = macros.at(from);
                        defs[to] = from;
                        macros[to] = from;
                        new_src += "#define "+to+" "+from+"\n";
                    }
                    i = l-1;
                    continue;
                } else
                if (args.size()) {
                    // syntax: @e0(i0,i1,...,in) -> e0_p[i0*e0_stride0+i1*e0_stride1+...]
                    ASSERT(expr.size());

                    int nid=(int)expr.size();
                    while (nid && isdigit(expr[nid-1])) nid--;
                    string prefix = expr.substr(0, nid);
                    string suffix = expr.substr(nid);
                    string dim;
                    if (expr == "x" && defs.count("XDIM")) {
                        dim = "XDIM";
                        prefix = "x";
                    } else
                    if (prefix == "e") {
                        // TODO: unify interface
                        prefix = "extras" + suffix;
                        dim = "EDIM" + suffix;
                    } else {
                        prefix = expr+"_";
                        dim = prefix + "dim";
                    }
                    CHECK(macros.count(dim)) << expr << "not exsit" << macros;
                    CHECKop(macros.at(dim),==,S(args.size())) << expr << "dimension not matched";
                    std::stringstream ss;
                    ss << prefix << "p[";
                    for (uint ii=0; ii<args.size(); ii++) {
                        string arg = precompile(defs, args[ii], macros);
                        if (ii) ss << "+";
                        ss << '(' << arg << ")*" << prefix << "stride" << ii;
                    }
                    ss << ']';
                    new_src += ss.str();
                    i = l-1;
                    continue;
                }
                // syntax: @x
                auto iter = defs.find(expr);
                ASSERT(iter!=defs.end()) << "Jit var " << expr << " not found.";
                new_src += precompile(defs, iter->second, macros);
                i = k-1;
                continue;
            } else if (src[j]=='@') {
                // seperater syntex: @@
                i++;
                continue;
            } else
                LOGf << "Jit error: Invalid syntax.";
        } else
            new_src += src[i];
        } catch (std::exception& e) {
            int il = i, ir = i;
            while (il>0 && src[il-1] != '\n') il--;
            while (ir+1<src.size() && src[ir+1] != '\n') ir++;
            string this_line = src.substr(il, ir-il+1);
            LOGf << e.what() >> "\nJit compiler error:\n" >> this_line;
        }
    }
    return new_src;
}

string OpCompiler::precompile(const unordered_map<string,string>& defs, const string& src) {
    unordered_map<string, string> macros = defs;
    return jittor::precompile(defs, src, macros);
}

string OpCompiler::get_jit_src(Op* op) {
    string name = op->name();
    string name2 = Op::op_name_to_file_name(name);
    string name3 = Op::file_name_to_class_name(name2);
    if (name == "fused") {
        string src = get_fused_src((FusedOp*)op);
        ASSERT(src.size());
        return src;
    }
    auto op_info = get_op_info(name);
    auto& src_path = op_info.source_path;
    
    string begin_src = "", end_src = "";
    // source that need to be added after the last #include statement
    string after_include_src = "";
    auto jit_define = op->get_jit_define();
    for (auto &t : jit_define) {
        // don't add CODE in define
        // this allowed comment exsit in CODE
        if (t.first == "CODE" || t.first == "HEADER")
            continue;
        string src = "#define " + t.first + " ";
        for (char c : t.second) {
            if (c=='\n') src += '\\';
            src += c;
        }
        src += '\n';
        if (startswith(t.first, "JIT"))
            begin_src += src;
        else
            after_include_src += src;
    }
    ASSERT(file_exist(src_path)) << src_path;
    LOGvvv << "Read from" << src_path; 
    string src = read_all(src_path);
    ASSERT(src.size()) << "Source read failed:" << src_path;

    unordered_map<string,string> defs(jit_define.begin(), jit_define.end());
    LOGvvv << "Precompile with key:" << defs;
    src = precompile(defs, src);

    // find the last occur of #include "..."\n
    auto pos = src.rfind("#include");
    if (pos == string::npos) pos=0;
    else {
        // find \n
        pos = src.find("\n", pos);
        if (pos == string::npos)
            pos = src.size();
        else
            pos++;
    }
    
    string new_src = begin_src + src.substr(0, pos) +
        after_include_src + src.substr(pos) + "\n" + end_src;
    return new_src;
}

string OpCompiler::get_fused_src(FusedOp* op) {
    vector<string> op_srcs;
    vector<bool> relay_switch(op->context->vrm.relay_groups.size());
    for (uint i=0; i<relay_switch.size(); i++) {
        auto relay_key = "relay"+S(i);
        if (op->loop_options->count(relay_key) &&
            op->loop_options->at(relay_key) == 1)
            relay_switch[i] = 1;
    }
    auto relay_source = op->context->vrm.get_op_relay_info(relay_switch);
    std::set<pair<int,int>> relayed;
    for (uint oi=0; oi<op->ops.size(); oi++) {
        // relay group id, pair id
        auto p = relay_source[oi];
        if (p.first != -1) {
            if (relayed.count(p)) {
                op_srcs.push_back("");
                continue;
            }
            relayed.insert(p);
            auto src = op->context->vrm.get_relay_src(p.first, p.second);
            op_srcs.push_back(src);
            // op_srcs.push_back(get_relayed_src(src));
            continue;
        }
        Op* opi = op->ops[oi];
        string src = get_jit_src(opi);
        op_srcs.push_back(move(src));
    }
    return OpCompiler::__get_fused_src(op->ops, op_srcs, op_members);
}

static void fix_op_member(
    const vector<Op*>& ops,
    vector<vector<string>>& op_members
) {
    // fill op member: [in0, in1, ... inN, fill, fill, out0, out1, ...]
    for (int i=0; i<ops.size(); i++) {
        auto op = ops[i];
        auto var_num = op->inputs().size() + op->outputs().size();
        auto& member = op_members.at(i);
        if (!member.size()) {
            continue;
        }
        ASSERT(member.size() <= var_num);
        while (member.size() < var_num) {
            member.insert(member.end() - op->outputs().size(), "__fill__");
        }
    }
}

string OpCompiler::__get_fused_src(
    const vector<Op*>& ops,
    const vector<string>& op_srcs,
    vector<vector<string>>& op_members
) {
    string fused_begin;
    string fused_includes;
    string fused_defines;
    string fused_kernel_args;
    string fused_kernel;
    // definitions of fused_begin
    map<string,string> defs;
    unordered_set<string> kernel_args;
    op_members = vector<vector<string>>(op_srcs.size());
    fused_begin += "#define JIT 1\n";
    defs["JIT"] = "1";
    const string pattern = "::jit_run() {";
    // TODO: better check member
    const unordered_set<string> members = {
        "x", "y", "z", "cond", "output", "extras"
    };
    const unordered_set<string> unchanged = {
        "for", "const", "auto", "get_random_engine",
        "int", "float", "bool", "CHECK", "STRINGIZE",
        "void", "__restrict__", "if", "true", "false",
        "Op", "Var", "Node", "itof", "assert", "ASSERT"
    };
    auto not_change = [&](const string& s) -> bool {
        if (unchanged.count(s)) return true;
        return (s.find("::") != string::npos) || (s.find("LOG") != string::npos);
    };
    // regex find XxxXxxOp::jit_run
    std::regex e(R"([^]*\s(\S*)Op::jit_run[^]*)");
    for (uint oi=0; oi<op_srcs.size(); oi++) {
        const string& src = op_srcs[oi];
        if (src.size()==0) continue;
        if (src.find("@relay_op") != string::npos) {
            fused_kernel += src;
            continue;
        }
        if (ops[oi]->name()==string("array")) {
            string op_name = "op" + S(oi);
            string arg_name = op_name + "_output";
            string argp_name = op_name + "_outputp";
            string T = ((ArrayOp*)ops[oi])->output->dtype().to_cstring();
            fused_kernel_args += "    ArrayOp* " + op_name + " = (ArrayOp*)(ops[" + S(oi) + "]);\n";
            // op_name = "((ArrayOp*)(ops[" + S(oi) + "]))";
            fused_kernel_args += "    Var* " + arg_name + " = " + op_name + "->output;\n";

            fused_kernel += "    auto* " + argp_name + " = " + arg_name + "->ptr<" + T + ">();\n";
            fused_kernel += "    " + argp_name + "[0] = " + op_name + "->ptr<" + T + ">()[0];\n";
            fused_kernel += "    int " + arg_name + "shape0 = 1;\n";
            fused_kernel += "    int " + arg_name + "stride0 = 1;\n";

            fused_includes += "#include \"ops/array_op.h\"\n";
            op_members[oi].push_back(arg_name);
            // auto opi = (ArrayOp*)(ops[i]);
            // auto opi_output = opi->output;
            // auto* opi_outputp = opi_output->ptr<T>();
            // opi_outputp[0] = ((T*)(opi->buffer.get()))[0];
            continue;
        }
        std::smatch cm;
        std::regex_match(src, cm, e);
        ASSERT(cm.size()>=2) << src;
        string name3 = cm[1];
        for (uint i=0; i<src.size(); i++) {
            if (src[i] == '#' &&
                (i+1<src.size() && src[i+1] == 'i') &&
                (i+2<src.size() && src[i+2] == 'n'))
            {
                // #include ...
                uint j=i+1;
                while (j<src.size() && src[j] != '\n') j++;
                if (j<src.size()) j++;
                for (uint k=i; k<j; k++) fused_includes += src[k];
                i = j-1;
                continue;
            }
            if (src[i] == '#' && (i+1<src.size() && src[i+1] == 'd')) {
                // #define aaa bbb
                // i       j  k   l
                // TODO: multi-line define
                uint j=i+1;
                while (j<src.size() && src[j] != ' ') j++;
                while (j<src.size() && src[j] == ' ') j++;
                uint k=j;
                while (k<src.size() && src[k] != ' ') k++;
                uint l=k;
                while (l<src.size() && src[l] != '\n') l++;
                if (l<src.size()) l++;
                CHECK(i<j && j<k && k<l);
                // define startswith JIT should be added at the very beginning
                if (startswith(src, "JIT", j)) {
                    string key = src.substr(j,k-j);
                    string value = src.substr(k+1, l-k-2);
                    if (defs.count(key))
                        CHECKop(defs[key],==,value);
                    else {
                        defs[key] = value;
                        fused_begin += "#define ";
                        for (; j<l; j++) fused_begin += src[j];
                    }
                    j = l;
                } else {
                    fused_defines += "#define op" + S(oi) + "_";
                    for (; j<l; j++) fused_defines += src[j];
                }
                i = j-1;
                continue;
            }
            // find the first function match the pattern "jit_run"
            bool found = true;
            for (uint j=0; j<pattern.size(); j++)
                if (pattern[j] != src[i+j]) {
                    found = false;
                    break;
                }
            if (!found) continue;
            uint j = i+pattern.size();
            uint k = j;
            int presum = 1;
            while (k<src.size() && presum) {
                if (src[k] == '}')
                    presum--;
                else if (src[k] == '{')
                    presum++;
                k++;
            }
            CHECK(presum==0) << "Jit error: braces are not matched.";
            for (;j < k-2; j++) {
                if (isvar(src[j])) {
                    uint l=j;
                    while (l<src.size() && isvar(src[l])) l++;
                    auto var = src.substr(j, l-j);
                    if (var[0] == ':' || isdigit(var[0]) || not_change(var) || src[j-1]=='.' || src[j-1]=='>') {} else
                    if (members.count(var)) {
                        string arg_name = "op" + S(oi) + "_" + var;
                        if (l<src.size() && src[l]=='[') {
                            // handle extras[...]
                            //              l   r
                            uint r = l+1;
                            while (r<src.size() && src[r]!=']') r++;
                            ASSERT(r<src.size());
                            for (uint i=l+1; i<r; i++) {
                                ASSERT(isdigit(src[i]));
                                arg_name += src[i];
                            }
                            l = r+1;
                            var = src.substr(j, l-j);
                            // arg_name = opi_extra0
                            // var = extra[0]
                        }
                        if (!kernel_args.count(arg_name)) {
                            fused_kernel_args +=
                                string("    auto ") + arg_name +
                                " = (("+name3+"Op*)(ops[" + S(oi) + "]))->" + var;
                            fused_kernel_args += ";\n";
                            kernel_args.insert(arg_name);
                            op_members[oi].push_back(arg_name);
                        }
                        fused_kernel += arg_name;
                        j = l-1;
                        continue;
                    } else
                        fused_kernel += "op" + S(oi) + "_";
                    for (uint p=j; p<l; p++) fused_kernel += src[p];
                    j = l-1;
                    continue;
                }
                fused_kernel += src[j];
            }
            break;
        }
    }
    fix_op_member(ops, op_members);
    CHECK(!(defs.count("JIT_cpu") && defs.count("JIT_cuda")))
        << "CPU op and GPU op cannot be fused together.";

    fused_kernel = fused_kernel_args + "\n" + fused_kernel;
    LOGvvvv << "Fused kernel:\n" >> fused_kernel;
    
    auto fused_src = fused_begin + fused_includes +
        "\n#include <assert.h>\n" + 
        "\n#include \"fused_op.h\"\n" + 
        fused_defines + '\n' +
        "void jittor::FusedOp::jit_run() {\n" + fused_kernel + "\n}\n";
        
    // we assume the member name is in lexicographical order
    // for (auto& v : op_members) std::sort(v.begin(), v.end());

    return fused_src;
}

string OpCompiler::get_src() {
    if (op==nullptr) return src;
    for (const auto& p : *op->loop_options)
        if (startswith(p.first, "relay")) {
            // return get jit src if has relay op
            return get_jit_src(op);
        }
    return src;
}

OpCompiler::OpCompiler(Op* op) {
    _op = op;
    this->op = op->name()==string("fused") ? (FusedOp*)op : nullptr;
    src = get_jit_src(op);
}

jit_op_entry_t OpCompiler::compile(const string& jit_key, const string& src) {
    // add extra flags for custom ops
    bool is_cuda = _op->flags.get(NodeFlags::_cuda);
    auto op_info = get_op_info(_op->name());
    return jit_compiler::compile(jit_key, src, is_cuda, op_info.extra_flags);
}

jit_op_entry_t OpCompiler::do_compile(Op* op) {
    jittor::lock_guard lg;
    OpCompiler oc(op);
    string* src = &oc.src;
    string src_after_passes;
    // if is fused op
    if (oc.op) {
        TunerManager tm(&oc);
        src_after_passes = tm.tune();
        src = &src_after_passes;
    }
    op->compile_optimize(*src);
    auto ret = oc.compile(op->get_jit_key(jk), *src);
    return ret;
}

}
