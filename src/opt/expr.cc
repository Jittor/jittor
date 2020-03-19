// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/expr.h"
#include "misc/str_utils.h"

namespace jittor {
namespace expr {
// operator precedence and associativity
// equivalence: https://en.cppreference.com/w/cpp/language/operator_precedence
// different from c++, precedence 17(,;) is right associativity

static const unordered_set<int> is_left_associativity({
    1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, /* 17 */ 
});

static const unordered_map<string, int> precedence({
    {"::", 1},

    {"++", 2}, {"--", 2},
    {"(", 2}, {")", 2}, {"()", 2},
    {"[", 2}, {"]", 2}, {"[]", 2},
    {"{", 2}, {"}", 2}, {"{}", 2},
    {".", 2}, {"->", 2}, {"<>", 2},

    {"!", 3}, {"~", 3}, 
    {"*", 5}, {"/", 5}, {"%", 5}, 
    {"+", 6}, {"-", 6}, 
    {"<<", 7}, {">>", 7}, 
    {"<=", 9}, {"<", 9}, {">=", 9}, {">", 9}, 
    {"!=", 10}, {"==", 10}, 
    {"&", 11}, 
    {"^", 12}, 
    {"|", 13}, 
    {"&&", 14}, 
    {"||", 15},

    // a @> b = !a || b
    // a @< b = a || !b
    {"@", 16}, {"@>", 16}, {"@<", 16}, 

    {"?", 26}, {":", 26},  {"?:", 26}, 
    {"=", 26}, {"+=", 26}, {"-=", 26}, 
    {"*=", 26}, {"/=", 26}, {"*=", 26}, 
    {"<<=", 26}, {">>=", 26}, {"&=", 26},  {"^=", 26},  {"|=", 26}, 

    // precedence 27 used for little higher than ","
    {",", 28}, {";", 28}
});

static const unordered_set<string> is_unary_op({
    "++", "--", "!", "~", "+", "-", "&", "*", "::"
});

static const unordered_set<string> is_left_unary_op({
    "!", "~", "+", "-", "&", "*", "::"
});


static const unordered_set<string> is_associative_op({
    "+", "*", "&", "|", "&&", "||"
});
static const unordered_set<string> is_commutative_op({
    "+", "*", "&", "|", "&&", "||"
});

static bool isvar(char x) { return isalnum(x) || x == '_'; }
static bool isempty(char x) { return x==' ' || x=='\t' || x=='\n';}

static inline int64 ex_stoll(const string& str) {
    if (startswith(str, "0x") || startswith(str, "0X"))
        return std::stoll(str,0,16);
    else if (startswith(str, "0b") || startswith(str, "0B")) 
        return std::stoll(str.substr(2),0,2);
    return std::stoll(str,0,10);
}

Expr::Expr(size_t flags, const string& str, vector<unique_ptr<Expr>>&& children)
    : flags(flags), str(str), father(0), fid(0), children(move(children)) {
    for (uint i=0; i<this->children.size(); i++) {
        this->children[i]->father = this;
        this->children[i]->fid = i;
    }
    if (is(_float)) set_data(std::stof(str)); else
    if (is(_int)) set_data(ex_stoll(str));
    maintain();
}

unique_ptr<Expr> make(const string& str, vector<unique_ptr<Expr>>&& children) {
    size_t flags = 0;
    if (is_associative_op.count(str))
        flags |= (_binary_op | _asso_op);
    if (is_commutative_op.count(str))
        flags |= (_binary_op | _comm_op);
    if (!flags) {
        if (children.size()==1) flags |= (_unary_op); else
        if (children.size()==2) flags |= (_binary_op); else
        if (children.size()==3) flags |= (_ternary_op); else
            LOGf << str << children.size();
    }
    if ((flags&_unary_op) && is_left_unary_op.count(str))
        flags |= _left_op;
    auto e = std::make_unique<Expr>(flags, str, move(children));
    return e;
}

Expr::Expr(const string& src) : flags(0) {
    vector<pair<int,int>> values;
    vector<unique_ptr<Expr>> nodes;
    vector<pair<int,int>> ops;
    vector<size_t> op_flags;

    auto comsume_op_and_var = [&](uint op_num, uint var_num) -> bool {
        if (!(ops.size() >= op_num)) return false;
        if (!(values.size() >= var_num)) return false;
        int op_pos = ops.size()>op_num ? ops[ops.size()-op_num-1].first : -1;
        if (!(op_pos < values[values.size()-var_num].first)) return false;
        int flag = op_flags.back();
        auto expression = make(flag, "");
        // is left op: ++a, --a, !a, &a
        if (ops[ops.size()-op_num].first < values[values.size()-var_num].first)
            expression->set_is(_left_op);
        for (uint i=0; i<op_num; i++) {
            auto p = ops[ops.size()-op_num+i];
            expression->str += src.substr(p.first, p.second-p.first);
        }
        for (uint i=0; i<var_num; i++) {
            expression->add_child(move(nodes[nodes.size()-var_num+i]));
        }
        auto l = values[values.size()-var_num].first;
        auto r = values[values.size()-1].second;
        for (uint i=0; i<op_num; i++) {
            ops.pop_back();
            op_flags.pop_back();
        }
        for (uint i=0; i<var_num; i++) {
            nodes.pop_back();
            values.pop_back();
        }
        expression->set_is_not(_op);
        if (var_num==1) expression->set_is(_unary_op);
        if (var_num==2) {
            expression->set_is(_binary_op);
            if (is_associative_op.count(expression->str))
                expression->set_is(_asso_op);
            if (is_commutative_op.count(expression->str))
                expression->set_is(_comm_op);
        }
        if (var_num==3) expression->set_is(_ternary_op);
        expression->maintain();
        nodes.emplace_back(move(expression));
        values.push_back({l, r});

        return true;
    };

    auto execute_back = [&](const string& op) {
        if (op==":") {
            // a?b:c
            ASSERT(ops.size()>=2);
            ASSERT(src[ops[ops.size()-2].first]=='?');
            ASSERT(values.size()>=3);
            comsume_op_and_var(2, 3);
            return;
        }
        if (comsume_op_and_var(1, 2)) return;
        ASSERT(is_unary_op.count(op)) << op << "is not unary op";
        ASSERT(comsume_op_and_var(1, 1)) << ops.size() << values.size();
    };

    auto substr = [&](const pair<int,int>& p) -> string {
        return src.substr(p.first, p.second-p.first);
    };
    vector<pair<int,int>> tokens;
    vector<size_t> flags;
    get_tokens(src, tokens, flags);

    for (uint x=0; x<tokens.size(); x++) {
        auto cp = tokens[x];
        auto op = substr(cp);
        auto flag = flags[x];
        if (!(flag & _op)) {
            values.push_back(cp);
            nodes.push_back(make(flag, op));
            continue;
        }
        string target;
        if (op == ")") target = "(";
        if (op == "]") target = "[";
        if (op == "}") target = "{";
        if (op == ">") {
            // parse template a<T>();
            auto i = tokens.at(x+1).first;
            if (src.at(i)=='(' && src.at(i+1)==')')
                target = "<";
        }
        if (target.size()) {
            int tid = ops.size()-1;
            while (tid>=0 && target != substr(ops[tid]))
                tid--;
            ASSERT(tid>=0) << "braces not match" << src;
            // a(...)
            //  ^ tpos
            // ^  bpos
            int tpos = ops[tid].first;
            int bpos = values.size()-1;
            while (bpos>=0 && values[bpos].first>=tpos) bpos--;
            bpos = bpos >= 0 ? values[bpos].first : -1;
            // find first outside braces op pos
            //   +a(...) or  a+(...)
            //   ^            ^ opos
            int opos = tid>0 ? ops[tid-1].first : -1;
            if (bpos > opos) {
                //   +a(args)
                //    ^    bpos
                //   ^     opos
                vector<unique_ptr<Expr>> args;
                while (1) {
                    auto prev_op = substr(ops.back());
                    if (prev_op == "," || prev_op == target) {
                        if (ops.back().first < values.back().first) {
                            args.push_back(move(nodes.back()));
                            nodes.pop_back();
                            values.pop_back();
                        } else
                            break;
                        if (prev_op == ",") {
                            ops.pop_back();
                            op_flags.pop_back();
                        } else
                            break;
                    } else
                        execute_back(prev_op);
                }
                ops.push_back(cp);
                op_flags.push_back(flag);
                ASSERT(comsume_op_and_var(2, 1)) << ops << op << values << nodes;
                for (uint i=0; i<args.size(); i++)
                    nodes.back()->add_child(move(args.rbegin()[i]));
                // is a function call: a(b), a[b], a{b}
                nodes.back()->set_is(_call);
            } else {
                // not a function call
                // a+(...)
                while ((int)ops.size()>tid+1) {
                    auto prev_op = substr(ops.back());
                    execute_back(prev_op);
                }
                // pop left braces
                ops.pop_back();
                op_flags.pop_back();
            }
            continue;
        }
        int pd = precedence.at(op);
        bool is_left = is_left_associativity.count(pd);
        while (ops.size()) {
            auto prev_op = src.substr(ops.back().first, ops.back().second - ops.back().first);
            if (prev_op == "(" || prev_op == "[" || prev_op == "{") break;
            auto ppd = precedence.at(prev_op);
            if (ppd < pd || (ppd==pd && is_left)) {
                execute_back(prev_op);
            } else {
                break;
            }
        }
        ops.push_back(cp);
        op_flags.push_back(flag);
    }
    while (ops.size()) {
        auto prev_op = substr(ops.back());
        execute_back(prev_op);
    }

    ASSERT(nodes.size() == 1) << "Left multiple nodes:" << nodes;
    move_from(nodes[0]);
}

string Expr::to_string(int try_reduce_braces, int debug) const {
    std::stringstream ss;
    int pd = try_reduce_braces?100:-1;
    to_string(ss, pd, pd, debug);
    return ss.str();
}

int64 Expr::as_int() const {
    if (is(_float))
        return int64(as_float());
    ASSERT(is(_int));
    return data.i;
}

float64 Expr::as_float() const {
    if (is(_int))
        return float64(as_int());
    ASSERT(is(_float));
    return data.f;
}

void get_tokens(
    const string& src,
    vector<pair<int,int>>& tokens,
    vector<size_t>& flags
) {
    int end = src.size();
    while (end && (src[end-1]==' ' || src[end-1]=='\n' || src[end-1]==';'))
        end--;
    for (uint i=0; i<end;) {
        while (i<src.size() && isempty(src[i])) i++;
        if (i>=src.size()) break;
        size_t flag=0;
        uint j = i+1;
        if (src[i]=='\'' || src[i]=='\"') {
            while (j<src.size() && (src[j]!=src[i])) {
                if (src[j]=='\\') j++;
                j++;
            }
            ASSERT(j<src.size()) << "string or char not end.";
            j++;
            if (src[i]=='\'')
                flag |= _char;
            else
                flag |= _string;
        } else
        if (isdigit(src[i])) {
            while (j<src.size() && (isalnum(src[j]) || src[j]=='.')) {
                if (src[j]=='x' || src[j]=='b') {
                    if (!(flag & _float)) flag |= _int;
                } else
                if (src[j]=='.' || src[j]=='f' || src[j]=='e') {
                    if (!(flag & _int)) flag |= _float;
                }
                j++;
            }
            if (!(flag & _float)) flag |= _int;
        } else
        if (isvar(src[i])) {
            while (j<src.size() && isvar(src[j])) j++;
        } else {
            while (j<src.size() && !isvar(src[j]) && !isempty(src[j]) &&
                precedence.count(src.substr(i, j-i+1))) j++;
            if (src[i]=='(' && src[j-1]==')') j--;
            if (src[i]=='<' && src[j-1]=='>') j--;
            if (src[i]=='[' && src[j-1]==']') j--;
            if (src[i]=='{' && src[j-1]=='}') j--;
            if (src[j-1]=='@' && j<src.size() && isvar(src[j])) {
                while (j<src.size() && isvar(src[j])) j++;
            } else
                flag |= _op;
        }
        tokens.push_back({i, j});
        flags.push_back(flag);
        i = j;
    }
}

static int64 eval_binary_int(const string& op, int64 a, int64 b) {
#define m(o) if (op == #o) return int64(a o b);
    m(+) m(-) m(*) m(/) m(%)
    m(<<) m(>>) m(<=) m(<) m(>=)
    m(>) m(!=) m(==)
    m(&) m(^) m(|) m(&&) m(||)
#undef m
    if (op==",") return b;
    LOGf << "Op" << op << a << b << "not support";
    return 0;
}

static float64 eval_binary_float(const string& op, float64 a, float64 b) {
#define m(o) if (op == #o) return float64(a o b);
    m(+) m(-) m(*) m(/)
    m(<=) m(<) m(>=)
    m(>) m(!=) m(==)
    m(&&) m(||)
#undef m
    if (op==",") return b;
    LOGf << "Op" << op << a << b << "not support";
    return 0;
}

static int64 eval_unary_left_int(const string& op, int64 a) {
#define m(o) if (op == #o) return int64(o a);
    m(+) m(-) m(!) m(~)
#undef m
    LOGf << "Op" << op << a << "not support";
    return 0;
}

static float64 eval_unary_left_float(const string& op, float64 a) {
#define m(o) if (op == #o) return float64(o a);
    m(+) m(-) m(!)
#undef m
    LOGf << "Op" << op << a << "not support";
    return 0;
}

static void _eval(Expr* e) {
    auto& c = e->children;
    string op = move(e->str);
    if (e->is(_binary_op)) {
        ASSERT(c.size()==2);
        if (c[0]->is(_float) | c[1]->is(_float)) {
            e->set_is_only(_float);
            e->set_data(eval_binary_float(op, c[0]->as_float(), c[1]->as_float()));
        } else {
            e->set_is_only(_int);
            e->set_data(eval_binary_int(op, c[0]->as_int(), c[1]->as_int()));
        }
    } else
    if (e->is(_ternary_op)) {
        ASSERTop(op,==,"?:");
        if (c[1]->is(_float) | c[2]->is(_float)) {
            e->set_is_only(_float);
            e->set_data(c[0]->as_int() ? c[1]->as_float() : c[2]->as_float());
        } else {
            e->set_is_only(_int);
            e->set_data(c[0]->as_int() ? c[1]->as_int() : c[2]->as_int());
        }
    } else {
        ASSERTop(c.size(),==,1);
        ASSERT(e->is(_left_op));
        if (c[0]->is(_float))
            e->set_data(eval_unary_left_float(op, c[0]->as_float()));
        else
            e->set_data(eval_unary_left_int(op, c[0]->as_int()));
        e->set_is_only(c[0]->flags);
    }
}

static void eval_asso_binary(Expr* e) {
    vector<unique_ptr<Expr>> nc;
    nc.reserve(e->children.size());
    for (uint i=0; i<e->children.size(); i++) {
        auto& c = e->children[i];
        if (!nc.size() || nc.back()->is_not(_number) || c->is_not(_number)) {
            nc.push_back(move(c));
            continue;
        }
        auto& b = nc.back();
        if (b->is(_float) | c->is(_float)) {
            b->set_data(eval_binary_float(e->str, b->as_float(), c->as_float()));
            b->set_is_only(_float);
        } else {
            b->set_data(eval_binary_int(e->str, b->as_int(), c->as_int()));
            b->set_is_only(_int);
        }
    }
    e->children.clear();
    if (nc.size()==1) {
        e->move_from(nc.back());
        return;
    }
    e->insert(0, nc);
    
    // eval x*0 -> 0
    if (e->str=="*") {
        for (auto& c : e->children) {
            if (c->is(_number) && c->data.i==0) {
                e->swap(c->clone().get());
                return;
            }
        }
    }
}

pair<int64,int64> get_zero_elem(const string& op) {
    if (op=="+") return {1,0};
    if (op=="-") return {1,0};
    if (op=="*") return {1,1};
    if (op=="/") return {1,1};
    return {0,0};
}

unique_ptr<Expr> Expr::eval() {
    auto a = make(flags, str);
    if (is(_op)) {
        a->children.reserve(children.size());
        auto p = get_zero_elem(str);
        bool can_eval = true;
        for (auto& c : children) {
            a->children.push_back(c->eval());
            auto& x = a->children.back();
            if (!x->is(_number))
                can_eval = false;
            if (x->is(_int)) {
                if (p.first && p.second == x->as_int()) {
                    if (a->children.size()>1 || a->is(_asso_op))
                        a->children.pop_back();
                }
            }
        }
        if (a->is(_asso_op)) {
            if (a->children.size()==0) {
                a->children.push_back(make(S(p.second)));
            }
            eval_asso_binary(a.get());
            if (a->children.size()==1) {
                return move(a->children.back());
            }
            return a;
        }
        if (can_eval) {
            _eval(a.get());
            a->children.clear();
            return a;
        }
        if (a->children.size()==1 && p.first) {
            return move(a->children.back());
        }
    } else {
        a->data.i = data.i;
    }
    return a;
}


unique_ptr<Expr> Expr::assign_symbol(const unordered_map<string,string>& symbols) {
    auto a = clone();
    a->dfs([&](Expr* e) {
        if (!e->is_sym()) return;
        auto iter = symbols.find(e->str);
        if (iter == symbols.end()) return;
        e->swap(make(iter->second).get());
    });
    return a;
}

unique_ptr<Expr> Expr::simplify() {
    auto e = eval();
    return e;
}


std::ostream& operator<<(std::ostream& os, const Flags& f) {
    #define m(x) if (f & x) os << "is" #x << ",";
    m(_unary_op);
    m(_binary_op);
    m(_ternary_op);
    m(_op);
    m(_call);
    m(_left_op);
    m(_char);
    m(_string);
    m(_int);
    m(_float);
    m(_number);
    #undef m
    return os;
}

void Expr::move_from(unique_ptr<Expr>& e) {
    flags=e->flags;
    str=move(e->str);
    children = move(e->children);
    for (uint i=0; i<children.size(); i++)
        children[i]->father = this;
    data.i = e->data.i;
    e = nullptr;
}

void Expr::to_string(std::ostream& os, int olp, int orp, int debug) const {
    if (is_not(_op)) {
        // TODO: negtive value need braces to protect -
        bool need_bc = is(_number) && as_float()<0;
        if (need_bc) os << '(';
        if (is(_int)) os << as_int(); else
        if (is(_float)) os << as_float(); else
        os << str;
        if (need_bc) os << ')';
        return;
    }
    string s;
    bool need_bc = 1;
    int pd = olp;
    if (olp>=0) {
        pd = precedence.at(str);
        bool is_left = is_left_associativity.count(pd);
        bool check_left = pd < olp || (pd==olp && !is_left);
        bool check_right = pd < orp || (pd==orp && is_left);
        need_bc = !(check_left && check_right);
    }
    if (need_bc)
        os << "(";
    if (debug) {
        os << "/*f:";
        os << (Flags)flags;
        os << ";s:";
        os << str;
        os << ";c:";
        os << children.size();
        os << "*/";
    }
    if (is(_ternary_op)) {
        // a?b:c
        ASSERT(children.size()==3 && str=="?:");
        children[0]->to_string(os, olp, pd, debug);
        os << "?";
        children[1]->to_string(os, pd, pd, debug);
        os << ":";
        children[2]->to_string(os, pd, orp, debug);
    } else if (is(_call)) {
        // a(b,c,d)
        ASSERT(children.size() && str.size()==2);
        children[0]->to_string(os, olp, pd, debug);
        os << str[0];
        for (uint i=1; i<children.size(); i++) {
            // precedence 27 used for little higher than ","
            // make a(x,(y,z)) correct
            int npd = orp<0 ? -1 : 27;
            children[i]->to_string(os, npd, npd, debug);
            if (i+1 == children.size())
                os << str[1];
            else
                os << ",";
        }
        if (children.size()==1) os << str[1];
    } else if (is(_left_op)) {
        // ++a, --a
        os << str;
        ASSERT(children.size()==1) << (print_trace(), 0);
        children[0]->to_string(os, pd, orp, debug);
    } else {
        // a--, a+b
        ASSERT(children.size());
        ASSERT(children.size()>=2 || is(_unary_op) || is(_asso_op)) << str << children;
        children[0]->to_string(os, olp, pd, debug);
        if (is(_unary_op)) os << str;
        else {
            for (uint i=1; i<children.size()-1; i++) {
                os << str;
                children[i]->to_string(os, pd, pd, debug);
            }
            if (children.size()>1) {
                os << str;
                children.back()->to_string(os, pd, orp, debug);
            }
        }
    }
    if (need_bc)
        os << ")";
}

std::ostream& operator<<(std::ostream& os, const Expr& expression) {
    return os << expression.to_string();
}

unique_ptr<Expr> make(size_t flags, const string& str, vector<unique_ptr<Expr>>&& children) {
    unique_ptr<Expr> e(new Expr(flags, str, move(children)));
    return e;
}

void Expr::add_child(unique_ptr<Expr>&& c) {
    c->father = this;
    c->fid = children.size();
    children.push_back(move(c));
}

unique_ptr<Expr> Expr::move_out() {
    ASSERT(father);
    auto& fc = father->children;
    unique_ptr<Expr> e = move(fc[fid]);
    fc.erase(fc.begin()+fid);
    for (uint i=fid; i<fc.size(); i++)
        fc[i]->fid = i;
    father = nullptr;
    fid = 0;
    return e;
}

void Expr::swap(Expr* e) {
    std::swap(flags, e->flags);
    std::swap(str, e->str);
    std::swap(father, e->father);
    std::swap(fid, e->fid);
    std::swap(data, e->data);
    std::swap(children, e->children);
}

void Expr::erase() {
    move_out();
}

unique_ptr<Expr> Expr::clone() {
    auto e = make(flags, str);
    e->data.i = data.i;
    e->children.reserve(children.size());
    for (auto& c : children)
        e->add_child(c->clone());
    return e;
}

void Expr::insert(int pos, vector<unique_ptr<Expr>>& v) {
    children.insert(
        children.begin()+pos,
        make_move_iterator(v.begin()),
        make_move_iterator(v.end())
    );
    for (uint i=pos; i<children.size(); i++) {
        children[i]->father = this;
        children[i]->fid = i;
    }
}

vector<unique_ptr<Expr>> Expr::move_out(int start, int end) {
    if (end<=0) end += children.size();
    vector<unique_ptr<Expr>> v;
    v.reserve(end-start);
    for (int i=start; i<end; i++) {
        v.push_back(move(children[i]));
        v.back()->father = nullptr;
        v.back()->fid = 0;
    }
    children.erase(children.begin()+start, children.begin()+end);
    for (uint i=end; i<children.size(); i++)
        children[i]->fid = i;
    return v;
}

void Expr::collapse_children(uint& cid) {
    auto c = children[cid].get();
    auto v = c->move_out(0);
    auto ncid = cid + v.size() - 1;
    children.erase(children.begin()+cid);
    insert(cid, v);
    cid = ncid;
}

void Expr::maintain() {
    if (is(_asso_op)) {
        // a+(b+c) -> a+b+c
        for (uint i=0; i<children.size(); i++) {
            if (children[i]->is(_asso_op) && children[i]->str==str) {
                collapse_children(i);
            }
        }
    }
}

static void rule_minus(Expr* e) {
    if (e->is(_unary_op)) {
        // -a -> (-1)*a
        e->move_from(make_op("*",
            make(_int, "-1"),
            e->children[0]
        ));
    } else {
        // a-b -> a+(-1)*b
        auto c = e->move_out(0);
        e->move_from(make_op("+",
            c[0],
            make_op("*",
                make(_int, "-1"),
                c[1]
            )
        ));
    }
}

static bool rule_not(Expr* e) {
    ASSERT(e->children.size()==1);
    auto& c = e->children[0];
    // !var not change
    if (c->is_var()) return false;
    if (c->str == "&&" || c->str=="||") {
        // !(a&&b) -> !a || !b
        // !(a||b) -> !a && !b
        vector<unique_ptr<Expr>> cc(c->children.size());
        for (uint i=0; i<c->children.size(); i++) {
            cc[i] = make_op("!", move(c->children[i]));
        }
        e->move_from(make(c->str[0]=='&'?"||":"&&", move(cc)));
        return true;
    }
    if (c->str == "!") {
        // !!a -> a
        ASSERT(c->children.size()==1);
        e->move_from(c->children[0]->move_out());
        if (e->str == "!")
            rule_not(e);
        return true;
    }
    static const unordered_map<string,string> nmap = {
        {"<",">="}, {"<=",">"}, {">","<="}, {">=","<"}, 
        {"==","!="}, {"!=","=="}
    };
    auto iter = nmap.find(c->str);
    if (iter != nmap.end()) {
        // !(a<b) -> a>=b
        e->move_from(c->move_out());
        e->str = iter->second;
        return true;
    }
    return false;
}

static void rule_mul(Expr* e, const string& add="+") {
    string mul = e->str;
    if (e->is(_binary_op)) {
        // (a+b)*(c+d) -> a*c + a*d + b*c + b*d
        vector<int> add_index, add_range, add_cid;
        for (uint i=0; i<e->children.size(); i++) {
            auto c = e->children[i].get();
            if (c->str==add && c->is(_binary_op)) {
                add_cid.push_back(add_range.size());
                add_range.push_back(c->children.size());
                add_index.push_back(0);
            } else
                add_cid.push_back(-1);
        }
        if (!add_range.size()) return;
        vector<unique_ptr<Expr>> nc;
        int n = add_index.size();
        while (1) {
            vector<unique_ptr<Expr>> nm;
            for (uint i=0; i<e->children.size(); i++) {
                auto c = e->children[i].get();
                if (add_cid[i] == -1)
                    nm.emplace_back(c->clone());
                else
                    nm.emplace_back(c->children[add_index[add_cid[i]]]->clone());
            }
            nc.emplace_back(make(mul, move(nm)));
            int p = n-1;
            add_index[p]++;
            while (add_index[p] >= add_range[p]) {
                add_index[p] = 0;
                p--;
                if (p<0) break;
                add_index[p]++;
            }
            if (p<0) break;
        }
        e->move_from(make(add, move(nc)));
    }
}

static void rule_at(Expr* e) {
    if (e->str == "@>") {
        // a @> b = !a || b
        e->move_from(make_op("||",
            make_op("!", e->children[0]),
            e->children[1]
        ));
    } else
    if (e->str == "@<") {
        // a @< b = !a || b
        e->move_from(make_op("||",
            make_op("!", e->children[1]),
            e->children[0]
        ));
    }
}

static void rule_cmp(Expr* e) {
    auto b = e->children[1]->move_out();
    auto a = e->children[0]->move_out();
    auto a2 = a->clone();
    auto b2 = b->clone();
    if (e->str == "==") {
        // a==b -> a>=b&&a<=b
        e->move_from(make_op("&&",
            make_op(">=", a, b),
            make_op("<=", a2, b2)
        ));
    } else {
        // a!=b -> a<b||a>b
        e->move_from(make_op("||",
            make_op("<", a, b),
            make_op(">", a2, b2)
        ));
    }
}

unique_ptr<Expr> expand(Expr* e) {
    auto h = e->clone();
    e = h.get();
    uint cid = 0;
    // while loop dfs
    while (1) {
        if (cid==0) {
            // first enter
            if (e->str == "-") {
                rule_minus(e);
            } else
            if (e->str == "!") {
                if (rule_not(e)) continue;
            } else
            if (e->str.size() && e->str[0] == '@') {
                rule_at(e);
            } else
            if (e->str=="==" || e->str=="!=") {
                rule_cmp(e);
            }
        }
        if (cid>=e->children.size()) {
            // before return
            e->maintain();
            if (e->str == "*") {
                rule_mul(e);
            } else
            if (e->str == "&&") {
                rule_mul(e, "||");
            }
            // return to father
            cid = e->fid;
            // auto c = e;
            e = e->father;
            if (!e) break;
            // back from child
            cid ++;
            continue;
        }
        // recursive to child
        e = e->children[cid].get();
        cid = 0;
    }
    return h;
}

bool match(Expr* src, Expr* target) {
    vector<unique_ptr<Expr>> results;
    return match(src, target, {}, {}, results);
}

bool match(
    Expr* src, Expr* target,
    const vector<string>& solve_symbols,
    const vector<string>& exclude_symbols,
    vector<unique_ptr<Expr>>& results
) {
    auto s = src->expand()->simplify();
    auto t = target->expand()->simplify();
    int n = solve_symbols.size();
    unordered_map<string,int> ss;
    for (int i=0; i<n; i++)
        ss[solve_symbols[i]] = i;
    unordered_set<string> es(exclude_symbols.begin(), exclude_symbols.end());

    auto solve_id = [&](Expr* e) -> int {
        if (!e->is_sym()) return -1;
        auto iter = ss.find(e->str);
        if (iter == ss.end()) return -1;
        return iter->second;
    };

    std::function<bool(Expr*)> has_exclude = [&](Expr* e) -> bool {
        if (e->is_sym()) return es.count(e->str);
        if (e->is(_op)) {
            for (auto& c : e->children)
                if (has_exclude(c.get()))
                    return true;
        }
        return false;
    };

    std::function<bool(Expr*, Expr*, vector<unique_ptr<Expr>>&)> log_do_match;

    std::function<bool(Expr*, Expr*, vector<unique_ptr<Expr>>&)> do_match = 
    [&](Expr* s, Expr* t, vector<unique_ptr<Expr>>& results) -> bool {
        if (t->is_not(_op)) {
            int tid = solve_id(t);
            // if is a symbol need to solve
            if (tid>=0) {
                if (has_exclude(s))
                    return false;
                results[tid] = s->clone();
                return true;
            } else {
                return s->flags == t->flags && s->to_string()==t->to_string();
            }
        } else {
            auto ze = get_zero_elem(t->str);
            if (s->is_not(_op)) {
                // if op don't have zero element
                if (!ze.first)
                    return false;
            } else
                if (s->flags != t->flags || s->str != t->str)
                    return false;
            if (!ze.first && s->children.size() != t->children.size())
                return false;
            int n = s->children.size();
            int m = t->children.size();
            unique_ptr<Expr> zep;
            if (ze.first) {
                zep = make(S(ze.second));
                n++;
            }
            std::function<bool(int,int)> check_match = 
            [&](int is, int it) -> bool {
                vector<unique_ptr<Expr>> ns(results.size());
                Expr* sp = zep.get();
                if (s->is_not(_op) && is==0) {
                    sp = s;
                } else
                if (is<(int)s->children.size())
                    sp = s->children[is].get();
                if (!log_do_match(sp, t->children[it].get(), ns)) {
                    return false;
                }
                for (uint j=0; j<ns.size(); j++) {
                    if (results[j]) {
                        if (ns[j]) {
                            // if both solved but not matched
                            if (!match(results[j].get(), ns[j].get()))
                                return false;
                        }
                    } else
                    if (ns[j])
                        results[j] = move(ns[j]);
                }
                return true;
            };
            int asso_wildcard_id = -1, asso_wildcard_tid=0;
            if (s->is(_asso_op)) {
                // wildcard assosiative id
                // a*b+c <----
                for (int i=m-1; i>=0; i--) {
                    auto tid = solve_id(t->children.at(i).get());
                    asso_wildcard_tid = tid;
                    if (tid>=0) {
                        // matched by other expr
                        if (results[tid])
                            continue;
                        LOGvvvv << "check asso wild" << t->children.at(i) << *t;
                        bool in_other_target = 0;
                        for (int j=0; j<m; j++) {
                            if (j==i) continue;
                            t->children.at(j)->dfs([&](Expr* e) {
                                if (!e->is_sym()) return;
                                if (e->str == t->children.at(i)->str)
                                    in_other_target = 1;
                            });
                            if (in_other_target) break;
                        }
                        if (!in_other_target) {
                            asso_wildcard_id = i;
                            break;
                        }
                    }
                }
                LOGvvvv << "asso_wildcard_id" << asso_wildcard_id <<
                    t->children.at(asso_wildcard_id);
                // asso_wildcard_id = -1;
            }
            if (s->is(_comm_op)) {
                // is commutative op, children can be matched in any order
                vector<bool> is_matched(m);
                for (int i=0; i<n; i++) {
                    bool matched = false;
                    for (int j=0; j<m; j++) {
                        if (j==asso_wildcard_id) continue;
                        if (is_matched[j]) continue;
                        if (check_match(i, j)) {
                            is_matched[j] = true;
                            matched = true;
                            break;
                        }
                    }
                    for (int _=0; _<results.size(); _++)
                        LOGvvvv << "results[" >> solve_symbols[_] >> "]=" >>
                            (results[_]?results[_]->to_string(1):"null");
                    if (!matched && asso_wildcard_id>=0) {
                        auto j = asso_wildcard_id;
                        auto& res = results[asso_wildcard_tid];
                        // if zero elem and results already matched
                        if (i==(int)s->children.size() && res)
                            continue;
                        // match a+b -> c
                        auto bk = move(res);
                        if (!check_match(i, j)) {
                            return false;
                        }
                        if (bk)
                            res = make_op(s->str, move(bk), move(res));
                        continue;
                    }
                    // if not matched and not zero elem
                    if (!matched && i<(int)s->children.size()) {
                        return false;
                    }
                }
                for (int j=0; j<is_matched.size(); j++)
                    if (j!=asso_wildcard_id && !is_matched[j])
                        return false;
                return true;
            } else {
                // not a commutative op, match in the same order
                for (int i=0; i<m; i++) {
                    if (!check_match(i,i))
                        return false;
                }
                return true;
            }
        }
        return false;
    };

    int depth = 0;
    log_do_match = 
    [&](Expr* s, Expr* t, vector<unique_ptr<Expr>>& results) -> bool {
        LOGvvvv >> string(depth*4, ' ') >>
            "match" << s->to_string(1) << t->to_string(1);
        depth++;
        auto res = do_match(s, t, results);
        depth--;
        LOGvvvv >> string(depth*4, ' ') >>
            "return=" >> res << s->to_string(1) << t->to_string(1);
        return res;
    };

    results.clear();
    results.resize(n);
    if (!log_do_match(s.get(), t.get(), results))
        return false;
    for (int i=0; i<n; i++) {
        if (!results[i]) {
            LOGvvvv << "unsolved symbol" << solve_symbols[i];
            return false;
        }
    }
    return true;
}


} // expr
} // jittor
