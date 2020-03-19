// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {
namespace expr {
// Expression
enum Flags {
    // is op or is a symbol
    _unary_op = 1<<7,
    _binary_op = 1,
    _ternary_op = 1<<8,
    _op = _unary_op | _binary_op | _ternary_op,
    // is a function call: a(b), a[b], a{b}
    _call = 1<<1,
    // is left op: ++a, --a, !a, &a
    _left_op = 1<<2,
    // is associative op: (a+b)+c -> a+(b+c)
    _asso_op = 1<<9,
    // is commutative op: a*b -> b*a
    _comm_op = 1<<10,
    // 'a'
    _char = 1<<3,
    // "string"
    _string = 1<<4,
    // int: 1, 0x1a, 0b1, 1u, 1ull
    _int = 1<<5,
    // float: 1. 1.0f 1e3
    _float = 1<<6,
    _number = _int | _float,
};

struct Expr {
    size_t flags;
    string str;
    Expr* father;
    // index in father's children
    size_t fid;
    // data for number
    union Data {
        int64 i;
        float64 f;
    } data;
    vector<unique_ptr<Expr>> children;

    Expr(const string& src);
    Expr(size_t flags, const string& str, vector<unique_ptr<Expr>>&& children);

    void add_child(unique_ptr<Expr>&& c);
    unique_ptr<Expr> move_out();
    vector<unique_ptr<Expr>> move_out(int start, int end=0);
    void insert(int pos, vector<unique_ptr<Expr>>& v);
    void erase();
    void swap(Expr* e);

    template <typename Func>
    void dfs(Func&& func) {
        func(this);
        for (auto& c : children)
            c->dfs(func);
    }

    int64 as_int() const;
    float64 as_float() const;
    inline void set_data(int64 x) { data.i = x; str=S(x);}
    inline void set_data(float64 x) { data.f = x; str=S(x); }

    unique_ptr<Expr> assign_symbol(const unordered_map<string,string>& symbols);

    // to_string: return expression string of this expression
    // args:
    //   try_reduce_braces: try to reduce brances if precedence correct
    //     for example:
    //     a+(b*c) -> a+b*c
    //     (a,(b,c)) -> a,b,c
    //   debug: output debug info in comment
    //     example: /*f:{flags};s:{str};c:{children.size()}*/
    //   return: expression string of this expression
    string to_string(int try_reduce_braces=false, int debug=false) const;
    // args:
    //     olp: outside left precedence, -1 for force add braces
    //     orp: outside right precedence, -1 for force add braces
    void to_string(std::ostream& os, int olp, int orp, int debug=false) const;

    // collapse children of cid-th child into father's children
    // a+(b+c) -> a+b+c
    void collapse_children(uint& cid);

    inline unique_ptr<Expr> expand();
    unique_ptr<Expr> eval();
    unique_ptr<Expr> simplify();
    unique_ptr<Expr> clone();
    void move_from(unique_ptr<Expr>& e);
    inline void move_from(unique_ptr<Expr>&& e) { move_from(e); };

    void maintain();

    inline bool is(size_t f) const { return flags & f;}
    inline bool is_not(size_t f) const  { return !(flags & f);}
    inline void set_is(size_t f) { flags |= f;}
    inline void set_is_not(size_t f) { flags &= ~f;}
    inline void set_is_only(size_t f) { flags = f;}
    inline bool is_sym() const { return is_not(_op | _char | _string | _number); }
    inline bool is_var() const { return is_not(_op); }
};

std::ostream& operator<<(std::ostream& os, const Expr& expression);
std::ostream& operator<<(std::ostream& os, const Flags& f);


inline unique_ptr<Expr> make(const string& str) { return std::make_unique<Expr>(str); };
unique_ptr<Expr> make(size_t flags, const string& str, vector<unique_ptr<Expr>>&& children={});
unique_ptr<Expr> make(const string& str, vector<unique_ptr<Expr>>&& children);
template<class... Args>
unique_ptr<Expr> make_op(const string& str, Args&&... args);

/* Match between source expression and target expression, try to solve symbols
arguments:
    src: source expression
    target: target expression
    solve_symbols: symbols in target expression which need to be solved
    exclude_symbols: symbols that should not occur in results
    results: same length with solve_symbols, return the solved symbols
    return: return true if solved success
example:
    auto src = make("3*i+j-1");
    auto target = make("i*stride+pad+j");
    vector<unique_ptr<Expr>> results;
    match(src.get(), target.get(), {"stride", "pad"}, {"i", "j"}, results);
    LOGv << results;
    // print [3,-1]
 */
bool match(
    Expr* src, Expr* target,
    const vector<string>& solve_symbols,
    const vector<string>& exclude_symbols,
    vector<unique_ptr<Expr>>& results
);

bool match(Expr* src, Expr* target);

void get_tokens(
    const string& src,
    vector<pair<int,int>>& tokens,
    vector<size_t>& flags
);

unique_ptr<Expr> expand(Expr* e);
inline unique_ptr<Expr> Expr::expand() { return expr::expand(this); }

template<class... Args>
unique_ptr<Expr> make_op(const string& str, Args&&... args) {
    vector<unique_ptr<Expr>> children;
    children.reserve(sizeof...(args));
    auto f = [&](unique_ptr<Expr>& c) { children.emplace_back(move(c)); };
    // Brace-enclosed initializers
    int dummy[] = {(f(args), 0)...};
    (void)dummy;
    return make(str, move(children));
}

template <typename Func>
void dfs(Expr* e, Func&& f) {
    f(e);
    for (auto& c : e->children)
        dfs(c.get(), f);
}

} // expr
} // jittor

